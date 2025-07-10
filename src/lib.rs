use std::{
    cell::{RefCell, RefMut},
    collections::HashMap,
    hash::{BuildHasher, Hash, Hasher},
    ops::Range,
};

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use pcre2::bytes::{Regex, RegexBuilder};
use rayon::prelude::*;
use thiserror::Error;

#[cfg(feature = "pyo3")]
pub mod bindings;
pub mod loader;

#[cfg(feature = "pyo3")]
use pyo3::{prelude::*, types::PyModule};

pub type Rank = u32;

type EncoderMap = HashMap<Word, Rank, NoOpHasher>;
type DecoderMap = HashMap<Rank, Vec<u8>, NoOpHasher>;

pub struct Token {
    pub bytes: Vec<u8>,
    pub rank: Rank,
}

pub struct Tokenizer {
    encoder: EncoderMap,
    special_encoder: EncoderMap,
    decoder: DecoderMap,
    special_tokens_decoder: DecoderMap,
    splitters: Vec<Splitter>,
}

impl Tokenizer {
    fn from_vocab_and_regex(
        vocab: Vec<Token>,
        special_vocab: Vec<Token>,
        regex_patterns: Vec<impl AsRef<str>>,
    ) -> Result<Self, Error> {
        let special_tokens_matcher = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(
                special_vocab
                    .iter()
                    .map(|v| v.bytes.as_slice())
                    .collect::<Vec<_>>(),
            )?;

        let regexes = regex_patterns
            .iter()
            .map(|pattern| {
                Ok(Splitter::Regex(
                    RegexBuilder::new()
                        .jit_if_available(true)
                        .max_jit_stack_size(Some(1024 * 1024))
                        .utf(true)
                        .ucp(true)
                        .build(pattern.as_ref())?,
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let mut splitters = vec![Splitter::AhoCorasick(special_tokens_matcher)];
        splitters.extend(regexes);

        let mut encoder = EncoderMap::default();
        let mut decoder = DecoderMap::default();

        for item in vocab {
            encoder.insert(Word::from_bytes(&item.bytes), item.rank);
            decoder.insert(item.rank, item.bytes);
        }

        let mut special_encoder = EncoderMap::default();
        let mut special_tokens_decoder = DecoderMap::default();

        for item in special_vocab {
            special_encoder.insert(Word::from_bytes(&item.bytes), item.rank);
            special_tokens_decoder.insert(item.rank, item.bytes);
        }

        Ok(Self {
            encoder,
            special_encoder,
            decoder,
            special_tokens_decoder,
            splitters,
        })
    }

    pub fn decode(&self, tokens: &[Rank]) -> Result<Vec<&[u8]>, Error> {
        tokens
            .par_iter()
            .try_fold(Vec::new, |mut acc, token| {
                if let Some(bytes) = self.decoder.get(token) {
                    acc.push(bytes.as_slice());
                } else if let Some(bytes) = self.special_tokens_decoder.get(token) {
                    acc.push(bytes.as_slice());
                } else {
                    return Err(Error::InvalidToken(*token));
                }

                Ok(acc)
            })
            .try_reduce(Vec::new, |mut a, b| {
                a.extend(b);
                Ok(a)
            })
    }

    pub fn decode_batch<T, I>(&self, tokens: T) -> Result<Vec<Vec<&[u8]>>, Error>
    where
        T: IntoParallelIterator<Item = I>,
        I: AsRef<[Rank]>,
    {
        tokens
            .into_par_iter()
            .map(|token| self.decode(token.as_ref()))
            .collect::<Result<Vec<_>, Error>>()
    }

    thread_local! {
        static SPLIT_SHARED_BUFFER: Buffer<Split> = Buffer(RefCell::new(Vec::new()));
        static RANK_SHARED_BUFFER: Buffer<Rank> = Buffer(RefCell::new(Vec::new()));
    }

    pub fn encode(&self, text: &[u8]) -> Result<Vec<Rank>, Error> {
        let result = self
            .splitters
            .iter()
            .try_fold(
                {
                    let mut splits = Vec::with_capacity(text.len() / 4);
                    splits.push(Split::Bytes(0..text.len()));
                    splits
                },
                |splits, splitter| {
                    splits
                        .into_par_iter()
                        .try_fold(
                            || {
                                Self::SPLIT_SHARED_BUFFER.with(|buffer| {
                                    buffer.prepare(0);
                                    buffer.take_buffer()
                                })
                            },
                            |mut acc, split| {
                                match split {
                                    Split::Bytes(r) => splitter.split(&text[r], &mut acc)?,
                                    literal => acc.push(literal),
                                }

                                Ok::<_, Error>(acc)
                            },
                        )
                        .try_reduce(Vec::new, |mut a, b| {
                            a.extend_from_slice(&b);

                            Self::SPLIT_SHARED_BUFFER.with(|buffer| buffer.return_buffer(b));

                            Ok(a)
                        })
                },
            )?
            .into_par_iter()
            .try_fold(
                || {
                    Self::RANK_SHARED_BUFFER.with(|buffer| {
                        buffer.prepare(text.len() / 4);
                        buffer.take_buffer()
                    })
                },
                |mut acc, split| {
                    match split {
                        Split::Literal(r) => {
                            let bytes = &text[r];
                            if let Some(rank) = self.special_encoder.get(&Word::from_bytes(bytes)) {
                                acc.push(*rank);
                            } else if let Some(rank) = self.encoder.get(&Word::from_bytes(bytes)) {
                                acc.push(*rank);
                            } else {
                                return Err(Error::NoValidToken(
                                    String::from_utf8_lossy(bytes).to_string(),
                                ));
                            }
                        }
                        Split::Bytes(r) => {
                            let bytes = &text[r];
                            if let Some(rank) = self.encoder.get(&Word::from_bytes(bytes)) {
                                acc.push(*rank);
                            } else {
                                self.bpe_merge(bytes, &mut acc)?;
                            }
                        }
                    }

                    Ok(acc)
                },
            )
            .try_reduce(Vec::new, |mut a, b| {
                a.extend_from_slice(&b);

                Self::RANK_SHARED_BUFFER.with(|buffer| buffer.return_buffer(b));

                Ok(a)
            })?;

        Ok(result)
    }

    pub fn encode_batch<T, I>(&self, texts: T) -> Result<Vec<Vec<Rank>>, Error>
    where
        T: IntoParallelIterator<Item = I>,
        I: AsRef<[u8]>,
    {
        texts
            .into_par_iter()
            .map(|text| self.encode(text.as_ref()))
            .collect()
    }

    thread_local! {
        static BPE_SHARED_BUFFER: (Buffer<WordState>, Buffer<Match>) =
            (Buffer(RefCell::new(Vec::new())), Buffer(RefCell::new(Vec::new())));
    }

    fn bpe_merge(&self, chunk: &[u8], output: &mut Vec<Rank>) -> Result<(), Error> {
        Self::BPE_SHARED_BUFFER.with(|buffer| {
            let (word_states_buffer, matches_buffer) = buffer;
            word_states_buffer.prepare(chunk.len());
            matches_buffer.prepare(chunk.len().saturating_sub(1));

            let word_states = &mut *word_states_buffer.borrow_mut();
            let matches = &mut *matches_buffer.borrow_mut();

            for (right_index, byte) in chunk.iter().enumerate() {
                let word = Word::from_bytes(std::slice::from_ref(byte));
                match self.encoder.get(&word) {
                    Some(rank) => {
                        word_states.push(WordState {
                            rank: *rank,
                            word,
                            left_index: right_index.wrapping_sub(1),
                            right_index,
                            is_removed: false,
                        });
                    }
                    None => return Err(Error::NoTokenForByte(*byte)),
                }
            }

            for (left_index, window) in word_states.windows(2).enumerate() {
                let (WordState { word: left, .. }, WordState { word: right, .. }) =
                    (&window[0], &window[1]);

                let combined = Word::combine(&left, &right);
                let rank = self.encoder.get(&combined).copied();

                matches.push(Match {
                    combined,
                    rank: rank.unwrap_or(Rank::MAX),
                    left_index,
                    right_index: left_index + 1,
                    is_removed: rank.is_none(),
                });
            }

            loop {
                let Some((
                    match_index,
                    Match {
                        combined,
                        rank,
                        left_index,
                        right_index,
                        ..
                    },
                )) = matches
                    .iter_mut()
                    .enumerate()
                    .filter(|(_, m)| !m.is_removed)
                    .min_by_key(|(_, m)| m.rank)
                    .map(|(index, m)| (index, m.clone()))
                else {
                    break;
                };

                let new_word_state = &word_states[left_index];
                let consumed_word_state = &word_states[right_index];

                if let Some(left_match) = matches.get_mut(new_word_state.left_index) {
                    let left_word = &word_states[left_match.left_index];
                    let left_new_combined = Word::combine(&left_word.word, &combined);
                    let left_new_rank = self.encoder.get(&left_new_combined).copied();

                    left_match.combined = left_new_combined;
                    left_match.rank = left_new_rank.unwrap_or(Rank::MAX);
                    left_match.is_removed = left_new_rank.is_none();
                }

                if let Some(right_match) = matches.get_mut(consumed_word_state.right_index) {
                    let right_word = &word_states[right_match.right_index];
                    let right_new_combined = Word::combine(&combined, &right_word.word);
                    let right_new_rank = self.encoder.get(&right_new_combined).copied();

                    right_match.combined = right_new_combined;
                    right_match.rank = right_new_rank.unwrap_or(Rank::MAX);
                    right_match.left_index = left_index;
                    right_match.is_removed = right_new_rank.is_none();
                }

                let new_right_match_index = consumed_word_state.right_index;
                let new_word_mut = &mut word_states[left_index];
                new_word_mut.right_index = new_right_match_index;
                new_word_mut.word = combined;
                new_word_mut.rank = rank;

                word_states[right_index].is_removed = true;
                matches[match_index].is_removed = true;
            }

            output.extend(word_states.iter().filter_map(|w| {
                if w.is_removed {
                    None
                } else {
                    Some(w.rank)
                }
            }));

            Ok(())
        })
    }
}

pub enum Splitter {
    AhoCorasick(AhoCorasick),
    Regex(Regex),
}

#[derive(Clone)]
enum Split {
    Literal(Range<usize>),
    Bytes(Range<usize>),
}

impl Splitter {
    fn split(&self, text: &[u8], output: &mut Vec<Split>) -> Result<(), Error> {
        match self {
            Splitter::AhoCorasick(aho_corasick) => {
                let mut start = 0;
                for m in aho_corasick.find_iter(text) {
                    if m.start() > start {
                        output.push(Split::Bytes(start..m.start()));
                    }
                    output.push(Split::Literal(m.start()..m.end()));
                    start = m.end();
                }
                if start < text.len() {
                    output.push(Split::Bytes(start..text.len()));
                }

                Ok(())
            }
            Splitter::Regex(regex) => {
                let mut start = 0;
                for m_result in regex.find_iter(text) {
                    let m = m_result?;
                    if m.start() > start {
                        output.push(Split::Bytes(start..m.start()));
                    }
                    output.push(Split::Bytes(m.start()..m.end()));
                    start = m.end();
                }
                if start < text.len() {
                    output.push(Split::Bytes(start..text.len()));
                }

                Ok(())
            }
        }
    }
}

struct WordState {
    word: Word,
    left_index: usize,
    right_index: usize,
    rank: Rank,
    is_removed: bool,
}

#[derive(Clone)]
struct Match {
    combined: Word,
    left_index: usize,
    right_index: usize,
    rank: Rank,
    is_removed: bool,
}

struct Buffer<T>(RefCell<Vec<T>>);

impl<T> Buffer<T> {
    fn prepare(&self, num_items: usize) {
        let mut items = self.0.borrow_mut();

        items.clear();

        let capacity = items.capacity();
        items.reserve(num_items.saturating_sub(capacity));
    }

    fn borrow_mut(&self) -> RefMut<Vec<T>> {
        self.0.borrow_mut()
    }

    fn take_buffer(&self) -> Vec<T> {
        std::mem::take(&mut self.0.borrow_mut())
    }

    fn return_buffer(&self, buffer: Vec<T>) {
        *self.0.borrow_mut() = buffer;
    }
}

#[derive(Clone)]
struct Word {
    hash: u64,
    length: usize,
}

const HASH_BASE: u64 = 257;
const HASH_MOD: u64 = (1u64 << 31) - 1;

impl Word {
    pub fn from_bytes(seq: &[u8]) -> Self {
        let mut hash = 0u64;
        for &byte in seq {
            hash = (hash * HASH_BASE + byte as u64) % HASH_MOD;
        }
        Self {
            hash,
            length: seq.len(),
        }
    }

    pub fn combine(left: &Self, right: &Self) -> Self {
        let base_pow = pow_mod(HASH_BASE, right.length, HASH_MOD);
        let left_shifted = (left.hash * base_pow) % HASH_MOD;
        let combined_hash = (left_shifted + right.hash) % HASH_MOD;

        Self {
            hash: combined_hash,
            length: left.length + right.length,
        }
    }
}

fn pow_mod(mut base: u64, mut exp: usize, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;

    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % modulus;
        }
        base = (base * base) % modulus;
        exp >>= 1;
    }

    result
}

#[derive(Default)]
struct NoOpHasher {
    hash: u64,
}

impl Hasher for NoOpHasher {
    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, _bytes: &[u8]) {
        panic!("my words won't come out right")
    }

    fn write_u32(&mut self, i: u32) {
        self.hash = i as u64;
    }

    fn write_u64(&mut self, i: u64) {
        self.hash = i;
    }

    fn write_usize(&mut self, i: usize) {
        self.hash = i as u64;
    }
}

impl BuildHasher for NoOpHasher {
    type Hasher = NoOpHasher;

    fn build_hasher(&self) -> Self::Hasher {
        NoOpHasher::default()
    }
}

impl PartialEq for Word {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.length == other.length
    }
}

impl Eq for Word {}

impl Hash for Word {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash.wrapping_mul(31).wrapping_add(self.length as u64));
    }
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("regex compilation failed: {0}")]
    RegexError(#[from] pcre2::Error),

    #[error("aho-corasick build failed: {0}")]
    AhoCorasickError(#[from] aho_corasick::BuildError),

    #[error("invalid token for decoding: {0}")]
    InvalidToken(Rank),

    #[error("token not found in vocabulary: {0}")]
    NoValidToken(String),

    #[error("token not found in vocabulary during bpe encoding: {0}")]
    NoTokenForByte(u8),

    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("json parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("utf-8 decode error: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),

    #[error("parse integer error: {0}")]
    ParseIntError(#[from] std::num::ParseIntError),

    #[error("invalid model format: expected '{{token}} {{rank}}' per line")]
    InvalidModelFormat,

    #[error("invalid model format: expected regex pre-tokenizer")]
    UnexpectedPreTokenizer,

    #[error("empty pre-tokenizer")]
    EmptyPreTokenizer,
}

#[cfg(feature = "pyo3")]
#[pymodule]
fn keep_talkin(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<bindings::Tokenizer>()?;
    m.add_class::<bindings::Token>()?;
    bindings::register_exceptions(py, m)?;
    Ok(())
}
