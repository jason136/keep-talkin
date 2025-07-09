use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    hash::{BuildHasher, Hash, Hasher},
};

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use pcre2::bytes::{Regex, RegexBuilder};
use priority_queue::PriorityQueue;
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
                    Ok(acc)
                } else if let Some(bytes) = self.special_tokens_decoder.get(token) {
                    acc.push(bytes.as_slice());
                    Ok(acc)
                } else {
                    Err(Error::InvalidToken(*token))
                }
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

    pub fn encode(
        &self,
        text: &[u8],
        _allowed_special: &HashSet<String>,
    ) -> Result<Vec<Rank>, Error> {
        self.splitters
            .iter()
            .try_fold(vec![Split::Unfinished(text)], |splits, splitter| {
                splits
                    .into_par_iter()
                    .map(|split| match split {
                        Split::Unfinished(s) => splitter.split(s),
                        finished => Ok(vec![finished]),
                    })
                    .try_fold(Vec::new, |mut acc, result_splits| {
                        result_splits.map(|splits| {
                            acc.extend(splits);
                            acc
                        })
                    })
                    .try_reduce(Vec::new, |mut a, b| {
                        a.extend(b);
                        Ok(a)
                    })
            })?
            .into_par_iter()
            .map(|split| match split {
                Split::Finished(s) => {
                    if let Some(rank) = self.special_encoder.get(&Word::from_bytes(s)) {
                        Ok(vec![*rank])
                    } else if let Some(rank) = self.encoder.get(&Word::from_bytes(s)) {
                        Ok(vec![*rank])
                    } else {
                        Err(Error::NoValidToken(String::from_utf8_lossy(s).to_string()))
                    }
                }
                Split::Unfinished(s) => {
                    if let Some(rank) = self.encoder.get(&Word::from_bytes(s)) {
                        Ok(vec![*rank])
                    } else {
                        self.bpe_merge(s)
                    }
                }
            })
            .try_fold(Vec::new, |mut acc, result_ranks| {
                result_ranks.map(|ranks| {
                    acc.extend(ranks);
                    acc
                })
            })
            .try_reduce(Vec::new, |mut a, b| {
                a.extend(b);
                Ok(a)
            })
    }

    pub fn encode_batch<T, I>(&self, texts: T) -> Result<Vec<Vec<Rank>>, Error>
    where
        T: IntoParallelIterator<Item = I>,
        I: AsRef<[u8]>,
    {
        texts
            .into_par_iter()
            .map(|text| self.encode(text.as_ref(), &HashSet::new()))
            .collect()
    }

    fn bpe_merge(&self, chunk: &[u8]) -> Result<Vec<Rank>, Error> {
        struct WordState {
            word: Word,
            left_index: usize,
            right_index: usize,
            is_removed: bool,
            rank: Rank,
        }

        let mut word_states = chunk
            .iter()
            .enumerate()
            .map(|(right_index, byte)| {
                let word = Word::from_bytes(std::slice::from_ref(byte));
                match self.encoder.get(&word) {
                    Some(rank) => Ok(WordState {
                        rank: *rank,
                        word,
                        left_index: right_index.wrapping_sub(1),
                        right_index,
                        is_removed: false,
                    }),
                    None => Err(Error::NoTokenForByte(*byte)),
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;

        #[derive(PartialEq, Eq)]
        struct Match {
            left_index: usize,
            right_index: usize,
            combined: Word,
            rank: Option<Rank>,
        }

        impl PartialOrd for Match {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for Match {
            fn cmp(&self, other: &Self) -> Ordering {
                match match (self.rank, other.rank) {
                    (Some(a), Some(b)) => b.cmp(&a),
                    (Some(_), None) => Ordering::Greater,
                    (None, Some(_)) => Ordering::Less,
                    (None, None) => Ordering::Equal,
                } {
                    Ordering::Equal => other.left_index.cmp(&self.left_index),
                    o => o,
                }
            }
        }

        let mut matches_queue = word_states
            .windows(2)
            .map(|window| (&window[0], &window[1]))
            .enumerate()
            .map(
                |(left_index, (WordState { word: left, .. }, WordState { word: right, .. }))| {
                    let combined = Word::combine(left, right);
                    let rank = self.encoder.get(&combined).copied();

                    (
                        left_index,
                        Match {
                            left_index,
                            right_index: left_index + 1,
                            combined,
                            rank,
                        },
                    )
                },
            )
            .collect::<PriorityQueue<_, _, NoOpHasher>>();

        loop {
            let Some((
                _,
                Match {
                    left_index: new_word_index,
                    right_index: consumed_word_index,
                    combined,
                    rank: Some(rank),
                    ..
                },
            )) = matches_queue.pop()
            else {
                break;
            };

            let consumed_word = word_states.get(consumed_word_index).unwrap();
            let new_word = word_states.get(new_word_index).unwrap();

            if let Some((left_match_index, left_match_left_index, left_word)) = matches_queue
                .get(&new_word.left_index)
                .map(|(i, m)| (*i, m.left_index, word_states.get(m.left_index).unwrap()))
            {
                let left_new_combined = Word::combine(&left_word.word, &combined);
                let left_new_rank = self.encoder.get(&left_new_combined).copied();

                matches_queue.change_priority(
                    &left_match_index,
                    Match {
                        left_index: left_match_left_index,
                        right_index: new_word_index,
                        combined: left_new_combined,
                        rank: left_new_rank,
                    },
                );
            }

            if let Some((right_match_index, right_match_right_index, right_word)) = matches_queue
                .get(&consumed_word.right_index)
                .map(|(i, m)| (*i, m.right_index, word_states.get(m.right_index).unwrap()))
            {
                let right_new_combined = Word::combine(&combined, &right_word.word);
                let right_new_rank = self.encoder.get(&right_new_combined).copied();

                matches_queue.change_priority(
                    &right_match_index,
                    Match {
                        left_index: new_word_index,
                        right_index: right_match_right_index,
                        combined: right_new_combined,
                        rank: right_new_rank,
                    },
                );
            }

            let new_right_match_index = consumed_word.right_index;

            let consumed_word_mut = word_states.get_mut(consumed_word_index).unwrap();
            consumed_word_mut.is_removed = true;

            let new_word_mut = word_states.get_mut(new_word_index).unwrap();
            new_word_mut.right_index = new_right_match_index;
            new_word_mut.word = combined;
            new_word_mut.rank = rank;
        }

        Ok(word_states
            .into_iter()
            .filter_map(
                |WordState {
                     is_removed, rank, ..
                 }| if is_removed { None } else { Some(rank) },
            )
            .collect())
    }
}

pub enum Splitter {
    AhoCorasick(AhoCorasick),
    Regex(Regex),
}

enum Split<'a> {
    Finished(&'a [u8]),
    Unfinished(&'a [u8]),
}

impl Splitter {
    fn split<'a>(&'a self, text: &'a [u8]) -> Result<Vec<Split<'a>>, Error> {
        match self {
            Splitter::AhoCorasick(aho_corasick) => {
                let mut chunks = Vec::new();
                let mut start = 0;
                for m in aho_corasick.find_iter(text) {
                    if m.start() > start {
                        chunks.push(Split::Unfinished(&text[start..m.start()]));
                    }
                    chunks.push(Split::Finished(&text[m.start()..m.end()]));
                    start = m.end();
                }
                if start < text.len() {
                    chunks.push(Split::Unfinished(&text[start..]));
                }

                Ok(chunks)
            }
            Splitter::Regex(regex) => {
                let mut chunks = Vec::new();
                let mut start = 0;
                for m in regex.find_iter(text).collect::<Result<Vec<_>, _>>()? {
                    if m.start() > start {
                        chunks.push(Split::Unfinished(&text[start..m.start()]));
                    }
                    chunks.push(Split::Unfinished(&text[m.start()..m.end()]));
                    start = m.end();
                }
                if start < text.len() {
                    chunks.push(Split::Unfinished(&text[start..]));
                }

                Ok(chunks)
            }
        }
    }
}

struct Word {
    hash: u64,
    length: usize,
}

const HASH_BASE: u64 = 257;
const HASH_MOD: u64 = (1u64 << 61) - 1;

impl Word {
    pub fn from_bytes(seq: &[u8]) -> Self {
        let mut hash = 0u64;
        for &byte in seq {
            hash = ((hash as u128 * HASH_BASE as u128 + byte as u128) % HASH_MOD as u128) as u64;
        }
        Self {
            hash,
            length: seq.len(),
        }
    }

    pub fn combine(left: &Self, right: &Self) -> Self {
        let left_shifted = (left.hash as u128 * pow_mod(HASH_BASE, right.length, HASH_MOD) as u128)
            % HASH_MOD as u128;
        let combined_hash = ((left_shifted + right.hash as u128) % HASH_MOD as u128) as u64;

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
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
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
