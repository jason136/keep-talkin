use std::{
    borrow::Cow,
    cell::{RefCell, RefMut},
    cmp::Ordering,
    collections::HashMap,
    ops::Range,
};

use aho_corasick::{AhoCorasickBuilder, MatchKind};
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use thiserror::Error;

#[cfg(feature = "pyo3")]
mod bindings;
pub mod loaders;
pub(crate) mod splitters;

#[cfg(feature = "pyo3")]
use pyo3::{prelude::*, types::PyModule};

use crate::splitters::{Split, Splitter, WordSplitter};

pub type Rank = u32;

struct EncoderEntry {
    priorities: Option<HashMap<usize, Rank, FxBuildHasher>>,
    rank: Rank,
}

type EncoderMap = HashMap<Vec<u8>, EncoderEntry, FxBuildHasher>;
type SpecialEncoderMap = HashMap<Vec<u8>, Rank, FxBuildHasher>;
type DecoderMap = HashMap<Rank, Vec<u8>, FxBuildHasher>;

pub struct Token {
    pub bytes: Vec<u8>,
    pub rank: Rank,
}

pub struct Tokenizer {
    encoder: EncoderMap,
    special_encoder: SpecialEncoderMap,
    decoder: DecoderMap,
    special_tokens_decoder: DecoderMap,
    prefix: Option<Vec<u8>>,
    splitters: Vec<Splitter>,
    word_splitter: WordSplitter,
}

impl Tokenizer {
    fn from_vocab_and_splitters(
        vocab: Vec<Token>,
        special_vocab: Vec<Token>,
        merges: Option<Vec<(Vec<u8>, Vec<u8>)>>,
        prefix: Option<Vec<u8>>,
        splitters: Vec<Splitter>,
        word_splitter: WordSplitter,
    ) -> Result<Self, Error> {
        let special_tokens_matcher = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(
                special_vocab
                    .iter()
                    .map(|v| v.bytes.as_slice())
                    .collect::<Vec<_>>(),
            )?;

        let mut splitters_acc = vec![Splitter::AhoCorasick(special_tokens_matcher)];
        splitters_acc.extend(splitters);

        let mut merge_priorities_map = HashMap::new();
        if let Some(merges) = merges {
            merges
                .into_iter()
                .enumerate()
                .for_each(|(priority, (mut acc, right))| {
                    let left_len = acc.len();
                    acc.extend_from_slice(&right);
                    merge_priorities_map
                        .entry(acc)
                        .or_insert_with(HashMap::default)
                        .insert(left_len, priority as Rank);
                });
        };

        let mut encoder = EncoderMap::default();
        let mut decoder = DecoderMap::default();

        for item in vocab {
            let priorities = merge_priorities_map.get(&item.bytes).cloned();
            encoder.insert(
                item.bytes.clone(),
                EncoderEntry {
                    rank: item.rank,
                    priorities,
                },
            );
            decoder.insert(item.rank, item.bytes);
        }

        let mut special_encoder = SpecialEncoderMap::default();
        let mut special_tokens_decoder = DecoderMap::default();

        for item in special_vocab {
            special_encoder.insert(item.bytes.clone(), item.rank);
            special_tokens_decoder.insert(item.rank, item.bytes);
        }

        Ok(Self {
            encoder,
            special_encoder,
            decoder,
            special_tokens_decoder,
            prefix,
            splitters: splitters_acc,
            word_splitter,
        })
    }

    pub fn decode(&self, tokens: &[Rank]) -> Result<Vec<&[u8]>, Error> {
        let get_bytes = |token: &Rank| -> Result<&[u8], Error> {
            if let Some(bytes) = self.decoder.get(&token) {
                Ok(bytes.as_slice())
            } else if let Some(bytes) = self.special_tokens_decoder.get(&token) {
                Ok(bytes.as_slice())
            } else {
                return Err(Error::InvalidToken(*token));
            }
        };

        let mut sequence = if tokens.len() < 1024 {
            tokens
                .iter()
                .map(get_bytes)
                .collect::<Result<Vec<_>, Error>>()
        } else {
            tokens
                .par_iter()
                .try_fold(Vec::new, |mut acc, token| {
                    acc.push(get_bytes(token)?);
                    Ok(acc)
                })
                .try_reduce(Vec::new, |mut a, b| {
                    a.extend(b);
                    Ok(a)
                })
        }?;

        if let Some(prefix) = &self.prefix {
            if sequence.first().map_or(false, |s| s.starts_with(prefix)) {
                sequence[0] = &sequence[0][prefix.len()..];
                return Ok(sequence);
            }
        }

        Ok(sequence)
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
        static SPLIT_SHARED_BUFFER: Buffer<Split> = const { Buffer(RefCell::new(Vec::new())) };
        static RANK_SHARED_BUFFER: Buffer<Rank> = const { Buffer(RefCell::new(Vec::new())) };
    }

    pub fn encode(&self, text: &[u8]) -> Result<Vec<Rank>, Error> {
        let text = if let Some(prefix) = &self.prefix {
            let mut text_acc = Vec::with_capacity(prefix.len() + text.len());
            text_acc.extend_from_slice(prefix);
            text_acc.extend_from_slice(text);
            Cow::Owned(text_acc)
        } else {
            Cow::Borrowed(text)
        };

        self.splitters
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
                                    Split::Bytes(r) => {
                                        splitter.split(&text[r.clone()], r.start, &mut acc)?
                                    }
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
            .par_iter()
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
                            let bytes = &text[r.start..r.end];
                            if let Some(rank) = self.special_encoder.get(bytes) {
                                acc.push(*rank);
                            } else if let Some(entry) = self.encoder.get(bytes) {
                                acc.push(entry.rank);
                            } else {
                                return Err(Error::NoValidToken(
                                    String::from_utf8_lossy(bytes).to_string(),
                                ));
                            }
                        }
                        Split::Bytes(r) => {
                            let bytes = &text[r.start..r.end];
                            if let Some(entry) = self.encoder.get(bytes) {
                                acc.push(entry.rank);
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
            })
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
        static BPE_SHARED_BUFFER: (Buffer<WordState>, Buffer<Match>) = const {
            (Buffer(RefCell::new(Vec::new())), Buffer(RefCell::new(Vec::new())))
        };
    }

    fn bpe_merge(&self, chunk: &[u8], output: &mut Vec<Rank>) -> Result<(), Error> {
        Self::BPE_SHARED_BUFFER.with(|buffer| {
            let (word_states_buffer, matches_buffer) = buffer;
            word_states_buffer.prepare(chunk.len());
            matches_buffer.prepare(chunk.len().saturating_sub(1));

            let word_states = &mut *word_states_buffer.borrow_mut();
            let matches = &mut *matches_buffer.borrow_mut();

            for (right_index, word_result) in self
                .word_splitter
                .into_iter(chunk, &self.encoder)
                .enumerate()
            {
                let (word, rank) = word_result?;
                word_states.push(WordState {
                    rank,
                    word,
                    left_index: right_index.wrapping_sub(1),
                    right_index,
                    is_removed: false,
                });
            }

            for (left_index, window) in word_states.windows(2).enumerate() {
                let (WordState { word: left, .. }, WordState { word: right, .. }) =
                    (&window[0], &window[1]);

                let combined = &chunk[left.start..right.end];
                let entry = self.encoder.get(combined);

                let (rank, priority, is_removed) = match entry {
                    Some(entry) => (
                        entry.rank,
                        entry
                            .priorities
                            .as_ref()
                            .and_then(|p| p.get(&left.len()).copied()),
                        false,
                    ),
                    None => (Rank::MAX, None, true),
                };

                matches.push(Match {
                    word: left.start..right.end,
                    left_index,
                    right_index: left_index + 1,
                    rank,
                    priority,
                    is_removed,
                });
            }

            loop {
                let active_matches: Vec<_> = matches
                    .iter_mut()
                    .enumerate()
                    .filter(|(_, m)| !m.is_removed)
                    .collect();

                if active_matches.is_empty() {
                    break;
                }

                let Some((
                    match_index,
                    Match {
                        word,
                        rank,
                        left_index,
                        right_index,
                        ..
                    },
                )) = active_matches
                    .into_iter()
                    .min_by(|(_, l), (_, r)| match (l.priority, r.priority) {
                        (Some(left), Some(right)) => left.cmp(&right),
                        (Some(_), None) => Ordering::Less,
                        (None, Some(_)) => Ordering::Greater,
                        (None, None) => l.rank.cmp(&r.rank),
                    })
                    .map(|(index, m)| (index, m.clone()))
                else {
                    break;
                };

                let new_word_state = &word_states[left_index];
                let consumed_word_state = &word_states[right_index];

                if let Some(left_match) = matches.get_mut(new_word_state.left_index) {
                    let left_word = &word_states[left_match.left_index];
                    let new_word = left_word.word.start..word.end;
                    let new_entry = self.encoder.get(&chunk[new_word.clone()]);

                    let (rank, priority, is_removed) = match new_entry {
                        Some(entry) => (
                            entry.rank,
                            entry
                                .priorities
                                .as_ref()
                                .and_then(|p| p.get(&left_word.word.len()).copied()),
                            false,
                        ),
                        None => (Rank::MAX, None, true),
                    };

                    left_match.word = new_word;
                    left_match.rank = rank;
                    left_match.priority = priority;
                    left_match.is_removed = is_removed;
                }

                if let Some(right_match) = matches.get_mut(consumed_word_state.right_index) {
                    let right_word = &word_states[right_match.right_index];
                    let new_word = word.start..right_word.word.end;
                    let new_entry = self.encoder.get(&chunk[new_word.clone()]);
                    let (rank, priority, is_removed) = match new_entry {
                        Some(entry) => (
                            entry.rank,
                            entry
                                .priorities
                                .as_ref()
                                .and_then(|p| p.get(&word.len()).copied()),
                            false,
                        ),
                        None => (Rank::MAX, None, true),
                    };

                    right_match.word = new_word;
                    right_match.rank = rank;
                    right_match.priority = priority;
                    right_match.is_removed = is_removed;
                    right_match.left_index = left_index;
                }

                let new_right_match_index = consumed_word_state.right_index;
                let new_word_mut = &mut word_states[left_index];
                new_word_mut.right_index = new_right_match_index;
                new_word_mut.word = word;
                new_word_mut.rank = rank;

                word_states[right_index].is_removed = true;
                matches[match_index].is_removed = true;
            }

            output.extend(
                word_states
                    .iter()
                    .filter_map(|w| (!w.is_removed).then_some(w.rank)),
            );

            Ok(())
        })
    }
}

struct WordState {
    word: Range<usize>,
    left_index: usize,
    right_index: usize,
    rank: Rank,
    is_removed: bool,
}

#[derive(Clone, Debug)]
struct Match {
    word: Range<usize>,
    left_index: usize,
    right_index: usize,
    rank: Rank,
    priority: Option<Rank>,
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

    #[error("token not found in vocabulary during bpe encoding: {0:?}")]
    NoTokenForWord(Vec<u8>),

    #[error("invalid unicode sequence detected: {0:?}")]
    InvalidUnicodeSequence(Vec<u8>),

    #[error("token not found in vocabulary during prefix decoding: {0:?}")]
    NoPrefixToken(Vec<u8>),

    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("json parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("utf-8 decode error: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),

    #[error("parse integer error: {0}")]
    ParseIntError(#[from] std::num::ParseIntError),

    #[error("utf-8 encode error: {0}")]
    Utf8EncodeError(#[from] std::string::FromUtf8Error),

    #[error("invalid model format: expected '{{token}} {{rank}}' per line")]
    InvalidModelFormat,

    #[error("unsupported normalizer and pretokenizer combination")]
    UnsupportedNormalizerPretokenizer,

    #[error("unexpected non-regex splitter pre-tokenizer")]
    UnexpectedNonRegexSplitter,

    #[error("missing supported normalizer and pretokenizer combination")]
    MissingNormalizerPretokenizer,
}

#[cfg(feature = "pyo3")]
#[pymodule]
fn keep_talkin(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<bindings::Tokenizer>()?;
    m.add_class::<bindings::Token>()?;
    bindings::register_exceptions(py, m)?;
    Ok(())
}
