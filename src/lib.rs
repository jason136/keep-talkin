use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    hash::{BuildHasher, Hash, Hasher},
};

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use priority_queue::PriorityQueue;
use rayon::prelude::*;
use thiserror::Error;

#[cfg(feature = "pcre2")]
use pcre2::bytes::{Regex, RegexBuilder};

#[cfg(not(feature = "pcre2"))]
use regex::bytes::Regex;

pub mod loader;

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
        regex_pattern: &str,
    ) -> Result<Self, Error> {
        let special_tokens_matcher = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(
                special_vocab
                    .iter()
                    .map(|v| v.bytes.as_slice())
                    .collect::<Vec<_>>(),
            )?;

        #[cfg(feature = "pcre2")]
        let regex = {
            let mut builder = RegexBuilder::new();
            #[cfg(feature = "jit-regex")]
            {
                // builder.jit_if_available(true);
                // builder.max_jit_stack_size(Some(1024 * 1024));
            }

            // builder.utf(true);
            // builder.ucp(true);

            builder.build(regex_pattern)?
        };

        #[cfg(not(feature = "pcre2"))]
        let regex = Regex::new(regex_pattern)?;

        let splitters = vec![
            Splitter::AhoCorasick(special_tokens_matcher),
            Splitter::Regex(regex),
        ];

        Self::from_vocab_and_splitters(vocab, special_vocab, splitters)
    }

    fn from_vocab_and_splitters(
        vocab: Vec<Token>,
        special_vocab: Vec<Token>,
        splitters: Vec<Splitter>,
    ) -> Result<Self, Error> {
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

    pub fn encode(
        &self,
        text: &[u8],
        _allowed_special: &HashSet<String>,
    ) -> Result<Vec<Rank>, Error> {
        let splits =
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
                })?;

        println!("splits: {:?}", splits);

        splits
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
                Split::Unfinished(s) => self.bpe_merge(s),
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
            .map(|(left_index, byte)| {
                let word = Word::from_bytes(std::slice::from_ref(byte));
                Ok(WordState {
                    rank: self
                        .encoder
                        .get(&word)
                        .copied()
                        .ok_or(Error::NoTokenForByte(*byte))?,
                    word,
                    left_index,
                    right_index: left_index + 1,
                    is_removed: false,
                })
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
                match (self.rank, other.rank) {
                    (Some(a), Some(b)) => b.cmp(&a),
                    (Some(_), None) => Ordering::Less,
                    (None, Some(_)) => Ordering::Greater,
                    (None, None) => Ordering::Equal,
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
                },
            )) = matches_queue.pop()
            else {
                break;
            };

            let consumed_word = word_states.get(consumed_word_index).unwrap();
            let new_word = word_states.get(new_word_index).unwrap();

            let left_match_and_word = matches_queue
                .get(&new_word.left_index)
                .map(|(i, m)| (*i, m.left_index, word_states.get(m.left_index).unwrap()));

            if let Some((left_match_index, left_match_left_index, left_word)) = left_match_and_word
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

            let right_match_and_word = matches_queue
                .get(&consumed_word.right_index)
                .map(|(i, m)| (*i, m.right_index, word_states.get(m.right_index).unwrap()));

            if let Some((right_match_index, right_match_right_index, right_word)) =
                right_match_and_word
            {
                let right_new_combined = Word::combine(&right_word.word, &combined);
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

#[derive(Debug)]
pub enum Splitter {
    AhoCorasick(AhoCorasick),
    Regex(Regex),
}

#[derive(Debug)]
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

                println!("ahocorasick chunks: {:?}", chunks);

                Ok(chunks)
            }
            Splitter::Regex(regex) => regex
                .find_iter(text)
                .map(|m| {
                    println!("regex match: {:?}", m);

                    #[cfg(feature = "pcre2")]
                    {
                        Ok(m.map(|m| Split::Unfinished(m.as_bytes()))?)
                    }
                    #[cfg(not(feature = "pcre2"))]
                    {
                        Ok(Split::Unfinished(m.as_bytes()))
                    }
                })
                .collect::<Result<Vec<_>, _>>()
                .map(|chunks| {
                    println!("regex chunks: {:?}", chunks);
                    chunks
                }),
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
            hash = (hash.wrapping_mul(HASH_BASE) + byte as u64) % HASH_MOD;
        }
        Self {
            hash,
            length: seq.len(),
        }
    }

    pub fn combine(left: &Self, right: &Self) -> Self {
        let combined_hash = (left
            .hash
            .wrapping_mul(pow_mod(HASH_BASE, right.length, HASH_MOD))
            + right.hash)
            % HASH_MOD;

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
            result = (result.wrapping_mul(base)) % modulus;
        }
        base = (base.wrapping_mul(base)) % modulus;
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
    #[cfg(not(feature = "pcre2"))]
    #[error("regex compilation failed: {0}")]
    RegexError(#[from] regex::Error),

    #[cfg(feature = "pcre2")]
    #[error("pcre2 regex compilation failed: {0}")]
    Pcre2Error(#[from] pcre2::Error),

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
