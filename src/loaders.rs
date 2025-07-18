use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use base64::{engine::general_purpose::STANDARD, Engine};
use serde::Deserialize;

use crate::{splitters::WordSplitter, Error, Rank, Splitter, Token, Tokenizer};

#[derive(Deserialize)]
struct HuggingFaceTokenizer {
    added_tokens: Vec<HuggingFaceAddedToken>,
    normalizer: Option<HuggingFaceNormalizer>,
    pre_tokenizer: HuggingFacePreTokenizer,
    model: HuggingFaceModel,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum HuggingFaceNormalizer {
    Sequence {
        normalizers: Vec<HuggingFaceSubNormalizer>,
    },
    Replace {
        pattern: HuggingFaceNormalizerPattern,
        content: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum HuggingFaceSubNormalizer {
    Replace {
        pattern: HuggingFaceNormalizerPattern,
        content: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
enum HuggingFaceNormalizerPattern {
    String(String),
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum HuggingFacePreTokenizer {
    Sequence {
        pretokenizers: Vec<HuggingFaceSubPreTokenizer>,
    },
    Split {
        pattern: HuggingFacePattern,
    },
    Metaspace {
        replacement: Option<String>,
        prepend_scheme: Option<String>,
    },
    ByteLevel,
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum HuggingFaceSubPreTokenizer {
    Split {
        pattern: HuggingFacePattern,
    },
    Metaspace {
        replacement: Option<String>,
        prepend_scheme: Option<String>,
    },
    ByteLevel,
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
enum HuggingFacePattern {
    Regex(String),
    String(String),
}

#[derive(Deserialize)]
struct HuggingFaceAddedToken {
    id: Rank,
    content: String,
    #[allow(dead_code)]
    special: bool,
}

#[derive(Deserialize)]
struct HuggingFaceModel {
    byte_fallback: Option<bool>,
    ignore_merges: Option<bool>,
    vocab: HashMap<String, Rank>,
    merges: Vec<HuggingFaceMerge>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum HuggingFaceMerge {
    String(String),
    Array(Vec<String>),
}

impl Tokenizer {
    pub fn from_tokenizer_json<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let hf_tokenizer: HuggingFaceTokenizer = serde_json::from_reader(reader)?;

        let normalizers = match hf_tokenizer.normalizer {
            Some(HuggingFaceNormalizer::Sequence { normalizers }) => normalizers
                .into_iter()
                .filter(|n| matches!(n, HuggingFaceSubNormalizer::Replace { .. }))
                .collect::<Vec<_>>(),
            Some(HuggingFaceNormalizer::Replace { pattern, content }) => {
                vec![HuggingFaceSubNormalizer::Replace { pattern, content }]
            }
            _ => vec![],
        };

        let pre_tokenizers = match hf_tokenizer.pre_tokenizer {
            HuggingFacePreTokenizer::Sequence { pretokenizers } => pretokenizers,
            HuggingFacePreTokenizer::Split { pattern } => {
                vec![HuggingFaceSubPreTokenizer::Split { pattern }]
            }
            HuggingFacePreTokenizer::Metaspace {
                replacement,
                prepend_scheme,
            } => {
                vec![HuggingFaceSubPreTokenizer::Metaspace {
                    replacement,
                    prepend_scheme,
                }]
            }
            HuggingFacePreTokenizer::ByteLevel => vec![HuggingFaceSubPreTokenizer::ByteLevel],
            HuggingFacePreTokenizer::Other => vec![],
        };

        #[derive(Debug)]
        enum SplitTechnique<'a> {
            Regex((Vec<&'a str>, bool)),
            Metaspace(&'a str, bool),
        }

        let split_technique = match (normalizers.as_slice(), pre_tokenizers.as_slice()) {
            ([normalizer], [pretokenizer]) => match (normalizer, pretokenizer) {
                (
                    HuggingFaceSubNormalizer::Replace {
                        pattern: HuggingFaceNormalizerPattern::String(replace_pattern),
                        content,
                    },
                    HuggingFaceSubPreTokenizer::Split {
                        pattern: HuggingFacePattern::String(split_pattern),
                    },
                ) if replace_pattern == split_pattern && split_pattern == " " => {
                    SplitTechnique::Metaspace(content, false)
                }
                _ => return Err(Error::UnsupportedNormalizerPretokenizer),
            },
            (
                [],
                [HuggingFaceSubPreTokenizer::Metaspace {
                    replacement,
                    prepend_scheme,
                }],
            ) => {
                let replacement = replacement.as_ref().map_or("▁", |s| s.as_str());
                SplitTechnique::Metaspace(replacement, prepend_scheme.is_some())
            }
            ([], [pretokenizers @ .., last]) => {
                let mut pretokenizers_acc = pretokenizers.iter().collect::<Vec<_>>();
                let byte_level_encoded = matches!(last, HuggingFaceSubPreTokenizer::ByteLevel);
                if !byte_level_encoded {
                    pretokenizers_acc.push(last);
                }

                let regex_patterns = pretokenizers_acc
                    .iter()
                    .map(|p| match p {
                        HuggingFaceSubPreTokenizer::Split {
                            pattern: HuggingFacePattern::Regex(regex_pattern),
                        } => Ok(regex_pattern.as_str()),
                        _ => Err(Error::UnexpectedNonRegexSplitter),
                    })
                    .collect::<Result<Vec<_>, Error>>()?;

                SplitTechnique::Regex((regex_patterns, byte_level_encoded))
            }
            _ => return Err(Error::MissingNormalizerPretokenizer),
        };

        let (splitters, word_splitter, prefix) = match &split_technique {
            SplitTechnique::Regex((regex_patterns, _)) => (
                regex_patterns
                    .iter()
                    .map(|p| Splitter::from_regex_pattern(p))
                    .collect::<Result<Vec<_>, Error>>()?,
                WordSplitter::Bytes,
                None,
            ),
            SplitTechnique::Metaspace(_, prepend_scheme) => (
                vec![Splitter::Delimiter(b' ')],
                WordSplitter::Unicode,
                prepend_scheme.then_some(vec![b' ']),
            ),
        };

        let transform_token = |token: String, rank: Rank| -> Token {
            if hf_tokenizer.model.byte_fallback.unwrap_or(false) {
                if let Some(hex_part) = token
                    .strip_prefix("<0x")
                    .and_then(|rest| rest.strip_suffix('>'))
                {
                    if hex_part.len() == 2 {
                        if let Ok(byte) = u8::from_str_radix(hex_part.to_lowercase().as_str(), 16) {
                            return Token {
                                bytes: vec![byte],
                                rank,
                            };
                        }
                    }
                }
            }

            match &split_technique {
                SplitTechnique::Metaspace(replacement, _) => Token {
                    bytes: reverse_sentence_piece(&token, replacement),
                    rank,
                },
                SplitTechnique::Regex((_, byte_level_encoded)) => {
                    if *byte_level_encoded {
                        return Token {
                            bytes: reverse_byte_level(&token),
                            rank,
                        };
                    } else {
                        Token {
                            bytes: token.into_bytes(),
                            rank,
                        }
                    }
                }
            }
        };

        let transform_merge = |merge: &str| -> Vec<u8> {
            match &split_technique {
                SplitTechnique::Metaspace(replacement, _) => {
                    reverse_sentence_piece(merge, replacement)
                }
                SplitTechnique::Regex((_, byte_level_encoded)) => {
                    if *byte_level_encoded {
                        reverse_byte_level(merge)
                    } else {
                        merge.as_bytes().to_vec()
                    }
                }
            }
        };

        let special_vocab = hf_tokenizer
            .added_tokens
            .into_iter()
            .map(|token| transform_token(token.content, token.id))
            .collect::<Vec<_>>();

        let mut vocab_vec = hf_tokenizer.model.vocab.into_iter().collect::<Vec<_>>();
        vocab_vec.sort_by_key(|(_, rank)| *rank);
        let vocab = vocab_vec
            .into_iter()
            .map(|(token, rank)| transform_token(token, rank))
            .collect::<Vec<_>>();

        let merges = if hf_tokenizer.model.ignore_merges.unwrap_or(false) {
            None
        } else {
            Some(
                hf_tokenizer
                    .model
                    .merges
                    .into_iter()
                    .map(|merge| {
                        let vecs = match merge {
                            HuggingFaceMerge::String(merge) => {
                                merge.split(' ').map(|s| s.to_string()).collect::<Vec<_>>()
                            }
                            HuggingFaceMerge::Array(merges) => merges,
                        };

                        let mut parts = vecs.into_iter();
                        let (Some(left), Some(right)) = (parts.next(), parts.next()) else {
                            return Err(Error::InvalidModelFormat);
                        };

                        Ok((transform_merge(&left), transform_merge(&right)))
                    })
                    .collect::<Result<Vec<_>, Error>>()?,
            )
        };

        Tokenizer::from_vocab_and_splitters(
            vocab,
            special_vocab,
            merges,
            prefix,
            splitters,
            word_splitter,
        )
    }
}

#[derive(Deserialize)]
struct TokenizerConfig {
    added_tokens_decoder: Option<HashMap<String, SpecialToken>>,
}

#[derive(Deserialize)]
struct SpecialToken {
    content: String,
    special: bool,
}

impl Tokenizer {
    pub fn from_model_and_config(
        model_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
        regex_patterns: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<Self, Error> {
        let model_file = File::open(model_path)?;
        let model_reader = BufReader::new(model_file);

        let vocab = model_reader
            .lines()
            .map(|line| {
                let line = line?;
                let parts = line.split(' ').collect::<Vec<_>>();
                let [token, rank_str] = parts[..2] else {
                    return Err(Error::InvalidModelFormat);
                };

                Ok(Token {
                    bytes: decode_base64_tokens(token),
                    rank: rank_str.parse()?,
                })
            })
            .collect::<Result<Vec<Token>, Error>>()?;

        let config_file = File::open(config_path)?;
        let config_reader = BufReader::new(config_file);
        let config: TokenizerConfig = serde_json::from_reader(config_reader)?;

        let special_vocab = config
            .added_tokens_decoder
            .unwrap_or_default()
            .into_iter()
            .filter(|(_, token)| token.special)
            .map(|(rank_str, token)| {
                Ok(Token {
                    bytes: decode_base64_tokens(&token.content),
                    rank: rank_str.parse()?,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let splitters = regex_patterns
            .into_iter()
            .map(|r| Splitter::from_regex_pattern(r.as_ref()))
            .collect::<Result<Vec<_>, Error>>()?;

        Tokenizer::from_vocab_and_splitters(
            vocab,
            special_vocab,
            None,
            None,
            splitters,
            WordSplitter::Bytes,
        )
    }
}

#[derive(Deserialize)]
struct TekkenTokenizer {
    config: TekkenConfig,
    vocab: Vec<TekkenToken>,
    special_tokens: Vec<TekkenSpecialToken>,
}

#[derive(Deserialize)]
struct TekkenConfig {
    pattern: String,
}

#[derive(Deserialize)]
struct TekkenToken {
    rank: Rank,
    token_bytes: String,
}

#[derive(Deserialize)]
struct TekkenSpecialToken {
    rank: Rank,
    token_str: String,
}

impl Tokenizer {
    pub fn from_tekken<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let tekken: TekkenTokenizer = serde_json::from_reader(reader)?;

        let vocab = tekken
            .vocab
            .into_iter()
            .map(|tekken_token| Token {
                bytes: decode_base64_tokens(&tekken_token.token_bytes),
                rank: tekken_token.rank,
            })
            .collect();

        let special_vocab = tekken
            .special_tokens
            .into_iter()
            .map(|special_token| Token {
                bytes: special_token.token_str.into_bytes(),
                rank: special_token.rank,
            })
            .collect();

        let splitters = vec![Splitter::from_regex_pattern(&tekken.config.pattern)?];

        Tokenizer::from_vocab_and_splitters(
            vocab,
            special_vocab,
            None,
            None,
            splitters,
            WordSplitter::Bytes,
        )
    }
}

fn reverse_sentence_piece(token: &str, pattern: &str) -> Vec<u8> {
    if token.starts_with('<') && token.ends_with('>') {
        return token.as_bytes().to_vec();
    }

    token.replace(pattern, " ").as_bytes().to_vec()
}

fn reverse_byte_level(token: &str) -> Vec<u8> {
    if token.starts_with('<') && token.ends_with('>') {
        return token.as_bytes().to_vec();
    }

    token
        .chars()
        .map(|c| {
            let unicode_val = c as u32;
            match unicode_val {
                33..=126 => unicode_val as u8,
                161..=172 | 174..=255 => unicode_val as u8,
                0x100..=0x17F => {
                    let offset = unicode_val - 0x100;
                    match offset {
                        0..=32 => offset as u8,
                        33..=66 => (offset + 94) as u8,
                        67 => 173,
                        _ => unicode_val as u8,
                    }
                }
                _ => unicode_val as u8,
            }
        })
        .collect()
}

fn decode_base64_tokens(token: &str) -> Vec<u8> {
    if token.starts_with('<') && token.ends_with('>') {
        return token.as_bytes().to_vec();
    }

    STANDARD
        .decode(token)
        .unwrap_or_else(|_| token.as_bytes().to_vec())
}
