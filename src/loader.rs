use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use base64::{engine::general_purpose::STANDARD, Engine};
use serde::Deserialize;

use crate::{Error, Rank, Token, Tokenizer};

#[derive(Deserialize)]
struct HuggingFaceTokenizer {
    added_tokens: Vec<HuggingFaceAddedToken>,
    pre_tokenizer: HuggingFacePreTokenizer,
    model: HuggingFaceModel,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum HuggingFacePreTokenizer {
    Sequence {
        pretokenizers: Vec<HuggingFaceSubPreTokenizer>,
    },
    ByteLevel,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum HuggingFaceSubPreTokenizer {
    Split { pattern: HuggingFacePattern },
    ByteLevel,
}

#[derive(Deserialize)]
enum HuggingFacePattern {
    Regex(String),
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
    vocab: HashMap<String, Rank>,
}

impl Tokenizer {
    pub fn from_tokenizer_json<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let hf_tokenizer: HuggingFaceTokenizer = serde_json::from_reader(reader)?;

        fn reverse_byte_level(token_str: &str) -> Vec<u8> {
            token_str
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

        let vocab_map = hf_tokenizer.model.vocab.clone();
        let vocab = vocab_map
            .iter()
            .map(|(token_str, rank)| Token {
                bytes: reverse_byte_level(token_str),
                rank: *rank,
            })
            .collect::<Vec<_>>();

        let special_vocab = hf_tokenizer
            .added_tokens
            .into_iter()
            // .filter(|token| token.special)
            .map(|token| Token {
                bytes: reverse_byte_level(&token.content),
                rank: token.id,
            })
            .collect::<Vec<_>>();

        let regexes = if let HuggingFacePreTokenizer::Sequence { mut pretokenizers } =
            hf_tokenizer.pre_tokenizer
        {
            if pretokenizers.len() < 2 {
                return Err(Error::UnexpectedPreTokenizer);
            }

            let last = pretokenizers.pop().ok_or(Error::EmptyPreTokenizer)?;

            if !matches!(last, HuggingFaceSubPreTokenizer::ByteLevel { .. }) {
                return Err(Error::UnexpectedPreTokenizer);
            }

            pretokenizers
                .into_iter()
                .map(|sub_tokenizer| match sub_tokenizer {
                    HuggingFaceSubPreTokenizer::Split {
                        pattern: HuggingFacePattern::Regex(regex),
                    } => Ok(regex),
                    _ => Err(Error::UnexpectedPreTokenizer),
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![]
        };

        Tokenizer::from_vocab_and_regex(vocab, special_vocab, None, regexes)
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
        regex_pattern: &str,
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

        Tokenizer::from_vocab_and_regex(vocab, special_vocab, None, vec![regex_pattern])
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

        let pattern = &tekken.config.pattern;
        Tokenizer::from_vocab_and_regex(vocab, special_vocab, None, vec![pattern])
    }
}

fn decode_base64_tokens(token: &str) -> Vec<u8> {
    if token.starts_with('<') && token.ends_with('>') {
        return token.as_bytes().to_vec();
    }

    STANDARD
        .decode(token)
        .unwrap_or_else(|_| token.as_bytes().to_vec())
}
