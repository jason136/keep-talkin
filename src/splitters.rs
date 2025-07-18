use std::ops::Range;

use aho_corasick::AhoCorasick;
use pcre2::bytes::{Regex, RegexBuilder};

use crate::{EncoderMap, Error, Rank};

pub enum Splitter {
    AhoCorasick(AhoCorasick),
    Regex(Regex),
    Delimiter(u8),
}

#[derive(Clone)]
pub enum Split {
    Literal(Range<usize>),
    Bytes(Range<usize>),
}

impl Splitter {
    pub fn from_regex_pattern(pattern: &str) -> Result<Self, Error> {
        Ok(Self::Regex(
            RegexBuilder::new()
                .jit_if_available(true)
                .max_jit_stack_size(Some(1024 * 1024))
                .utf(true)
                .ucp(true)
                .build(pattern)?,
        ))
    }

    pub fn split(&self, text: &[u8], offset: usize, output: &mut Vec<Split>) -> Result<(), Error> {
        match self {
            Self::AhoCorasick(aho_corasick) => {
                let mut start = 0;
                for m in aho_corasick.find_iter(text) {
                    if m.start() > start {
                        output.push(Split::Bytes(start + offset..m.start() + offset));
                    }
                    output.push(Split::Literal(m.start() + offset..m.end() + offset));
                    start = m.end();
                }
                if start < text.len() {
                    output.push(Split::Bytes(start + offset..text.len() + offset));
                }

                Ok(())
            }
            Self::Regex(regex) => {
                let mut start = 0;
                for m_result in regex.find_iter(text) {
                    let m = m_result?;
                    if m.start() > start {
                        output.push(Split::Bytes(start + offset..m.start() + offset));
                    }
                    output.push(Split::Bytes(m.start() + offset..m.end() + offset));
                    start = m.end();
                }
                if start < text.len() {
                    output.push(Split::Bytes(start + offset..text.len() + offset));
                }

                Ok(())
            }
            Self::Delimiter(delimiter) => {
                let mut start = 0;
                let mut i = 0;

                loop {
                    match (text.get(i), text.get(i + 1)) {
                        (None, _) => {
                            output.push(Split::Bytes(start + offset..i + offset));
                            break;
                        }
                        (Some(t), Some(n)) if t != delimiter && n == delimiter => {
                            i += 1;
                            output.push(Split::Bytes(start + offset..i + offset));
                            start = i;
                        }
                        (Some(_), _) => i += 1,
                    }
                }

                Ok(())
            }
        }
    }
}

pub enum WordSplitter {
    Unicode,
    Bytes,
}

impl WordSplitter {
    pub fn into_iter<'a>(&self, text: &'a [u8], encoder: &'a EncoderMap) -> WordSplitterIter<'a> {
        match self {
            Self::Unicode => WordSplitterIter::Unicode {
                text,
                encoder,
                start: 0,
                i: 0,
                pending_split: None,
            },
            Self::Bytes => WordSplitterIter::Bytes {
                text,
                encoder,
                i: 0,
            },
        }
    }
}

pub enum WordSplitterIter<'a> {
    Unicode {
        text: &'a [u8],
        encoder: &'a EncoderMap,
        start: usize,
        i: usize,
        pending_split: Option<Result<(Range<usize>, Rank), Error>>,
    },
    Bytes {
        text: &'a [u8],
        encoder: &'a EncoderMap,
        i: usize,
    },
}

impl<'a> Iterator for WordSplitterIter<'a> {
    type Item = Result<(Range<usize>, Rank), Error>;

    fn next(&mut self) -> Option<Result<(Range<usize>, Rank), Error>> {
        match self {
            Self::Unicode {
                text,
                encoder,
                start,
                i,
                pending_split,
            } => {
                if let Some(split) = pending_split.take() {
                    return Some(split);
                }

                while *i < text.len() {
                    match (|| {
                        let unicode_guess = match text[*i] {
                            b if b & 0b1000_0000 == 0 => return Some(&text[*i..*i + 1]),
                            b if b & 0b1110_0000 == 0b1100_0000 && *i + 1 < text.len() => {
                                &text[*i..*i + 2]
                            }
                            b if b & 0b1111_0000 == 0b1110_0000 && *i + 2 < text.len() => {
                                &text[*i..*i + 3]
                            }
                            b if b & 0b1111_1000 == 0b1111_0000 && *i + 3 < text.len() => {
                                &text[*i..*i + 4]
                            }
                            _ => return None,
                        };

                        unicode_guess[1..]
                            .iter()
                            .all(|b| b & 0b1100_0000 == 0b1000_0000)
                            .then_some(())?;

                        Some(unicode_guess)
                    })()
                    .and_then(|guess| encoder.get(guess).map(|entry| (guess, entry.rank)))
                    .or_else(|| {
                        encoder
                            .get(&text[*i..*i + 1])
                            .map(|entry| (&text[*i..*i + 1], entry.rank))
                    }) {
                        Some((word, rank)) => {
                            let current_split = if start < i {
                                *pending_split = Some(Ok((*i..*i + word.len(), rank)));
                                Some(Ok((*start..*i, rank)))
                            } else {
                                Some(Ok((*i..*i + word.len(), rank)))
                            };
                            *i += word.len();
                            *start = *i;
                            if let Some(split) = current_split {
                                return Some(split);
                            }
                        }
                        None => {
                            return Some(Err(Error::NoValidToken(
                                String::from_utf8_lossy(&text[*i..*i + 1]).to_string(),
                            )));
                        }
                    }
                }

                None
            }
            Self::Bytes { text, encoder, i } => {
                if *i < text.len() {
                    let split = *i..*i + 1;
                    let word = match encoder.get(&text[*i..*i + 1]) {
                        Some(entry) => Ok((split, entry.rank)),
                        None => Err(Error::NoValidToken(
                            String::from_utf8_lossy(&text[*i..*i + 1]).to_string(),
                        )),
                    };
                    *i += 1;
                    Some(word)
                } else {
                    None
                }
            }
        }
    }
}
