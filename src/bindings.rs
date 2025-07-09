use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyType};
use std::collections::HashSet;

use crate::{Error, Rank};

pyo3::create_exception!(keep_talkin, TokenizationError, PyValueError);
pyo3::create_exception!(keep_talkin, ModelLoadError, PyIOError);

#[pyclass(eq, hash, frozen)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Token {
    #[pyo3(get)]
    pub bytes: Vec<u8>,
    #[pyo3(get)]
    pub rank: Rank,
}

#[pymethods]
impl Token {
    #[new]
    fn new(bytes: Vec<u8>, rank: Rank) -> Self {
        Self { bytes, rank }
    }

    fn __repr__(&self) -> String {
        format!("Token(bytes={:?}, rank={})", self.bytes, self.rank)
    }

    fn __str__(&self) -> String {
        format!(
            "Token('{}', {})",
            String::from_utf8_lossy(&self.bytes),
            self.rank
        )
    }
}

impl From<crate::Token> for Token {
    fn from(token: crate::Token) -> Self {
        Self {
            bytes: token.bytes,
            rank: token.rank,
        }
    }
}

#[pyclass]
pub struct Tokenizer {
    inner: crate::Tokenizer,
}

#[pymethods]
impl Tokenizer {
    #[classmethod]
    fn from_tokenizer_json(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: crate::Tokenizer::from_tokenizer_json(path)?,
        })
    }

    #[classmethod]
    fn from_model_and_config(
        _cls: &Bound<'_, PyType>,
        model_path: &str,
        config_path: &str,
        regex_pattern: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: crate::Tokenizer::from_model_and_config(model_path, config_path, regex_pattern)?,
        })
    }

    #[classmethod]
    fn from_tekken(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: crate::Tokenizer::from_tekken(path)?,
        })
    }

    #[pyo3(signature = (text, *, allowed_special=None))]
    fn encode(
        &self,
        py: Python,
        text: &str,
        allowed_special: Option<Vec<String>>,
    ) -> PyResult<Vec<Rank>> {
        let start = std::time::Instant::now();
        println!("START: {:?}", start);

        let tokens = py.allow_threads(|| {
            let alloc_start = std::time::Instant::now();
            let bytes = text.as_bytes();
            let allowed_special_set = allowed_special
                .unwrap_or_default()
                .into_iter()
                .collect::<HashSet<_>>();
            println!("BYTES_CONV: {:?}ms", alloc_start.elapsed().as_millis());

            let encode_start = std::time::Instant::now();
            let result = self.inner.encode(bytes, &allowed_special_set)?;
            println!("ENCODE: {:?}ms", encode_start.elapsed().as_millis());
            
            Ok(result)
        });

        println!("TOTAL: {:?}ms", start.elapsed().as_millis());
        tokens
    }

    #[pyo3(signature = (data, *, allowed_special=None))]
    fn encode_bytes(
        &self,
        py: Python,
        data: &[u8],
        allowed_special: Option<Vec<String>>,
    ) -> PyResult<Vec<Rank>> {
        py.allow_threads(|| {
            let allowed_special_set = allowed_special
                .unwrap_or_default()
                .into_iter()
                .collect::<HashSet<_>>();

            Ok(self.inner.encode(data, &allowed_special_set)?)
        })
    }

    fn decode(&self, py: Python, tokens: Vec<Rank>) -> PyResult<String> {
        py.allow_threads(|| {
            let decoded = self.inner.decode(&tokens)?;
            let bytes = decoded.into_iter().flatten().copied().collect::<Vec<_>>();
            Ok(String::from_utf8(bytes)?)
        })
    }

    fn decode_bytes(&self, py: Python, tokens: Vec<Rank>) -> PyResult<PyObject> {
        let bytes = py.allow_threads(|| {
            let decoded = self.inner.decode(&tokens)?;
            Ok::<_, Error>(decoded.into_iter().flatten().copied().collect::<Vec<_>>())
        })?;

        Ok(PyBytes::new(py, &bytes).into())
    }

    #[pyo3(signature = (texts, *, allowed_special=None))]
    fn encode_batch(
        &self,
        py: Python,
        texts: Vec<String>,
        allowed_special: Option<Vec<String>>,
    ) -> PyResult<Vec<Vec<Rank>>> {
        py.allow_threads(|| {
            let byte_texts = texts.iter().map(|text| text.as_bytes()).collect::<Vec<_>>();
            let _allowed_special_set = allowed_special
                .unwrap_or_default()
                .into_iter()
                .collect::<HashSet<_>>();
            Ok(self.inner.encode_batch(byte_texts)?)
        })
    }

    #[pyo3(signature = (data, *, allowed_special=None))]
    fn encode_bytes_batch(
        &self,
        py: Python,
        data: Vec<Vec<u8>>,
        allowed_special: Option<Vec<String>>,
    ) -> PyResult<Vec<Vec<Rank>>> {
        py.allow_threads(|| {
            let byte_refs = data
                .iter()
                .map(|bytes| bytes.as_slice())
                .collect::<Vec<_>>();
            let _allowed_special_set = allowed_special
                .unwrap_or_default()
                .into_iter()
                .collect::<HashSet<_>>();
            Ok(self.inner.encode_batch(byte_refs)?)
        })
    }

    fn decode_batch(&self, py: Python, tokens: Vec<Vec<Rank>>) -> PyResult<Vec<String>> {
        py.allow_threads(|| {
            let token_refs = tokens.iter().map(|t| t.as_slice()).collect::<Vec<_>>();
            let decoded = self.inner.decode_batch(token_refs)?;

            decoded
                .into_iter()
                .map(|token_bytes| {
                    let bytes = token_bytes
                        .into_iter()
                        .flatten()
                        .copied()
                        .collect::<Vec<_>>();
                    Ok(String::from_utf8(bytes)?)
                })
                .collect::<Result<Vec<_>, _>>()
        })
    }

    fn decode_bytes_batch(&self, py: Python, tokens: Vec<Vec<Rank>>) -> PyResult<Vec<PyObject>> {
        let batch_bytes = py.allow_threads(|| {
            let token_refs = tokens.iter().map(|t| t.as_slice()).collect::<Vec<_>>();
            let decoded = self.inner.decode_batch(token_refs)?;

            Ok::<_, Error>(
                decoded
                    .into_iter()
                    .map(|token_bytes| {
                        token_bytes
                            .into_iter()
                            .flatten()
                            .copied()
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
            )
        })?;

        Ok(batch_bytes
            .into_iter()
            .map(|bytes| PyBytes::new(py, &bytes).into())
            .collect())
    }
}

impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        match err {
            Error::IoError(_) => ModelLoadError::new_err(err.to_string()),
            Error::JsonError(_) => ModelLoadError::new_err(err.to_string()),
            Error::RegexError(_) => ModelLoadError::new_err(err.to_string()),
            Error::AhoCorasickError(_) => ModelLoadError::new_err(err.to_string()),
            _ => TokenizationError::new_err(err.to_string()),
        }
    }
}

pub fn register_exceptions(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("TokenizationError", py.get_type::<TokenizationError>())?;
    m.add("ModelLoadError", py.get_type::<ModelLoadError>())?;
    Ok(())
}
