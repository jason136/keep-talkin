use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyType};

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

    fn encode(&self, py: Python, data: &[u8]) -> PyResult<Vec<Rank>> {
        py.allow_threads(|| Ok(self.inner.encode(data)?))
    }

    fn decode(&self, py: Python, tokens: Vec<Rank>) -> PyResult<PyObject> {
        let bytes = py.allow_threads(|| {
            let decoded = self.inner.decode(&tokens)?;
            Ok::<_, Error>(decoded.into_iter().flatten().copied().collect::<Vec<_>>())
        })?;

        Ok(PyBytes::new(py, &bytes).into())
    }

    fn encode_batch(&self, py: Python, data: Vec<Vec<u8>>) -> PyResult<Vec<Vec<Rank>>> {
        py.allow_threads(|| Ok(self.inner.encode_batch(data)?))
    }

    fn decode_batch(&self, py: Python, tokens: Vec<Vec<Rank>>) -> PyResult<Vec<PyObject>> {
        let batch_bytes = py.allow_threads(|| {
            let decoded = self.inner.decode_batch(tokens)?;

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
