use std::{fs, path::PathBuf};

use hf_hub::api::sync::{ApiBuilder, ApiError};
use keep_talkin::Error;

pub(crate) const TEST_MODELS: &[&str] = &[
    // "meta-llama/Meta-Llama-3-70B-Instruct",
    // "meta-llama/Llama-3.1-405B-Instruct",
    // "meta-llama/Llama-3.2-90B-Vision-Instruct",
    // "meta-llama/Llama-3.3-70B-Instruct",
    // "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    // "mistralai/Mistral-Small-24B-Instruct-2501",
    // "mistralai/Mixtral-8x22B-v0.1",
    "deepseek-ai/DeepSeek-V3-0324",
    "deepseek-ai/DeepSeek-R1-0528",
];

pub(crate) const CORPUS_FILES: &[(&str, &str)] = &[
    (
        "shakespeare.txt",
        "https://www.gutenberg.org/files/100/100-0.txt",
    ),
    (
        "king_james_bible.txt",
        "https://www.gutenberg.org/files/10/10-0.txt",
    ),
    (
        "rfc_index.txt",
        "https://www.rfc-editor.org/rfc/rfc-index.txt",
    ),
    (
        "naughty_strings.txt",
        "https://raw.githubusercontent.com/minimaxir/big-list-of-naughty-strings/master/blns.txt",
    ),
    (
        "unicode_test.txt",
        "https://www.unicode.org/Public/UNIDATA/UnicodeData.txt",
    ),
];

pub(crate) fn load_tokenizer_configs(
    cache_dir: Option<PathBuf>,
) -> Result<Vec<(String, PathBuf)>, ApiError> {
    Ok(TEST_MODELS
        .iter()
        .map(|model_name| {
            let api = ApiBuilder::new()
                .with_token(std::env::var("HF_TOKEN").ok())
                .with_cache_dir(
                    cache_dir
                        .as_ref()
                        .unwrap_or(&PathBuf::from("tests/configs"))
                        .to_path_buf(),
                )
                .build()?;

            let repo = api.model(model_name.to_string());
            Ok((model_name.to_string(), repo.get("tokenizer.json")?))
        })
        .collect::<Result<Vec<_>, ApiError>>()?)
}

pub(crate) fn load_corpus_files() -> Result<Vec<(String, String)>, Error> {
    let corpus_dir = PathBuf::from("tests/corpus");
    fs::create_dir_all(&corpus_dir)?;

    // for (filename, url) in CORPUS_FILES {
    //     let file_path = corpus_dir.join(filename);
    //     if !file_path.exists() {
    //         let response = reqwest::blocking::get(*url)
    //             .map_err(|e| Error::IoError(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

    //         let content = response
    //             .text()
    //             .map_err(|e| Error::IoError(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

    //         fs::write(&file_path, content)?;
    //     }
    // }

    Ok(fs::read_dir(corpus_dir)?
        .map(|entry| {
            let path = entry?.path();
            Ok((
                path.file_name().unwrap().to_string_lossy().to_string(),
                fs::read_to_string(&path)?,
            ))
        })
        .filter(|result| match result {
            Ok((name, _)) => !name.is_empty(),
            Err(_) => true,
        })
        .collect::<Result<Vec<_>, Error>>()?)
}
