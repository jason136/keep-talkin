use std::{collections::HashSet, fs, path::PathBuf};

use hf_hub::api::sync::{ApiBuilder, ApiError};
use keep_talkin::{Error, Tokenizer};
use tokenizers::Tokenizer as HfTokenizer;

const TEST_MODELS: &[&str] = &[
    "deepseek-ai/DeepSeek-V3-0324",
    "deepseek-ai/DeepSeek-R1-0528",
];

fn download_tokenizer(model_name: &str, cache_dir: Option<PathBuf>) -> Result<PathBuf, ApiError> {
    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir.unwrap_or_else(|| PathBuf::from("tests/configs")))
        .build()?;

    let repo = api.model(model_name.to_string());
    let filename = repo.get("tokenizer.json")?;

    Ok(filename)
}

fn load_corpus_files() -> Result<Vec<String>, Error> {
    let corpus_dir = PathBuf::from("tests/corpus");

    let texts = fs::read_dir(corpus_dir)?
        .map(|entry| fs::read_to_string(&entry?.path()))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(texts)
}

#[test]
fn test_compatibility() {
    let corpus_texts = load_corpus_files().unwrap();

    for model_name in TEST_MODELS {
        let tokenizer_path = download_tokenizer(model_name, None).unwrap();

        let hf_tokenizer = HfTokenizer::from_file(&tokenizer_path).unwrap();

        let our_tokenizer = Tokenizer::from_tokenizer_json(&tokenizer_path).unwrap();

        for text in &corpus_texts {
            let hf_encoding = hf_tokenizer.encode(text.as_str(), false).unwrap();
            let hf_tokens = hf_encoding.get_ids();

            let our_tokens = our_tokenizer
                .encode(text.as_bytes(), &HashSet::new())
                .unwrap();

            assert_eq!(
                hf_tokens, our_tokens,
                "Token mismatch for model {} on corpus text",
                model_name
            );

            println!("tokens match for model {} on corpus text", model_name);

            let hf_decoded = hf_tokenizer.decode(hf_tokens, false).unwrap();
            let our_decoded_bytes = our_tokenizer.decode(&our_tokens).unwrap();
            let decoded_bytes = our_decoded_bytes.concat();
            let our_decoded = String::from_utf8_lossy(&decoded_bytes);

            assert_eq!(
                hf_decoded, our_decoded,
                "Decode mismatch for model {} on corpus text",
                model_name
            );

            println!("decoded match for model {} on corpus text", model_name);
        }
    }
}
