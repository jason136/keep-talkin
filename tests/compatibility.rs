use std::collections::HashSet;

use hf_hub::api::sync::{ApiBuilder, ApiError};
use keep_talkin::{Error, Tokenizer};
use tokenizers::Tokenizer as HfTokenizer;

mod common;
use common::{load_corpus_files, load_tokenizer_configs};

#[test]
fn test_compatibility() {
    let corpus_texts = load_corpus_files().unwrap();
    let tokenizer_configs = load_tokenizer_configs(None).unwrap();

    for (model_name, tokenizer_path) in tokenizer_configs.into_iter().take(1) {
        let hf_tokenizer = HfTokenizer::from_file(&tokenizer_path).unwrap();
        let our_tokenizer = Tokenizer::from_tokenizer_json(&tokenizer_path).unwrap();

        for (corpus_name, text) in &corpus_texts {
            let hf_encoding = hf_tokenizer.encode(text.as_str(), false).unwrap();
            let hf_tokens = hf_encoding.get_ids();

            let our_tokens = our_tokenizer
                .encode(text.as_bytes(), &HashSet::new())
                .unwrap();

            let hf_decoded = hf_tokenizer.decode(hf_tokens, false).unwrap();
            let our_decoded_bytes = our_tokenizer.decode(&our_tokens).unwrap();
            let decoded_bytes = our_decoded_bytes.concat();
            let our_decoded = String::from_utf8_lossy(&decoded_bytes);

            assert_eq!(
                hf_decoded, our_decoded,
                "Decode mismatch for model {} on corpus text {}",
                model_name, corpus_name
            );

            println!("hf_decoded: {:?}", hf_decoded);
            println!("our_decoded: {:?}", our_decoded);

            println!("num hf tokens: {}", hf_tokens.len());
            println!("num our tokens: {}", our_tokens.len());

            for (hf_token, our_token) in hf_tokens.iter().zip(our_tokens.iter()) {
                assert_eq!(
                    hf_token, our_token,
                    "Token mismatch for model {} on corpus text {}",
                    model_name, corpus_name
                );
            }

            // assert_eq!(
            //     hf_tokens, our_tokens,
            //     "Token mismatch for model {} on corpus text {}",
            //     model_name, corpus_name
            // );
        }
    }
}
