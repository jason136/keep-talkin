use std::time::Instant;

use keep_talkin::Tokenizer;
use tokenizers::Tokenizer as HfTokenizer;

mod common;
use common::{load_corpus_files, load_tokenizer_configs};

#[test]
fn test_compatibility() {
    let corpus_texts = load_corpus_files().unwrap();
    let tokenizer_configs = load_tokenizer_configs(None).unwrap();

    let corpus_bytes = corpus_texts
        .iter()
        .map(|(_, text)| text.as_bytes())
        .collect::<Vec<_>>();

    let corpus_strs = corpus_texts
        .iter()
        .map(|(_, text)| text.as_str())
        .collect::<Vec<_>>();

    for (_, tokenizer_path) in tokenizer_configs.into_iter().take(1) {
        let hf_tokenizer = HfTokenizer::from_file(&tokenizer_path).unwrap();
        let our_tokenizer = Tokenizer::from_tokenizer_json(&tokenizer_path).unwrap();

        let start = Instant::now();
        let hf_encodings = hf_tokenizer
            .encode_batch(corpus_strs.clone(), false)
            .unwrap();
        let hf_tokens: Vec<_> = hf_encodings.iter().map(|enc| enc.get_ids()).collect();
        println!("hf batch encoding time: {:?}", start.elapsed());

        let start = Instant::now();
        let our_tokens = our_tokenizer.encode_batch(&corpus_bytes).unwrap();
        println!("our batch encoding time: {:?}", start.elapsed());

        let start = Instant::now();
        let hf_decoded_strings = hf_tokenizer.decode_batch(&hf_tokens, false).unwrap();
        println!("hf batch decoding time: {:?}", start.elapsed());

        let start = Instant::now();
        let our_decoded_strings = our_tokenizer
            .decode_batch(&our_tokens)
            .unwrap()
            .into_iter()
            .map(|bytes| String::from_utf8_lossy(&bytes.concat()).to_string())
            .collect::<Vec<_>>();

        println!("our batch decoding time: {:?}", start.elapsed());

        assert_eq!(hf_decoded_strings, our_decoded_strings,);

        assert_eq!(hf_tokens, our_tokens);
    }
}
