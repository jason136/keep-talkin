use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use keep_talkin::Tokenizer;

mod common;
use common::{load_corpus_files, load_tokenizer_configs};

fn bench_build(c: &mut Criterion) {
    let tokenizer_configs = load_tokenizer_configs(None).unwrap();

    let mut build_group = c.benchmark_group("tokenizer_build");
    build_group.sample_size(10);
    for (model_name, tokenizer_path) in &tokenizer_configs {
        build_group.bench_function(model_name, |b| {
            b.iter(|| black_box(Tokenizer::from_tokenizer_json(tokenizer_path).unwrap()))
        });
    }
    build_group.finish();
}

fn bench_encode(c: &mut Criterion) {
    let corpus_texts = load_corpus_files().unwrap();
    let tokenizer_configs = load_tokenizer_configs(None).unwrap();

    let tokenizers: Vec<_> = tokenizer_configs
        .iter()
        .map(|(_, tokenizer_path)| Tokenizer::from_tokenizer_json(tokenizer_path).unwrap())
        .collect();

    let mut encode_group = c.benchmark_group("encode");
    encode_group.sample_size(10);
    for (tokenizer, (model_name, _)) in tokenizers.iter().zip(&tokenizer_configs) {
        encode_group.bench_function(model_name, |b| {
            b.iter(|| {
                for (_corpus_name, text) in &corpus_texts {
                    black_box(tokenizer.encode(black_box(text.as_bytes())).unwrap());
                }
            })
        });
    }
    encode_group.finish();
}

fn bench_encode_batch(c: &mut Criterion) {
    let corpus_texts = load_corpus_files().unwrap();
    let tokenizer_configs = load_tokenizer_configs(None).unwrap();

    let tokenizers: Vec<_> = tokenizer_configs
        .iter()
        .map(|(_, tokenizer_path)| Tokenizer::from_tokenizer_json(tokenizer_path).unwrap())
        .collect();

    let mut encode_batch_group = c.benchmark_group("encode_batch");
    encode_batch_group.sample_size(10);
    for (tokenizer, (model_name, _)) in tokenizers.iter().zip(&tokenizer_configs) {
        encode_batch_group.bench_function(model_name, |b| {
            b.iter(|| {
                for (_corpus_name, text) in &corpus_texts {
                    black_box(
                        tokenizer
                            .encode_batch(black_box(vec![text.as_bytes(); 100]))
                            .unwrap(),
                    );
                }
            })
        });
    }
    encode_batch_group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let corpus_texts = load_corpus_files().unwrap();
    let tokenizer_configs = load_tokenizer_configs(None).unwrap();

    let tokenizers: Vec<_> = tokenizer_configs
        .iter()
        .map(|(_, tokenizer_path)| Tokenizer::from_tokenizer_json(tokenizer_path).unwrap())
        .collect();

    let mut decode_group = c.benchmark_group("decode");
    decode_group.sample_size(10);
    for (tokenizer, (model_name, _)) in tokenizers.iter().zip(&tokenizer_configs) {
        let encoded_corpus_texts = corpus_texts
            .iter()
            .map(|(_corpus_name, text)| tokenizer.encode(text.as_bytes()).unwrap())
            .collect::<Vec<_>>();

        decode_group.bench_function(model_name, |b| {
            b.iter(|| {
                for tokens in &encoded_corpus_texts {
                    black_box(tokenizer.decode(black_box(tokens)).unwrap());
                }
            })
        });
    }
    decode_group.finish();
}

fn bench_decode_batch(c: &mut Criterion) {
    let corpus_texts = load_corpus_files().unwrap();
    let tokenizer_configs = load_tokenizer_configs(None).unwrap();

    let tokenizers: Vec<_> = tokenizer_configs
        .iter()
        .map(|(_, tokenizer_path)| Tokenizer::from_tokenizer_json(tokenizer_path).unwrap())
        .collect();

    let mut decode_batch_group = c.benchmark_group("decode_batch");
    decode_batch_group.sample_size(10);
    for (tokenizer, (model_name, _)) in tokenizers.iter().zip(&tokenizer_configs) {
        let encoded_corpus_texts = corpus_texts
            .iter()
            .map(|(_corpus_name, text)| tokenizer.encode(text.as_bytes()).unwrap())
            .collect::<Vec<_>>();

        decode_batch_group.bench_function(model_name, |b| {
            b.iter(|| {
                for tokens in &encoded_corpus_texts {
                    black_box(
                        tokenizer
                            .decode_batch(black_box(&vec![tokens.as_slice(); 100]))
                            .unwrap(),
                    );
                }
            })
        });
    }
    decode_batch_group.finish();
}

fn generate_test_text(size_mb: f64) -> String {
    let words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on",
        "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we",
        "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their",
    ];

    let target_bytes = (size_mb * 1024.0 * 1024.0) as usize;
    let mut result = String::with_capacity(target_bytes);

    while result.len() < target_bytes {
        let paragraph_length = 50 + (result.len() % 150);
        for i in 0..paragraph_length {
            if i > 0 {
                result.push(' ');
            }
            result.push_str(words[result.len() % words.len()]);
        }
        result.push_str(".\n\n");
    }

    result.truncate(target_bytes);
    result
}

fn bench_throughput(c: &mut Criterion) {
    let tokenizer_configs = load_tokenizer_configs(None).unwrap();
    let test_text = generate_test_text(1024.0); // 10MB for benchmarking

    let tokenizers: Vec<_> = tokenizer_configs
        .iter()
        .map(|(_, tokenizer_path)| Tokenizer::from_tokenizer_json(tokenizer_path).unwrap())
        .collect();

    let num_chunks = 100;
    let text_chunks = (0..num_chunks)
        .map(|i| {
            let chunk_size = test_text.len() / num_chunks;
            let start = i * chunk_size;
            let end = test_text.len().min((i + 1) * chunk_size);
            test_text[start..end].as_bytes()
        })
        .collect::<Vec<_>>();

    let mut throughput_group = c.benchmark_group("throughput");
    throughput_group.sample_size(10);
    for (tokenizer, (model_name, _)) in tokenizers.iter().zip(&tokenizer_configs) {
        throughput_group.bench_function(model_name, |b| {
            b.iter(|| black_box(tokenizer.encode_batch(&text_chunks).unwrap()))
        });
    }
    throughput_group.finish();
}

criterion_group!(build, bench_build);
criterion_group!(encode, bench_encode);
criterion_group!(encode_batch, bench_encode_batch);
criterion_group!(decode, bench_decode);
criterion_group!(decode_batch, bench_decode_batch);
criterion_group!(throughput, bench_throughput);
criterion_main!(
    build,
    encode,
    encode_batch,
    decode,
    decode_batch,
    throughput
);
