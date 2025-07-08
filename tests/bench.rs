use std::{collections::HashSet, hint::black_box};

use criterion::{Criterion, criterion_group, criterion_main};
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
                    black_box(
                        tokenizer
                            .encode(black_box(text.as_bytes()), &HashSet::new())
                            .unwrap(),
                    );
                }
            })
        });
    }
    encode_group.finish();
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
            .map(|(_corpus_name, text)| tokenizer.encode(text.as_bytes(), &HashSet::new()).unwrap())
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

criterion_group!(build, bench_build);
criterion_group!(encode, bench_encode);
criterion_group!(decode, bench_decode);
criterion_main!(build, encode, decode);
