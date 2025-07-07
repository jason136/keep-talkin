use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use keep_talkin::Tokenizer;

const EMOJI_CORPUS: &str = include_str!("../tests/corpus/emoji.txt");
const LOREM_CORPUS: &str = include_str!("../tests/corpus/lorem_large.txt");
const TEKKEN_CONFIG: &str = include_str!("../tests/configs/tekken.json");

fn bench_tokenizer(c: &mut Criterion) {
    let configs = [("tekken", TEKKEN_CONFIG)];
    let corpus = [("emoji", EMOJI_CORPUS), ("lorem", LOREM_CORPUS)];

    for (config_name, config_data) in configs {
        let (vocab, special_vocab, pattern) = load_tokendagger_vocab_from_str(config_data).unwrap();
        let tokenizer = CoreBPE::new(&pattern, vocab, special_vocab).unwrap();

        let mut group = c.benchmark_group(format!("encode_{}", config_name));
        for (corpus_name, text) in corpus {
            group.bench_with_input(BenchmarkId::from_parameter(corpus_name), text, |b, text| {
                b.iter(|| tokenizer.encode_ordinary(text).unwrap())
            });
        }
        group.finish();

        let mut group = c.benchmark_group(format!("decode_{}", config_name));
        for (corpus_name, text) in corpus {
            let tokens = tokenizer.encode_ordinary(text).unwrap();
            group.bench_with_input(
                BenchmarkId::from_parameter(corpus_name),
                &tokens,
                |b, tokens| b.iter(|| tokenizer.decode_bytes(tokens).unwrap()),
            );
        }
        group.finish();
    }
}

criterion_group!(benches, bench_tokenizer);
criterion_main!(benches);
