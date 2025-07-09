import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from tokenizers import Tokenizer as HFTokenizer
import keep_talkin

# Global configuration
TOKENIZER_JSON_PATH = "tests/configs/models--deepseek-ai--DeepSeek-V3-0324/snapshots/e9b33add76883f293d6bf61f6bd89b497e80e335/tokenizer.json"
TEXT_SIZE_MB = 1024.0  # 1GB by default
BATCH_SIZE = 1000
ITERATIONS = 10
OUTPUT_FILE = "benchmark_results.png"


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    tokenizer_name: str
    text_size_mb: float
    total_tokens: int
    total_time_seconds: float
    throughput_mb_per_sec: float
    throughput_tokens_per_sec: float


def generate_test_text(size_mb: float = 1.0) -> str:
    """Generate realistic test text of specified size in MB."""
    print(f"Generating {size_mb:.1f} MB of test text...")
    
    # Common words for realistic text generation
    words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", 
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their",
        "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
        "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
        "take", "people", "into", "year", "your", "good", "some", "could", "them",
        "see", "other", "than", "then", "now", "look", "only", "come", "its", "over",
        "think", "also", "back", "after", "use", "two", "how", "our", "work", "first",
        "well", "way", "even", "new", "want", "because", "any", "these", "give", "day",
    ]
    
    target_bytes = int(size_mb * 1024 * 1024)
    text_chunks = []
    current_size = 0
    
    while current_size < target_bytes:
        # Create a paragraph with 50-200 words
        paragraph_length = random.randint(50, 200)
        paragraph_words = random.choices(words, k=paragraph_length)
        
        paragraph = " ".join(paragraph_words).capitalize()
        
        # Add random punctuation
        for i in range(random.randint(2, 5)):
            pos = random.randint(10, len(paragraph) - 10)
            if paragraph[pos] == ' ':
                paragraph = paragraph[:pos] + random.choice([',', '.', '!', '?']) + paragraph[pos:]
        
        paragraph += "\n\n"
        text_chunks.append(paragraph)
        current_size += len(paragraph.encode('utf-8'))
        
        # Progress indicator for large files
        if len(text_chunks) % 10000 == 0:
            print(f"Generated {current_size / (1024*1024):.1f} MB...")
    
    full_text = "".join(text_chunks)
    
    # Trim to exact size if needed
    if len(full_text.encode('utf-8')) > target_bytes:
        left, right = 0, len(full_text)
        while left < right:
            mid = (left + right + 1) // 2
            if len(full_text[:mid].encode('utf-8')) <= target_bytes:
                left = mid
            else:
                right = mid - 1
        full_text = full_text[:left]
    
    actual_size_mb = len(full_text.encode('utf-8')) / (1024 * 1024)
    print(f"Generated {actual_size_mb:.2f} MB of text")
    
    return full_text


class TokenizerBenchmark:
    def __init__(self, text_size_mb: float = TEXT_SIZE_MB):
        self.tokenizer_json_path = Path(TOKENIZER_JSON_PATH)
        self.test_text = generate_test_text(text_size_mb)
        self.results: List[BenchmarkResult] = []

    def load_tokenizer_config(self) -> Tuple[str, Dict, List, List]:
        with open(self.tokenizer_json_path, "r") as f:
            config = json.load(f)

        pattern = None
        if "pre_tokenizer" in config:
            if config["pre_tokenizer"]["type"] == "Sequence":
                patterns = []
                for pt in config["pre_tokenizer"]["pretokenizers"]:
                    if pt["type"] == "Split" and "pattern" in pt:
                        patterns.append(pt["pattern"]["Regex"])
                
                # Look for the main text pattern (usually the longest/most complex one)
                # Skip simple patterns like just numbers
                for p in patterns:
                    if len(p) > 20:  # Main pattern is usually longer
                        pattern = p
                        break
                
                # Fallback to first pattern if no long pattern found
                if pattern is None and patterns:
                    pattern = patterns[0]

        vocab = config.get("model", {}).get("vocab", {})
        merges = config.get("model", {}).get("merges", [])
        special_tokens = {}

        if "added_tokens" in config:
            for token in config["added_tokens"]:
                if token.get("special", False):
                    special_tokens[token["content"]] = token["id"]

        return pattern, vocab, special_tokens, merges

    def reverse_byte_level_encoding(self, token_str: str) -> bytes:
        """Convert HuggingFace byte-level encoded tokens back to bytes using KeepTalkin's exact logic."""
        result = []
        for c in token_str:
            unicode_val = ord(c)
            
            if 33 <= unicode_val <= 126:
                result.append(unicode_val)
            elif 161 <= unicode_val <= 172 or 174 <= unicode_val <= 255:
                result.append(unicode_val)
            elif 0x100 <= unicode_val <= 0x17F:
                offset = unicode_val - 0x100
                if 0 <= offset <= 32:
                    result.append(offset)
                elif 33 <= offset <= 66:
                    result.append(offset + 94)
                elif offset == 67:
                    result.append(173)
                else:
                    result.append(unicode_val & 0xFF)  # Mask to byte range
            else:
                result.append(unicode_val & 0xFF)  # Mask to byte range
        
        return bytes(result)

    def setup_tiktoken(
        self, pattern: str, vocab: Dict, special_tokens: Dict, merges: List
    ) -> tiktoken.Encoding:
        # Convert vocabulary using KeepTalkin's exact byte-level decoding logic
        mergeable_ranks = {}
        for token_str, rank in vocab.items():
            token_bytes = self.reverse_byte_level_encoding(token_str)
            mergeable_ranks[token_bytes] = rank
        
        # Convert special tokens the same way
        clean_special_tokens = {}
        for token_content, token_id in special_tokens.items():
            clean_special_tokens[token_content] = token_id

        return tiktoken.Encoding(
            name="benchmark",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=clean_special_tokens,
        )

    def setup_custom_tokenizer(self) -> keep_talkin.Tokenizer:
        return keep_talkin.Tokenizer.from_tokenizer_json(str(self.tokenizer_json_path))

    def setup_huggingface_tokenizer(self) -> HFTokenizer:
        return HFTokenizer.from_file(str(self.tokenizer_json_path))

    def benchmark_tokenizer(self, tokenizer, tokenizer_name: str) -> BenchmarkResult:
        """Benchmark tokenizer using encode_batch for throughput testing."""
        print(f"  Benchmarking {tokenizer_name}...")

        # Split text into chunks for batch processing
        chunk_size = len(self.test_text) // BATCH_SIZE
        text_chunks = []
        
        for i in range(BATCH_SIZE):
            start_idx = i * chunk_size
            if i == BATCH_SIZE - 1:
                chunk = self.test_text[start_idx:]
            else:
                end_idx = (i + 1) * chunk_size
                chunk = self.test_text[start_idx:end_idx]
            text_chunks.append(chunk)

        start_time = time.perf_counter()

        if tokenizer_name == "HuggingFace":
            # HuggingFace batch encoding
            batch_encoding = tokenizer.encode_batch(text_chunks)
            token_results = [encoding.ids for encoding in batch_encoding]
        elif tokenizer_name == "KeepTalkin":
            # KeepTalkin batch encoding
            token_results = tokenizer.encode_batch(text_chunks)
        else:  # tiktoken
            # TikToken batch encoding
            token_results = tokenizer.encode_batch(text_chunks)

        end_time = time.perf_counter()

        # Calculate statistics
        total_time = end_time - start_time
        total_tokens = sum(len(tokens) for tokens in token_results)
        text_size_mb = len(self.test_text.encode("utf-8")) / (1024 * 1024)
        
        throughput_mb_per_sec = text_size_mb / total_time
        throughput_tokens_per_sec = total_tokens / total_time

        return BenchmarkResult(
            tokenizer_name=tokenizer_name,
            text_size_mb=text_size_mb,
            total_tokens=total_tokens,
            total_time_seconds=total_time,
            throughput_mb_per_sec=throughput_mb_per_sec,
            throughput_tokens_per_sec=throughput_tokens_per_sec
        )

    def run_benchmarks(self):
        """Run all tokenizer benchmarks."""
        print(f"Running benchmarks on {len(self.test_text)} characters...")
        print(f"Batch size: {BATCH_SIZE}")
        print()

        try:
            # Load configuration
            pattern, vocab, special_tokens, merges = self.load_tokenizer_config()
            print(f"Loaded config: {len(vocab)} vocab items, {len(special_tokens)} special tokens, {len(merges)} merges")

            # Setup and benchmark tokenizers
            tokenizers = [
                ("TikToken", lambda: self.setup_tiktoken(pattern, vocab, special_tokens, merges)),
                ("KeepTalkin", self.setup_custom_tokenizer),
                ("HuggingFace", self.setup_huggingface_tokenizer)
            ]

            for name, setup_func in tokenizers:
                try:
                    tokenizer = setup_func()
                    print(f"✓ {name} tokenizer ready")
                    
                    result = self.benchmark_tokenizer(tokenizer, name)
                    self.results.append(result)
                    print(f"  {name}: {result.throughput_mb_per_sec:.2f} MB/s, "
                          f"{result.throughput_tokens_per_sec:.0f} tokens/s")
                except Exception as e:
                    print(f"✗ {name} failed: {e}")

        except Exception as e:
            print(f"Benchmark failed: {e}")
            return False

        return True

    def print_results(self):
        """Print detailed benchmark results."""
        if not self.results:
            print("No results to display!")
            return

        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)

        print(f"{'Tokenizer':<15} {'MB/s':<10} {'Tokens/s':<12} {'Time':<10} {'Tokens':<12}")
        print("-" * 70)

        for result in self.results:
            print(f"{result.tokenizer_name:<15} {result.throughput_mb_per_sec:<10.2f} "
                  f"{result.throughput_tokens_per_sec:<12.0f} "
                  f"{result.total_time_seconds:<10.2f}s {result.total_tokens:<12}")

        # Speed comparisons
        if len(self.results) > 1:
            print("\nSpeed Comparisons:")
            baseline = self.results[0]
            
            for result in self.results[1:]:
                speedup = result.throughput_mb_per_sec / baseline.throughput_mb_per_sec
                print(f"  {result.tokenizer_name} vs {baseline.tokenizer_name}: {speedup:.2f}x")

    def generate_graph(self, output_path: str = "benchmark_results.png"):
        """Generate performance comparison graph."""
        if not self.results:
            print("No results to plot!")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        names = [result.tokenizer_name for result in self.results]
        mb_per_sec = [result.throughput_mb_per_sec for result in self.results]
        tokens_per_sec = [result.throughput_tokens_per_sec for result in self.results]

        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"][:len(names)]

        # MB/s chart
        bars1 = ax1.bar(names, mb_per_sec, color=colors, alpha=0.8)
        ax1.set_ylabel("Throughput (MB/s)")
        ax1.set_title("Throughput by Data Size")
        ax1.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, value in zip(bars1, mb_per_sec):
            ax1.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + max(mb_per_sec) * 0.01,
                    f"{value:.1f}", ha="center", va="bottom", fontweight="bold")

        # Tokens/s chart
        bars2 = ax2.bar(names, tokens_per_sec, color=colors, alpha=0.8)
        ax2.set_ylabel("Throughput (Tokens/s)")
        ax2.set_title("Throughput by Token Count")
        ax2.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, value in zip(bars2, tokens_per_sec):
            ax2.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + max(tokens_per_sec) * 0.01,
                    f"{value:.0f}", ha="center", va="bottom", fontweight="bold")

        # Style improvements
        for ax in [ax1, ax2]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📊 Results saved to: {output_path}")

        # Save raw data
        data_file = Path(output_path).with_suffix(".json")
        results_dict = {result.tokenizer_name: {
            "throughput_mb_per_sec": result.throughput_mb_per_sec,
            "throughput_tokens_per_sec": result.throughput_tokens_per_sec,
            "total_time_seconds": result.total_time_seconds,
            "total_tokens": result.total_tokens,
            "text_size_mb": result.text_size_mb
        } for result in self.results}
        
        with open(data_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"📈 Raw data saved to: {data_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Tokenizer throughput benchmark")
    parser.add_argument("--text-size", type=float, default=TEXT_SIZE_MB,
                       help=f"Size of test text in MB (default: {TEXT_SIZE_MB})")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark (10 MB instead of 1 GB)")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE,
                       help=f"Output file (default: {OUTPUT_FILE})")
    
    args = parser.parse_args()
    
    # Quick mode adjustment
    text_size = 10.0 if args.quick else args.text_size
    
    print(f"Tokenizer benchmark configuration:")
    print(f"  Tokenizer JSON: {TOKENIZER_JSON_PATH}")
    print(f"  Text size: {text_size} MB")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Output: {args.output}")
    print()

    benchmark = TokenizerBenchmark(text_size)

    if benchmark.run_benchmarks():
        benchmark.print_results()
        benchmark.generate_graph(args.output)
    else:
        print("Benchmark failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
