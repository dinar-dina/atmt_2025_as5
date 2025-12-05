#!/usr/bin/env python3
"""
Calculate average SentencePiece token length for translation output files.

Usage:
    python compute_avg_length.py output1.txt output2.txt ...
"""

import sentencepiece as spm
import sys
from pathlib import Path

def load_tokenizer(model_path):
    """Load SentencePiece tokenizer."""
    try:
        tgt = spm.SentencePieceProcessor()
        tgt.load(model_path)
        return tgt
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

def compute_length_stats(tokenizer, path):
    """Compute token length statistics for a file."""
    lengths = []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        ids = tokenizer.EncodeAsIds(line)
                        lengths.append(len(ids))
                    except Exception as e:
                        print(f"Warning: Error encoding line {line_num} in {path}: {e}", 
                              file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: File '{path}' not found", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error reading '{path}': {e}", file=sys.stderr)
        return None
    
    if not lengths:
        print(f"Warning: No valid lines found in {path}", file=sys.stderr)
        return None
    
    return {
        'avg': sum(lengths) / len(lengths),
        'min': min(lengths),
        'max': max(lengths),
        'total': sum(lengths),
        'count': len(lengths)
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_avg_length.py <file1> <file2> ...", file=sys.stderr)
        print("\nExample:")
        print("  python compute_avg_length.py output_base.txt output_relative.txt output_node.txt")
        sys.exit(1)
    
    # Load tokenizer
    tokenizer = load_tokenizer("toy_example/tokenizers/en-bpe-1000.model")
    
    files = sys.argv[1:]
    results = []
    
    print("Computing average token lengths...\n")
    
    for filepath in files:
        stats = compute_length_stats(tokenizer, filepath)
        
        if stats:
            # Store for comparison table
            results.append((Path(filepath).name, stats))
            
            # Print individual stats
            print(f"{filepath}:")
            print(f"  Average length: {stats['avg']:.2f} tokens")
            print(f"  Range: {stats['min']}-{stats['max']} tokens")
            print(f"  Total tokens: {stats['total']}")
            print(f"  Sentences: {stats['count']}")
            print()
    
    # Print comparison table
    if len(results) > 1:
        print("\n" + "="*60)
        print("COMPARISON TABLE")
        print("="*60)
        print(f"{'File':<30} {'Avg Length':>12} {'Min':>6} {'Max':>6}")
        print("-"*60)
        for filename, stats in results:
            print(f"{filename:<30} {stats['avg']:>12.2f} {stats['min']:>6} {stats['max']:>6}")
        print("="*60)

if __name__ == "__main__":
    main()