import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import regex as re


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if any(not token for token in special_tokens):
        raise ValueError("特殊 token 不能为空字符串")
    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError("特殊 token 不能重复")
    vocab = {i: bytes([i]) for i in range(256)}
    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        raise ValueError("vocab_size 不足以容纳 256 个基础字节和特殊 token")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    if special_tokens:
        # 长的标记先匹配，避免一个特殊 token 是另一个的前缀。
        special_regex = "|".join(
            re.escape(token)
            for token in sorted(special_tokens, key=len, reverse=True)
        )
        parts = re.split(f"({special_regex})", text)
        special_set = set(special_tokens)
        train_segments = [part for part in parts if part not in special_set]
    else:
        train_segments = [text]

    gpt2_pat = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    raw_counts = Counter()
    for segment in train_segments:
        for word in gpt2_pat.findall(segment):
            key = tuple(bytes([b]) for b in word.encode("utf-8"))
            raw_counts[key] += 1

    words_list = []
    counts_list = []
    for word_tuple, freq in raw_counts.items():
        words_list.append(list(word_tuple))
        counts_list.append(freq)

    stats = defaultdict(int)
    indices = defaultdict(set)
    for idx, word in enumerate(words_list):
        freq = counts_list[idx]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            stats[pair] += freq
            indices[pair].add(idx)

    merges = []
    for _ in range(num_merges):
        if not stats:
            break
        best_pair = max(stats.items(), key=lambda item: (item[1], item[0]))[0]
        if stats[best_pair] <= 0:
            break
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        relevant_indices = list(indices[best_pair])

        for idx in relevant_indices:
            word = words_list[idx]
            freq = counts_list[idx]
            i = 0
            while i < len(word) - 1:
                if word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                
                    if i > 0:
                        prev_pair = (word[i - 1], word[i])
                        stats[prev_pair] -= freq
                        if stats[prev_pair] == 0:
                            del stats[prev_pair]
                    if i < len(word) - 2:
                        next_pair = (word[i + 1], word[i + 2])
                        stats[next_pair] -= freq
                        if stats[next_pair] == 0:
                            del stats[next_pair]

                    word[i] = new_token
                    del word[i + 1]

                    
                    if i > 0:
                        new_prev = (word[i - 1], word[i])
                        stats[new_prev] += freq
                        indices[new_prev].add(idx)
                    if i < len(word) - 1:
                        new_next = (word[i], word[i + 1])
                        stats[new_next] += freq
                        indices[new_next].add(idx)
                    i += 1
                else:
                    i += 1

    
        stats.pop(best_pair, None)
        indices.pop(best_pair, None)

    for pair in merges:
        vocab[len(vocab)] = pair[0] + pair[1]
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    return vocab, merges


def bytes_to_unicode():
    
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(codepoint) for codepoint in cs)))



bytes_to_uinicode = bytes_to_unicode


def save_tokenizer_files(vocab, merges, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    byte_encoder = bytes_to_unicode()
    json_vocab = {}
    for token_id, token_bytes in vocab.items():
        json_vocab[token_id] = "".join(byte_encoder[b] for b in token_bytes)
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, indent=4)
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")


def main():
 
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="训练并保存字节级 BPE 分词器")
    parser.add_argument(
        "--input", type=Path,
        default=project_root / "data" / "TinyStoriesV2-GPT4-train.txt",
    )
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument(
        "--output", type=Path,
        default=project_root / "data" / "TinyStoriesV2-GPT4-train",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"找不到训练语料：{args.input}；请放入语料或使用 --input 指定文本文件")
    print(f"开始训练 BPE 分词器（目标词表大小：{args.vocab_size}）...")
    vocab, merges = train_bpe(args.input, args.vocab_size, ["<|endoftext|>"])
    save_tokenizer_files(vocab, merges, args.output)
    print(f"完成：{len(vocab)} 个词表项、{len(merges)} 条规则；保存到 {args.output}")


if __name__ == "__main__":
    main()
