import regex as re
from collections.abc import Iterable
class Tokenizer:
    def __init__(self,vocab,merges,special_tokens=None):
        self.vocab=vocab
        self.id_to_byte=vocab
        self.merges={pair:i for i ,pair in enumerate(merges)}

        self.byte_to_id={
            token_bytes:token_id
            for token_id,token_bytes in vocab.items()

        }

        self.special_tokens=list(special_tokens or [])
        if self.special_tokens:
            sorted_special=sorted(special_tokens,key=len,reverse=True)
            special_parttern="|".join(re.escape(t)for t in sorted_special)
            self.special_regex=re.compile(special_parttern)
        else:
            self.special_regex=None
        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")






    def encode(self,text):
        if not text:
            return []
        if not self.special_regex:
            return self._encode_text_segment(text)
        tokens=[]
        last_pos=0
        for match in self.special_regex.finditer(text):
            pre_text=text[last_pos:match.start()]
            tokens.extend(self._encode_text_segment(pre_text))
            special_tok=match.group()
            tokens.append(self.byte_to_id[special_tok.encode("utf-8")])
            last_pos=match.end()
        remaining_text=text[last_pos:]
        if remaining_text:
            tokens.extend(self._encode_text_segment(remaining_text))
        return tokens

            
    def _encode_text_segment(self,text:str)-> list[int] :


        ids=[]
        pre_tokens=self.gpt2_pat.findall(text)
        for p_tok in pre_tokens:
            byte_parts=[bytes([i]) for i in p_tok.encode("utf-8")]
            while len(byte_parts)>=2:

                best_pair=None
                min_rank=float("inf")
                for i in range(len(byte_parts)-1):
                    pair=(byte_parts[i],byte_parts[i+1])
                    if pair in self.merges:
                        rank=self.merges[pair]
                        if rank<min_rank:
                            min_rank=rank
                            best_pair=pair
                if best_pair==None:
                    break 
                new_byte_pairs=[]
                i=0
                while i <len(byte_parts):
                    if i<len(byte_parts)-1 and best_pair==(byte_parts[i],byte_parts[i+1]):
                        new_byte_pairs.append(best_pair[0]+best_pair[1])
                        i+=2
                    else:
                        new_byte_pairs.append(byte_parts[i])
                        i+=1
                byte_parts = new_byte_pairs
            for part in byte_parts:
                ids.append(self.byte_to_id[part])
        return ids
                



    def decode(self,ids):
        token_bytes=[
            self.vocab[token_id]
            for token_id in ids
        ]
        combined =b"".join(token_bytes)
        return combined.decode("utf-8", errors="replace")
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        内存高效的迭代编码器。
        
        参数:
            iterable: 一个可迭代的字符串对象（例如文件句柄）。
        返回:
            一个生成器，逐个产出编码后的 ID。用于处理无法一次性读入内存的大文件。
        """
        for chunk in iterable:
            # 对每一块文本进行编码，并通过 yield 吐出结果
            yield from self.encode(chunk)
    