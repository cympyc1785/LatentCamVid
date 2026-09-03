# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import html
import string
import warnings

import ftfy
import regex as re
from transformers import AutoTokenizer

__all__ = ['HuggingfaceTokenizer']


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def canonicalize(text, keep_punctuation_exact_string=None):
    text = text.replace('_', ' ')
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans('', '', string.punctuation))
            for part in text.split(keep_punctuation_exact_string))
    else:
        text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class HuggingfaceTokenizer:

    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        assert clean in (None, 'whitespace', 'lower', 'canonicalize')
        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        self.vocab_size = self.tokenizer.vocab_size
        self._trunc_warned = False      # truncation 경고를 한 번만 내기 위한 래치

    def __call__(self, sequence, **kwargs):
        return_mask = kwargs.pop('return_mask', False)

        # arguments
        _kwargs = {'return_tensors': 'pt'}
        if self.seq_len is not None:
            _kwargs.update({
                'padding': 'max_length',
                'truncation': True,
                'max_length': self.seq_len
            })
        _kwargs.update(**kwargs)

        # tokenization
        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(u) for u in sequence]
        ids = self.tokenizer(sequence, **_kwargs)

        #    [2026-09-03] truncation 은 조용하다 — `text_len` 을 512 -> 128 로 내린 뒤
        #    캡션이 그 길이를 넘으면 뒤가 잘린 채로 아무 말 없이 학습된다. 실측상
        #    dynpose/vista/trumans/SD 코퍼스는 최장 61 tok 이라 안전하지만
        #    worldtraj/dynamicverse 는 max 240 tok / 31%가 128 초과다.
        #    검출: "마지막 토큰이 EOS 인가"로는 못 잡는다 — HF 는 자르고 **나서** special
        #    token 을 붙이므로 잘린 시퀀스도 EOS 로 끝난다 (실측). 그래서 길이가 꽉 찬
        #    항목에 한해 truncation 없이 한 번 더 토크나이즈해 실제 길이를 잰다. 꽉 차는
        #    일 자체가 드물어 상시 비용이 아니고, 경고는 프로세스당 한 번만 — 매 step
        #    찍으면 로그가 못 쓰게 된다.
        if self.seq_len is not None and not self._trunc_warned:
            full = (ids.attention_mask.sum(dim=1) >= self.seq_len).tolist()
            if any(full):
                raw = [s for s, f in zip(sequence, full) if f]
                true_len = max(len(u) for u in self.tokenizer(raw)['input_ids'])
                if true_len > self.seq_len:
                    self._trunc_warned = True
                    warnings.warn(
                        f'[tokenizer] 캡션이 seq_len={self.seq_len} 을 넘어 **잘렸다** '
                        f'(실제 {true_len} tok). config 의 text_len 을 올릴 것 '
                        f'(이 경고는 프로세스당 한 번만 뜬다).', stacklevel=2)

        # output
        if return_mask:
            return ids.input_ids, ids.attention_mask
        else:
            return ids.input_ids

    def _clean(self, text):
        if self.clean == 'whitespace':
            text = whitespace_clean(basic_clean(text))
        elif self.clean == 'lower':
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == 'canonicalize':
            text = canonicalize(basic_clean(text))
        return text