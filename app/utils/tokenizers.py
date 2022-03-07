from typing import Callable, List


def get_moses_tokenizer(lang: str) -> Callable[[str], str]:
    from sacremoses import MosesTokenizer

    moses_tokenizer = MosesTokenizer(lang=lang)
    tokenizer = lambda x: moses_tokenizer.tokenize(x, return_str=True)
    return tokenizer


def get_moses_detokenizer(lang: str) -> Callable[[str], str]:
    from sacremoses import MosesDetokenizer

    moses_detokenizer = MosesDetokenizer(lang=lang)
    tokenizer = lambda x: moses_detokenizer.detokenize(
        x.split(), return_str=True
    )
    return tokenizer


def tokenize_with_punkset(doc: str, punkset: List[str]) -> List[str]:
    tokens = []
    doc = ' '.join(doc.split())

    curr_sent = ''
    for c in doc:
        if c in punkset:
            curr_sent += c
            if curr_sent:
                tokens.append(curr_sent.strip())
                curr_sent = ''
        else:
            curr_sent += c
    if curr_sent:
        tokens.append(curr_sent.strip())

    return tokens


def get_custom_tokenizer(punkset: List[str]) -> Callable[[str], List[str]]:
    return lambda x: tokenize_with_punkset(x, punkset)
