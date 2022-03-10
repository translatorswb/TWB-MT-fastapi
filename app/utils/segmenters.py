import re
from typing import List, Optional, Callable

from nltk.tokenize import sent_tokenize


def nltk_sentence_segmenter(sentence: str) -> List[str]:
    return sent_tokenize(sentence)


def desegmenter(items: List[str]) -> str:
    return re.sub('(@@ )|(@@ ?$)', '', ' '.join(items))


def token_segmenter(sentence: str) -> List[str]:
    return sentence.strip().split()


def token_desegmenter(items: List[str]) -> str:
    return ' '.join(items)


def get_bpe_segmenter(
    bpe_codes_path: str,
) -> Optional[Callable[[str], List[str]]]:
    from subword_nmt import apply_bpe

    try:
        bpe = apply_bpe.BPE(codes=open(bpe_codes_path, 'r'))
    except Exception as e:
        return None
    segmenter = lambda x: bpe.process_line(x.strip()).split()
    return segmenter


def get_sentencepiece_segmenter(
    sp_model_path: str,
) -> Callable[[str], List[str]]:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    segmenter = lambda x: sp.encode_as_pieces(x)
    return segmenter


def get_sentencepiece_desegmenter(
    sp_model_path: str,
) -> Callable[[List[str]], str]:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    desentencepiece = lambda x: sp.decode_pieces(x)
    return desentencepiece
