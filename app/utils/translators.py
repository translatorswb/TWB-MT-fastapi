import os
from typing import List, Optional, Dict, Callable

from app.constants import (
    CTRANSLATE_DEVICE,
    CTRANSLATE_INTER_THREADS,
    HELSINKI_NLP,
    MODELS_ROOT_DIR,
)


def dummy_translator(content: str) -> str:
    return content


def get_ctranslator(ctranslator_model_path):
    from ctranslate2 import Translator

    ctranslator = Translator(ctranslator_model_path)
    translator = lambda x: ctranslator.translate_batch([x])[0][0][
        'tokens'
    ]  # list IN -> list OUT
    return translator


def get_batch_ctranslator(ctranslator_model_path):
    from ctranslate2 import Translator

    ctranslator = Translator(
        ctranslator_model_path,
        device=CTRANSLATE_DEVICE,
        inter_threads=CTRANSLATE_INTER_THREADS,
    )
    translator = lambda x: [
        s[0]['tokens'] for s in ctranslator.translate_batch(x)
    ]
    return translator


def get_batch_opustranslator(
    src: str, tgt: str
) -> Optional[Callable[[str], str]]:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = f'opus-mt-{src}-{tgt}'
    local_model = os.path.join(MODELS_ROOT_DIR, model_name)
    remote_model = f'{HELSINKI_NLP}/{model_name}'
    is_model_loaded, is_tokenizer_loaded = False, False

    def translator(src_texts):
        if not src_texts:
            return ''
        return tokenizer.batch_decode(
            model.generate(
                **tokenizer.prepare_seq2seq_batch(
                    src_texts=src_texts, return_tensors='pt'
                )
            ),
            skip_special_tokens=True,
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(local_model)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(remote_model)
        tokenizer.save_pretrained(local_model)
    finally:
        is_tokenizer_loaded = True

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(local_model)
    except OSError:
        model = AutoModelForSeq2SeqLM.from_pretrained(remote_model)
        model.save_pretrained(local_model)
    finally:
        is_model_loaded = True

    if is_tokenizer_loaded and is_model_loaded:
        return translator
