import os
import importlib
from typing import Optional, Callable

from app.constants import HELSINKI_NLP, NLLB_LANGS_DICT
from app.settings import (
    CTRANSLATE_DEVICE,
    CTRANSLATE_INTER_THREADS,
    MODELS_ROOT_DIR,
)

def dummy_translator(content: str) -> str:
    return content

def get_custom_translator(translator_id: str) -> Callable:
    translator_main_module = importlib.import_module('app.customtranslators.' + translator_id + '.src.interface')

    translator = lambda x: [translator_main_module.translate(i) for i in x] # list IN -> list OUT
    return translator

def get_ctranslator(ctranslator_model_path: str) -> Callable:
    from ctranslate2 import Translator

    ctranslator = Translator(ctranslator_model_path)
    translator = lambda x: ctranslator.translate_batch([x])[0][0][
        'tokens'
    ]  # list IN -> list OUT
    return translator


def get_batch_ctranslator(ctranslator_model_path: str) -> Callable:
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

    def translator(src_texts, src=None, tgt=None):
        print('get_batch_opustranslator/translator', src_texts)
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
    return None

def get_batch_opusbigtranslator(
    src: str, tgt: str
) -> Optional[Callable[[str], str]]:
    from transformers import MarianMTModel, MarianTokenizer

    model_name = f'opus-mt-tc-big-{src}-{tgt}'
    local_model = os.path.join(MODELS_ROOT_DIR, model_name)
    remote_model = f'{HELSINKI_NLP}/{model_name}'
    is_model_loaded, is_tokenizer_loaded = False, False

    def translator(src_texts, src=None, tgt=None):
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
        tokenizer = MarianTokenizer.from_pretrained(local_model)
    except OSError:
        tokenizer = MarianTokenizer.from_pretrained(remote_model)
        tokenizer.save_pretrained(local_model)
    finally:
        is_tokenizer_loaded = True

    try:
        model = MarianMTModel.from_pretrained(local_model)
    except OSError:
        model = MarianMTModel.from_pretrained(remote_model)
        model.save_pretrained(local_model)
    finally:
        is_model_loaded = True

    if is_tokenizer_loaded and is_model_loaded:
        return translator
    return None


def get_batch_nllbtranslator() -> Optional[Callable[[str], str]]:

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

    model_name = 'nllb-distilled-600M'
    local_model = os.path.join(MODELS_ROOT_DIR, model_name)
    remote_model = "facebook/nllb-200-distilled-600M"

    is_model_loaded, is_tokenizer_loaded = False, False

    def translator(src_texts, src, tgt):
        nllb_src = NLLB_LANGS_DICT.get(src)
        nllb_tgt = NLLB_LANGS_DICT.get(tgt)

        print(f'get_batch_nllbtranslator/translator {nllb_src}-{nllb_tgt}', src_texts)

        if not src_texts:
            return ''
        else:
            nllb_translator = pipeline(
                "translation",
                model=model,
                tokenizer=tokenizer,
                src_lang=nllb_src,
                tgt_lang=nllb_tgt,
            )

            return [nllb_translator(text, max_length=400)[0]["translation_text"] 
                    for text in src_texts]

    try:
        tokenizer = AutoTokenizer.from_pretrained(local_model)
    except Exception as e: #TODO need to change this exception type
        tokenizer = AutoTokenizer.from_pretrained(remote_model)
        tokenizer.save_pretrained(local_model)
    finally:
        is_tokenizer_loaded = True

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(local_model)
    except Exception as e: 
        model = AutoModelForSeq2SeqLM.from_pretrained(remote_model)
        model.save_pretrained(local_model)
    finally:
        is_model_loaded = True


    if is_tokenizer_loaded and is_model_loaded:
        return translator
    return None
