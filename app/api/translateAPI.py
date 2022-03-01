from typing import List, Optional, Dict, Callable

from fastapi import Header, APIRouter, HTTPException
from pydantic import BaseModel

from app.helpers.config import Config
from app.utils.utils import get_model_id, parse_model_id

translate = APIRouter(prefix='/api/v1/translate')
config = Config()


def map_lang_to_closest(lang):
    if lang in config.language_codes:
        return lang
    elif '_' in lang:
        superlang = lang.split('_')[0]
        if superlang in config.language_codes:
            return superlang
    return ''


def do_translate(model_id, text):
    # print("do_translate")
    # print(text)

    if model_id in config.loaded_models:
        if config.loaded_models[model_id]['sentence_segmenter']:
            sentence_batch = config.loaded_models[model_id]['sentence_segmenter'](text)
        else:
            sentence_batch = [text]

        # print("sentence_batch")
        # print(sentence_batch)

        #preprocess
        # print("Preprocess")
        for step in config.loaded_models[model_id]['preprocessors']:
            sentence_batch = [step(s) for s in sentence_batch]
            # print(step)
            # print(sentence_batch)

        #translate batch (ctranslate only)
        if config.loaded_models[model_id]['translator']:
            translated_sentence_batch = config.loaded_models[model_id]['translator'](sentence_batch)
            # print("translated_sentence_batch")
            # print(translated_sentence_batch)
        else:
            translated_sentence_batch = sentence_batch

        #postprocess
        # print("Postprocess")
        tgt_sentences = translated_sentence_batch
        for step in config.loaded_models[model_id]['postprocessors']:
            tgt_sentences = [step(s) for s in tgt_sentences]
            # print(step)
            # print(tgt_sentences)

        tgt_text = " ".join(tgt_sentences)
        # print("tgt_text")
        # print(tgt_text)

        return tgt_text
    else:
        return 0

#HTTP operations
class TranslationRequest(BaseModel):
    src: str
    tgt: str
    alt: Optional[str] = None
    text: str

class BatchTranslationRequest(BaseModel):
    src: str
    tgt: str
    alt: Optional[str] = None
    texts: List[str]

class TranslationResponse(BaseModel):
    translation: str

class BatchTranslationResponse(BaseModel):
    translation: List[str]

class LanguagesResponse(BaseModel):
    models: Dict
    languages: Dict

@translate.post('/', status_code=200)
async def translate_sentence(request: TranslationRequest):

    model_id = get_model_id(map_lang_to_closest(request.src), map_lang_to_closest(request.tgt), request.alt)

    if not model_id in config.loaded_models:
        raise HTTPException(status_code=404, detail="Language pair %s is not supported."%model_id)
    
    translation = do_translate(model_id, request.text)

    response = TranslationResponse(translation=translation)
    return response

@translate.post('/batch', status_code=200)
async def translate_batch(request: BatchTranslationRequest):
    print(request.texts)
    print(type(request.texts))

    model_id = get_model_id(map_lang_to_closest(request.src), map_lang_to_closest(request.tgt), request.alt)

    if not model_id in config.loaded_models:
        raise HTTPException(status_code=404, detail="Language pair %s is not supported."%model_id)
    
    translated_batch = []
    for sentence in request.texts:
        translation = do_translate(model_id, sentence)
        translated_batch.append(translation)

    response = BatchTranslationResponse(translation=translated_batch)
    return response

@translate.get('/', status_code=200)
async def languages():
    languages_list = {}
    for model_id in config.loaded_models.keys():
        source, target, alt = parse_model_id(model_id)
        if not source in languages_list:
            languages_list[source] = {}
        if not target in languages_list[source]:
            languages_list[source][target] = []

        languages_list[source][target].append(model_id)

    return LanguagesResponse(languages=config.language_codes, models=languages_list)
