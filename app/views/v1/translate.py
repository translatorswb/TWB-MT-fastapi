from typing import List, Optional, Dict, Callable

from fastapi import Header, APIRouter, HTTPException

from app.helpers.config import Config
from app.utils.utils import get_model_id, parse_model_id
from app.models.translate import (
    BatchTranslationRequest,
    BatchTranslationResponse,
    LanguagesResponse,
    TranslationRequest,
    TranslationResponse,
)
from app.utils.translate import translate_text

translate = APIRouter(prefix='/api/v1/translate')
config = Config()


@translate.post('/', status_code=200)
async def translate_sentence(request: TranslationRequest):

    model_id = get_model_id(
        config.map_lang_to_closest(request.src),
        config.map_lang_to_closest(request.tgt),
        request.alt,
    )

    if not model_id in config.loaded_models:
        raise HTTPException(
            status_code=404,
            detail="Language pair %s is not supported." % model_id,
        )

    translation = translate_text(model_id, request.text)

    return TranslationResponse(translation=translation)


@translate.post('/batch', status_code=200)
async def translate_batch(request: BatchTranslationRequest):
    print(request.texts)
    print(type(request.texts))

    model_id = get_model_id(
        config.map_lang_to_closest(request.src),
        config.map_lang_to_closest(request.tgt),
        request.alt,
    )

    if not model_id in config.loaded_models:
        raise HTTPException(
            status_code=404,
            detail="Language pair %s is not supported." % model_id,
        )

    translated_batch = []
    for sentence in request.texts:
        translation = translate_text(model_id, sentence)
        translated_batch.append(translation)

    return BatchTranslationResponse(translation=translated_batch)


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

    return LanguagesResponse(
        languages=config.language_codes, models=languages_list
    )

