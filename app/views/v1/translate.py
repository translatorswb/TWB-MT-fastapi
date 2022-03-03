from typing import List, Optional, Dict, Callable

from fastapi import Header, APIRouter, HTTPException, status

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


@translate.post('/', status_code=status.HTTP_200_OK)
async def translate_sentence(request: TranslationRequest):

    model_id = get_model_id(
        config.map_lang_to_closest(request.src),
        config.map_lang_to_closest(request.tgt),
        request.alt,
    )

    if not model_id in config.loaded_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Language pair {model_id} is not supported.'
        )

    translation = translate_text(model_id, request.text)

    return TranslationResponse(translation=translation)


@translate.post('/batch', status_code=status.HTTP_200_OK)
async def translate_batch(request: BatchTranslationRequest):
    model_id = get_model_id(
        config.map_lang_to_closest(request.src),
        config.map_lang_to_closest(request.tgt),
        request.alt,
    )

    if not model_id in config.loaded_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Language pair {model_id} is not supported.'
        )

    translated_batch = []
    for sentence in request.texts:
        translation = translate_text(model_id, sentence)
        translated_batch.append(translation)

    return BatchTranslationResponse(translation=translated_batch)


@translate.get('/', status_code=status.HTTP_200_OK)
async def languages():
    return LanguagesResponse(
        languages=config.language_codes, models=config.languages_list
    )

