from fastapi import APIRouter, HTTPException, status

from app.helpers.config import Config
from app.utils.utils import get_model_id
from app.models.v1.translate import (
    BatchTranslationRequest,
    BatchTranslationResponse,
    LanguagesResponse,
    TranslationRequest,
    TranslationResponse,
)
from app.utils.translate import translate_text
from app.tasks import translate_text_async


translate_v1 = APIRouter(prefix='/api/v1/translate')


@translate_v1.post('/', status_code=status.HTTP_200_OK)
async def translate_sentence(
    request: TranslationRequest,
) -> TranslationResponse:
    config = Config()

    model_id = get_model_id(
        config.map_lang_to_closest(request.src),
        config.map_lang_to_closest(request.tgt),
        request.alt,
    )

    if not model_id in config.loaded_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Language pair {model_id} is not supported.',
        )

    translation = translate_text(model_id, request.text)

    return TranslationResponse(translation=translation)


@translate_v1.post('/t', status_code=status.HTTP_200_OK)
async def translate_sentence_async(request: TranslationRequest):
    config = Config()

    model_id = get_model_id(
        config.map_lang_to_closest(request.src),
        config.map_lang_to_closest(request.tgt),
        request.alt,
    )

    if not model_id in config.loaded_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Language pair {model_id} is not supported.',
        )

    task = translate_text_async.delay(model_id, request.text)

    return {'uid': task.id}


@translate_v1.post('/batch', status_code=status.HTTP_200_OK)
async def translate_batch(
    request: BatchTranslationRequest,
) -> BatchTranslationResponse:
    config = Config()

    model_id = get_model_id(
        config.map_lang_to_closest(request.src),
        config.map_lang_to_closest(request.tgt),
        request.alt,
    )

    if not model_id in config.loaded_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Language pair {model_id} is not supported.',
        )

    translated_batch = []
    for sentence in request.texts:
        translation = translate_text(model_id, sentence)
        translated_batch.append(translation)

    return BatchTranslationResponse(translation=translated_batch)


@translate_v1.get('/', status_code=status.HTTP_200_OK)
async def languages() -> LanguagesResponse:
    config = Config()

    return LanguagesResponse(
        languages=config.language_codes, models=config.languages_list
    )


@translate_v1.get('/hello', status_code=status.HTTP_200_OK)
async def say_hello():
    from app.tasks import hello
    task = hello.delay()
    return {'uid': task.id}

@translate_v1.get('/{uid}', status_code=status.HTTP_200_OK)
async def say_hello_result(uid):
    from celery.result import AsyncResult
    result = AsyncResult(uid)
    if result:
        return result.result

