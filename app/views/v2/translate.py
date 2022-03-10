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


translate_v2 = APIRouter(prefix='/api/v2/translate')


@translate_v2.post('/', status_code=status.HTTP_200_OK)
async def translate_sentence_async(request: TranslationRequest):
    config = Config()

    model_id = get_model_id(request.src, request.tgt)

    task = translate_text_async.delay(model_id, request.text)

    return {'uid': task.id, 'status': task.status}


@translate_v2.post('/batch', status_code=status.HTTP_200_OK)
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


@translate_v2.get('/', status_code=status.HTTP_200_OK)
async def languages() -> LanguagesResponse:
    config = Config()

    return LanguagesResponse(
        languages=config.language_codes, models=config.languages_list
    )


@translate_v2.get('/{uid}', status_code=status.HTTP_200_OK)
async def translation_async_result(uid):
    from celery.result import AsyncResult

    result = AsyncResult(uid)
    if result.successful():
        return TranslationResponse(translation=result.result)
    return {'status': result.status, 'info': result.info}
