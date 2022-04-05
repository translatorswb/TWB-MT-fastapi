from typing import Dict

from fastapi import APIRouter, HTTPException, status
from celery.result import AsyncResult

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
from app.tasks import translate_text_async, translate_batch_async


translate_v2 = APIRouter(prefix='/api/v2/translate')


@translate_v2.post('', status_code=status.HTTP_200_OK)
@translate_v2.post('/', status_code=status.HTTP_200_OK)
async def translate_sentence_async(request: TranslationRequest):
    model_id = get_model_id(request.src, request.tgt)
    task = translate_text_async.delay(model_id, request.text)
    return {'uid': task.id, 'status': task.status}


@translate_v2.post('/batch', status_code=status.HTTP_200_OK)
async def translate_batch(
    request: BatchTranslationRequest,
):
    model_id = get_model_id(request.src, request.tgt)
    task = translate_batch_async.delay(model_id, request.texts)
    return {'uid': task.id, 'status': task.status}


@translate_v2.get('', status_code=status.HTTP_200_OK)
@translate_v2.get('/', status_code=status.HTTP_200_OK)
async def languages() -> Dict:
    config = Config()
    return {'models': config.get_all_potential_languages()}


@translate_v2.get('/{uid}', status_code=status.HTTP_200_OK)
async def translate_sentence_async_result(uid):
    result = AsyncResult(uid)
    if result.successful():
        return TranslationResponse(translation=result.result)
    return {'status': result.status, 'info': result.info}


@translate_v2.get('/batch/{uid}', status_code=status.HTTP_200_OK)
async def translate_batch_async_result(uid):
    result = AsyncResult(uid)
    if result.successful():
        return BatchTranslationResponse(translation=result.result)
    return {'status': result.status, 'info': result.info}
