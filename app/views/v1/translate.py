from fastapi import APIRouter, HTTPException, status
import logging

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
from app.constants import NLLB_LANGS_DICT, MULTIMODALCODE

translate_v1 = APIRouter(prefix='/api/v1/translate')

DEVDEBUG = True
logger = logging.getLogger('console_logger')

def fetch_model_data_from_request(request):
    config = Config()

    src = config.map_lang_to_closest(request.src)
    tgt = config.map_lang_to_closest(request.tgt)
    use_multi = True if request.use_multi == 'True' else False

    #Get regular model_id
    model_id = get_model_id(
        src=src,
        tgt=tgt,
        alt_id=request.alt
    )

    compatible_model_ids = config._lookup_pair_in_languages_list(src, tgt, request.alt)

    if not compatible_model_ids:
        raise HTTPException(
                status_code=406,
                detail=f'Language pair {model_id} is not supported.',
            )

    if DEVDEBUG: 
        logger.debug(f'compatible_model_ids {compatible_model_ids}')
        if use_multi:
            logger.debug(f'use_multi {use_multi}')
    
    regular_model_exists = model_id in config.loaded_models
    multilingual_model_exists_for_pair = any([mid.startswith(MULTIMODALCODE) for mid in compatible_model_ids])

    if not regular_model_exists and not use_multi and multilingual_model_exists_for_pair:
        use_multi = True

    if use_multi:
        if multilingual_model_exists_for_pair:
            #fetch multimodal 
            model_id = get_model_id(src=MULTIMODALCODE,
                                    tgt=MULTIMODALCODE,
                                    alt_id=request.alt)
        else:
            raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'No multilingual model support for pair {src}-{tgt}. Remove flag `use_multi` from request',
        )

    if DEVDEBUG: logger.debug(f'model_id {model_id}')

    return model_id, src, tgt

@translate_v1.post("", status_code=status.HTTP_200_OK)
@translate_v1.post('/', status_code=status.HTTP_200_OK)
async def translate_sentence(
    request: TranslationRequest,
) -> TranslationResponse:

    model_id, src, tgt = fetch_model_data_from_request(request)

    translation = translate_text(model_id, request.text, src, tgt)

    return TranslationResponse(translation=translation)

@translate_v1.post('/batch', status_code=status.HTTP_200_OK)
async def translate_batch(
    request: BatchTranslationRequest,
) -> BatchTranslationResponse:
    config = Config()

    model_id, src, tgt = fetch_model_data_from_request(request)

    translated_batch = []
    for sentence in request.texts:
        translation = translate_text(model_id, sentence, src, tgt)
        translated_batch.append(translation)
    
    #TODO: translated_batch = translate_text(model_id, request.texts)

    return BatchTranslationResponse(translation=translated_batch)

@translate_v1.get('', status_code=status.HTTP_200_OK)
@translate_v1.get('/', status_code=status.HTTP_200_OK)
async def languages() -> LanguagesResponse:
    config = Config()

    return LanguagesResponse(
        languages=config.language_codes, models=config.languages_list
    )
