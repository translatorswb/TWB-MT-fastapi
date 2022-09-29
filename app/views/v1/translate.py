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
from app.constants import NLLB_LANGS_DICT, MULTIMODALCODE

translate_v1 = APIRouter(prefix='/api/v1/translate')

@translate_v1.post("", status_code=status.HTTP_200_OK)
@translate_v1.post('/', status_code=status.HTTP_200_OK)
async def translate_sentence(
    request: TranslationRequest,
) -> TranslationResponse:
    config = Config()

    src = config.map_lang_to_closest(request.src)
    tgt = config.map_lang_to_closest(request.tgt)
    use_multi = request.use_multi

    #Get regular model_id
    model_id = get_model_id(
        src=src,
        tgt=tgt,
        alt_id=request.alt
    )

    regular_model_exists = model_id in config.loaded_models
    multilingual_model_exists = config.loaded_multilingual_models

    if use_multi and multilingual_model_exists:
        print("Using multilingual")
        use_multi = True
        model_id = get_model_id(src=MULTIMODALCODE,
                                tgt=MULTIMODALCODE,
                                alt_id=request.alt)
    elif use_multi and not multilingual_model_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'No multilingual exists. Remove flag `use_multi` from request',
        )
    else:
        #Prioritize regular model if there's any
        if regular_model_exists:
            print(f"Regular model found for {model_id}")
        elif multilingual_model_exists:
            use_multi = True
            model_id = get_model_id(src=MULTIMODALCODE,
                                tgt=MULTIMODALCODE,
                                alt_id=request.alt)
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f'Language pair {model_id} is not supported.',
            )

    if use_multi:
        if model_id not in config.loaded_multilingual_models and request.alt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f'No multilingual model with alt id {request.alt} found.',
            )
        if not src in NLLB_LANGS_DICT:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f'Language {src} is not supported.',
            )
        if not tgt in NLLB_LANGS_DICT:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f'Language {tgt} is not supported.',
            )

    translation = translate_text(model_id, request.text, src, tgt, use_multi)

    return TranslationResponse(translation=translation)

#TODO: NOT REVISED WRT TO NEW MULTILINGUAL IMPLEMENTATION
@translate_v1.post('/batch', status_code=status.HTTP_200_OK)
async def translate_batch(
    request: BatchTranslationRequest,
) -> BatchTranslationResponse:
    config = Config()

    src = config.map_lang_to_closest(request.src)
    tgt = config.map_lang_to_closest(request.tgt)

    model_id = get_model_id(#request.model_type,
        src=src,
        tgt=tgt,
        alt_id=request.alt#,
        #multilingual
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
    #translated_batch = translate_text(model_id, request.texts)

    return BatchTranslationResponse(translation=translated_batch)

#TODO: NOT REVISED WRT TO NEW MULTILINGUAL IMPLEMENTATION
@translate_v1.get('', status_code=status.HTTP_200_OK)
@translate_v1.get('/', status_code=status.HTTP_200_OK)
async def languages() -> LanguagesResponse:
    config = Config()

    return LanguagesResponse(
        languages=config.language_codes, models=config.languages_list
    )
