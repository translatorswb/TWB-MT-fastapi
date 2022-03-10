from typing import Optional, List, Dict

from pydantic import BaseModel


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
