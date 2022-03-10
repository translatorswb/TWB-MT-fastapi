from fastapi import FastAPI

from app.helpers.config import Config
from . import tasks


def create_app() -> FastAPI:
    app = FastAPI()

    from app.celery_utils import create_celery

    app.celery_app = create_celery()

    from app.views.v1.translate import translate_v1

    app.include_router(translate_v1)

    from app.views.v2.translate import translate_v2

    app.include_router(translate_v2)

    # @app.on_event('startup')
    # async def startup_event() -> None:
    #    config = Config(load_all_models=True)

    return app
