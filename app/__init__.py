from fastapi import FastAPI

from app.helpers.config import Config


def create_app() -> FastAPI:
    app = FastAPI()

    from app.views.v1.translate import translate_v1

    app.include_router(translate_v1)

    @app.on_event('startup')
    async def startup_event() -> None:
        config = Config(load_all_models=True)

    return app
