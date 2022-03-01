from fastapi import FastAPI

from app.helpers.config import Config
from app.constants import CONFIG_JSON_PATH

def create_app():
    app = FastAPI()

    from app.api.translateAPI import translate
    app.include_router(translate)

    @app.on_event('startup')
    async def startup_event():
        config = Config()

    return app
