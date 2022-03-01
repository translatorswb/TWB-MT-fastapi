from fastapi import FastAPI

def create_app():
    app = FastAPI()

    from app.api.translateAPI import translate
    app.include_router(translate, prefix='/api/v1/translate')

    return app
