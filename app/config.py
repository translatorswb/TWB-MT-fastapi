import os
import pathlib
from functools import lru_cache


class BaseConfig:
    BASE_DIR = pathlib.Path(__file__).parent.parent

    CELERY_BROKER_URL: str = os.environ.get(
        'CELERY_BROKER_URL', 'redis://127.0.0.1:6379/0'
    )
    result_backend: str = os.environ.get(
        'CELERY_RESULT_BACKEND', 'redis://127.0.0.1:6379/0'
    )


class DevelopmentConfig(BaseConfig):
    pass


class ProductionConfig(BaseConfig):
    pass


class TestingConfig(BaseConfig):
    task_always_eager = True


@lru_cache()
def get_settings():
    config_cls_dict = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig,
    }

    config_name = os.environ.get('FASTAPI_CONFIG', 'development')
    config_cls = config_cls_dict[config_name]
    return config_cls()


settings = get_settings()
