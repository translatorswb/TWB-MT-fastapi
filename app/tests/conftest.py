import os

import pytest


os.environ['FASTAPI_CONFIG'] = 'testing'  # noqa


@pytest.fixture
def settings():
    from app.config import settings as _settings

    return _settings
