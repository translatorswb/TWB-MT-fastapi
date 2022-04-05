import os

import pytest


os.environ['FASTAPI_CONFIG'] = 'testing'


@pytest.fixture
def settings():
    from app.config import settings as _settings

    return _settings
