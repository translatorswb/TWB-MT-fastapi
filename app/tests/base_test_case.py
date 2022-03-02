from fastapi.testclient import TestClient

from main import app
from app.helpers.config import Config


class BaseTestCase:
    API_VERSION = 1
    SERVICE = 'translate'

    def setup(self):
        self.client = TestClient(app)
        self.config = Config()

    def get_endpoint(self, endpoint: str = '/') -> str:
        endpoint = f'/{endpoint}' if not endpoint.startswith('/') else endpoint
        return f'/api/v{self.API_VERSION}/{self.SERVICE}{endpoint}'
