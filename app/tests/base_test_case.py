from fastapi.testclient import TestClient

from main import app
from app.helpers.config import Config


class BaseTestCase:
    def setup(self):
        self.client = TestClient(app)
        self.config_data = {
            'languages': {
                'en': 'English',
                'fr': 'French',
            },
            'models': [
                {
                    'src': 'en',
                    'tgt': 'fr',
                    'model_type': 'opus',
                    'load': True,
                    'sentence_split': 'nltk',
                    'pipeline': {
                        'lowercase': True,
                        'translate': True,
                        'recase': True,
                    },
                },
            ],
        }
        self.config = Config(config_data=self.config_data)


class APIBaseTestCase(BaseTestCase):
    API_VERSION = 1
    SERVICE = 'translate'

    def setup(self):
        self.client = TestClient(app)
        super().setup()

    def get_endpoint(self, endpoint: str = '/') -> str:
        endpoint = f'/{endpoint}' if not endpoint.startswith('/') else endpoint
        return f'/api/v{self.API_VERSION}/{self.SERVICE}{endpoint}'
