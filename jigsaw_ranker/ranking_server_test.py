import json
import sys

sys.path.append(".")  # allows for importing from the current directory
from jigsaw_ranker.sample_data import comments
from jigsaw_ranker import perspective_ranker
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def app():
    app = perspective_ranker.app
    yield app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_rank(client):
    statements = comments
    response = client.post("/jigsaw", json=comments)
            # Check if the request was successful (status code 200)
    if response.status_code != 200:
        print(f"Request failed with status code: {response.status_code}")
        try:
            print(json.dumps(response.json(), indent=2))
        except Exception:
            print(response.text)
        assert False

    result = response.json()
    print(result)
