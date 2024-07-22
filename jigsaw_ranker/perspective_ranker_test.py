import json
import sys

from unittest.mock import patch, Mock
from fastapi.encoders import jsonable_encoder

sys.path.append(".")  # allows for importing from the current directory
from jigsaw_ranker import perspective_ranker
from ranking_challenge.fake import fake_request

from collections import namedtuple

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
    comments = fake_request(n_posts=1, n_comments=2)

    response = client.post("/rank", json=jsonable_encoder(comments))
    # Check if the request was successful (status code 200)
    if response.status_code != 200:
        assert False, f"Request failed with status code: {response.status_code}"

    result = response.json()
    assert len(result['ranked_ids']) == 3


def test_arm_selection():
    rank = perspective_ranker.PerspectiveRanker()
    comments = fake_request(n_posts=1, n_comments=2)
    comments.session.cohort = 'arm1'
    result = rank.arm_selection(comments)
    
    assert result == perspective_ranker.arm_1
    
def test_sync_score():
    rank = perspective_ranker.PerspectiveRanker()

    with patch('jigsaw_ranker.perspective_ranker.discovery') as mock_discovery:
        api_response = {
            "attributeScores": {
                "TOXICITY": {
                    "summaryScore": {
                        "value": 0.5,
                    }
                    
                }
            }
        }

        config = {'comments.return_value.analyze.return_value.execute.return_value': api_response}
        mock_client = Mock()
        mock_client.configure_mock(**config)        
        mock_build = Mock()
        mock_discovery.build = mock_build
        mock_build.return_value = mock_client
        
        result = rank.sync_score(['TOXICITY'], 'Test statement', 'test_statement_id')
        
        assert result.attr_scores == [('TOXICITY', 0.5)]
        assert result.statement == 'Test statement'
        assert result.statement_id == 'test_statement_id'
        
def test_arm_sort():
    rank = perspective_ranker.PerspectiveRanker()
    
    scored_statements = [
        rank.ScoredStatement('Test statement 2', [('TOXICITY', 0.6), ('CONSTRUCTIVE_EXPERIMENTAL', 0.2)], 'test_statement_id_2'),
        rank.ScoredStatement('Test statement', [('TOXICITY', 0.1), ('CONSTRUCTIVE_EXPERIMENTAL', 0.1)], 'test_statement_id_1'),
        rank.ScoredStatement('Test statement 3', [('TOXICITY', 0.9), ('CONSTRUCTIVE_EXPERIMENTAL', 0.3)], 'test_statement_id_3'),
    ]    
    
    result = rank.arm_sort(scored_statements)
    
    assert result == {"ranked_ids": ['test_statement_id_1', 'test_statement_id_2', 'test_statement_id_3']}    
    