import pytest
from unittest.mock import patch, Mock

from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient

import perspective_ranker
from ranking_challenge.fake import fake_request


@pytest.fixture
def app():
    app = perspective_ranker.app
    yield app


@pytest.fixture
def client(app):
    return TestClient(app)


def mock_perspective_build(attributes):
    api_response = {"attributeScores": {}}

    for attr in attributes:
        api_response["attributeScores"][attr] = {
            "summaryScore": {
                "value": 0.5,
            }
        }

    config = {
        "comments.return_value.analyze.return_value.execute.return_value": api_response
    }
    mock_client = Mock()
    mock_client.configure_mock(**config)
    mock_build = Mock()
    mock_build.return_value = mock_client

    return mock_build


def test_rank(client):
    comments = fake_request(n_posts=1, n_comments=2)
    comments.session.cohort = "arm1"

    with patch("perspective_ranker.discovery") as mock_discovery:
        mock_discovery.build = mock_perspective_build(perspective_ranker.arm_1)

        response = client.post("/rank", json=jsonable_encoder(comments))
        # Check if the request was successful (status code 200)
        if response.status_code != 200:
            assert False, f"Request failed with status code: {response.status_code}"

        result = response.json()
        assert len(result["ranked_ids"]) == 3


def test_arm_selection():
    rank = perspective_ranker.PerspectiveRanker()
    comments = fake_request(n_posts=1, n_comments=2)
    comments.session.cohort = "arm1"
    result = rank.arm_selection(comments)

    assert result == perspective_ranker.arm_1


def test_sync_score():
    rank = perspective_ranker.PerspectiveRanker()

    with patch("perspective_ranker.discovery") as mock_discovery:
        mock_discovery.build = mock_perspective_build(["TOXICITY"])

        result = rank.sync_score(["TOXICITY"], "Test statement", "test_statement_id")

        assert result.attr_scores == [("TOXICITY", 0.5)]
        assert result.statement == "Test statement"
        assert result.statement_id == "test_statement_id"


def test_arm_sort():
    rank = perspective_ranker.PerspectiveRanker()

    scored_statements = [
        rank.ScoredStatement(
            "Test statement 2",
            [("TOXICITY", 0.6), ("CONSTRUCTIVE_EXPERIMENTAL", 0.2)],
            "test_statement_id_2",
        ),
        rank.ScoredStatement(
            "Test statement",
            [("TOXICITY", 0.1), ("CONSTRUCTIVE_EXPERIMENTAL", 0.1)],
            "test_statement_id_1",
        ),
        rank.ScoredStatement(
            "Test statement 3",
            [("TOXICITY", 0.9), ("CONSTRUCTIVE_EXPERIMENTAL", 0.3)],
            "test_statement_id_3",
        ),
    ]

    result = rank.arm_sort(scored_statements)

    assert result == {
        "ranked_ids": [
            "test_statement_id_1",
            "test_statement_id_2",
            "test_statement_id_3",
        ]
    }
