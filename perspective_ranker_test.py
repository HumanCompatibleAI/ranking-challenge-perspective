import os
import pytest

from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient
from aioresponses import aioresponses

import perspective_ranker
from ranking_challenge.fake import fake_request


@pytest.fixture
def app():
    app = perspective_ranker.app
    yield app


@pytest.fixture
def client(app):
    return TestClient(app)


def api_response(attributes):
    api_response = {"attributeScores": {}}

    for attr in attributes:
        api_response["attributeScores"][attr] = {
            "summaryScore": {
                "value": 0.5,
            }
        }

    return api_response


def test_rank(client):
    comments = fake_request(n_posts=1, n_comments=2)
    comments.session.cohort = "perspective_baseline"

    with aioresponses() as mocked:
        mocked.post(
            perspective_ranker.PERSPECTIVE_URL,
            payload=api_response(perspective_ranker.perspective_baseline),
            repeat=True
        )

        response = client.post("/rank", json=jsonable_encoder(comments))

    assert response.status_code == 200
    result = response.json()
    assert len(result["ranked_ids"]) == 3

def test_rank_no_score(client):
    comments = fake_request(n_posts=1, n_comments=2)
    comments.session.cohort = "perspective_baseline"

    with aioresponses() as mocked:
        mocked.post(
            perspective_ranker.PERSPECTIVE_URL,
            payload={},
            repeat=True
        )

        response = client.post("/rank", json=jsonable_encoder(comments))

    assert response.status_code == 200
    result = response.json()
    assert len(result["ranked_ids"]) == 3


def test_arm_selection():
    rank = perspective_ranker.PerspectiveRanker()
    comments = fake_request(n_posts=1, n_comments=2)
    comments.session.cohort = "perspective_baseline"
    result = rank.arm_selection(comments)

    assert result == perspective_ranker.perspective_baseline


@pytest.mark.asyncio
async def test_score():
    rank = perspective_ranker.PerspectiveRanker()

    with aioresponses() as mocked:
        mocked.post(
            perspective_ranker.PERSPECTIVE_URL,
            payload=api_response(["TOXICITY"]),
            repeat=True
        )

        result = await rank.score(["TOXICITY"], "Test statement", "test_statement_id")

    assert result.attr_scores == [("TOXICITY", 0.5)]
    assert result.statement == "Test statement"
    assert result.statement_id == "test_statement_id"


def test_arm_sort():
    rank = perspective_ranker.PerspectiveRanker()

    scored_statements = [
        rank.ScoredStatement(
            "Test statement 2",
            [("TOXICITY", 0.6), ("REASONING_EXPERIMENTAL", 0.2)],
            "test_statement_id_2",
            True,
            0.1,
        ),
        rank.ScoredStatement(
            "Test statement",
            [("TOXICITY", 0.1), ("REASONING_EXPERIMENTAL", 0.1)],
            "test_statement_id_1",
            True,
            0.1,
        ),
        rank.ScoredStatement(
            "Test statement",
            [("TOXICITY", 0), ("REASONING_EXPERIMENTAL", 0)],
            "test_statement_id_unscorable",
            False,
            0.1,
        ),
        rank.ScoredStatement(
            "Test statement 3",
            [("TOXICITY", 0.9), ("REASONING_EXPERIMENTAL", 0.3)],
            "test_statement_id_3",
            True,
            0.1,
        ),
    ]

    result = rank.arm_sort(perspective_ranker.perspective_toxicity, scored_statements)

    assert result["ranked_ids"] == [
        "test_statement_id_1",
        "test_statement_id_unscorable",
        "test_statement_id_2",
        "test_statement_id_3",
    ]
