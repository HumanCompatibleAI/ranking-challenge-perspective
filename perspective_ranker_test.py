import pytest
import aiohttp
from aioresponses import aioresponses
import asyncio

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


def api_response(attributes):
    api_response = {"attributeScores": {}}

    for attr in attributes:
        api_response["attributeScores"][attr] = {
            "summaryScore": {
                "value": 0.5,
            }
        }

    return api_response


PERSPECTIVE_URL = f"{perspective_ranker.PERSPECTIVE_HOST}/v1alpha1/comments:analyze?key=AIzaSyA4S0_Nqr8DUx7QdC2Wzqq9U9CiKp6cMmc"

async def test_rank(client):
    comments = fake_request(n_posts=1, n_comments=2)
    comments.session.cohort = "perspective_baseline"
    
    mock  = api_response(perspective_ranker.perspective_baseline)
    
    with aioresponses() as mocked:
        mocked.post(PERSPECTIVE_URL, payload=mock)
        response = await client.post("/rank", json=jsonable_encoder(comments))

        assert response.status_code == 200, f"Request failed with status code: {response.status_code}"


        result = response.json()
        assert len(result["ranked_ids"]) == 3
        
async def test_rank_no_score(client):
    comments = fake_request(n_posts=1, n_comments=2)
    comments.session.cohort = "perspective_baseline"
    
    with aioresponses() as mocked:
        mocked.post(PERSPECTIVE_URL, payload={})
        response = await client.post("/rank", json=jsonable_encoder(comments))

        if response.status_code != 200:
            assert False, f"Request failed with status code: {response.status_code}"

        result = response.json()
        assert len(result["ranked_ids"]) == 3

@pytest.mark.asyncio
async def test_arm_selection():
    rank = perspective_ranker.PerspectiveRanker()
    comments = fake_request(n_posts=1, n_comments=2)
    comments.session.cohort = "perspective_baseline"
    result = rank.arm_selection(comments)

    assert result == perspective_ranker.perspective_baseline
    await rank.close()


@pytest.mark.asyncio
async def test_score():
    rank = perspective_ranker.PerspectiveRanker()

    mock = api_response(["TOXICITY"])
    
    with aioresponses() as mocked:
        mocked.post(PERSPECTIVE_URL, payload=mock)
        result = await rank.score(["TOXICITY"], "Test statement", "test_statement_id")

        assert result.attr_scores == [("TOXICITY", 0.5)]
        assert result.statement == "Test statement"
        assert result.statement_id == "test_statement_id"
        await rank.close()


def test_arm_sort():
    rank = perspective_ranker.PerspectiveRanker()

    scored_statements = [
        rank.ScoredStatement(
            "Test statement 2",
            [("TOXICITY", 0.6), ("REASONING_EXPERIMENTAL", 0.2)],
            "test_statement_id_2",
            True,
        ),
        rank.ScoredStatement(
            "Test statement",
            [("TOXICITY", 0.1), ("REASONING_EXPERIMENTAL", 0.1)],
            "test_statement_id_1",
            True,
        ),
        rank.ScoredStatement(
            "Test statement",
            [("TOXICITY", 0), ("REASONING_EXPERIMENTAL", 0)],
            "test_statement_id_unscorable",
            False,
        ),
        rank.ScoredStatement(
            "Test statement 3",
            [("TOXICITY", 0.9), ("REASONING_EXPERIMENTAL", 0.3)],
            "test_statement_id_3",
            True,
        ),
    ]

    result = rank.arm_sort(perspective_ranker.perspective_toxicity, scored_statements)

    assert result["ranked_ids"] == [
        "test_statement_id_1",
        "test_statement_id_unscorable",
        "test_statement_id_2",
        "test_statement_id_3",
    ]

