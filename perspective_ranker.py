import asyncio
from collections import namedtuple
import os
import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import dotenv

from ranking_challenge.request import RankingRequest
from ranking_challenge.response import RankingResponse
from ranking_challenge.prometheus_metrics_otel_middleware import (
    expose_metrics,
    CollectorRegistry,
)
from prometheus_client import Counter

dotenv.load_dotenv()

# Create a registry
registry = CollectorRegistry()

rank_calls = Counter(
    "rank_calls", "Number of calls to the rank endpoint", registry=registry
)
exceptions_count = Counter(
    "exceptions_count", "Number of unhandled exceptions", registry=registry
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
log_level = os.getenv("LOGGING_LEVEL", "INFO")
numeric_level = logging.getLevelName(log_level.upper())
if not isinstance(numeric_level, int):
    numeric_level = logging.INFO
logger.setLevel(numeric_level)
logger.info("Starting up")

PERSPECTIVE_HOST = os.getenv(
    "PERSPECTIVE_HOST", "https://commentanalyzer.googleapis.com"
)


class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/health") == -1


# Filter out /health since it's noisy
if numeric_level != logging.DEBUG:
    logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

app = FastAPI(
    title="Prosocial Ranking Challenge Jigsaw Query",
    description="Ranks 3 separate arms of statements based on Perspective API scores.",
    version="0.1.0",
)

expose_metrics(app, endpoint="/metrics", registry=registry)


@app.middleware("http")
async def catch_all_error_handler_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        return PlainTextResponse(str(e), status_code=500)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["HEAD", "OPTIONS", "GET", "POST"],
    allow_headers=["*"],
)

perspective_baseline = [
    "CONSTRUCTIVE_EXPERIMENTAL",
    "PERSONAL_STORY_EXPERIMENTAL",
    "AFFINITY_EXPERIMENTAL",
    "COMPASSION_EXPERIMENTAL",
    "RESPECT_EXPERIMENTAL",
    "CURIOSITY_EXPERIMENTAL",
]

perspective_outrage = [
    "CONSTRUCTIVE_EXPERIMENTAL",
    "PERSONAL_STORY_EXPERIMENTAL",
    "AFFINITY_EXPERIMENTAL",
    "COMPASSION_EXPERIMENTAL",
    "RESPECT_EXPERIMENTAL",
    "CURIOSITY_EXPERIMENTAL",
    "FEARMONGERING_EXPERIMENTAL",
    "GENERALIZATION_EXPERIMENTAL",
    "SCAPEGOATING_EXPERIMENTAL",
    "MORAL_OUTRAGE_EXPERIMENTAL",
    "ALIENATION_EXPERIMENTAL",
]

perspective_toxicity = [
    "CONSTRUCTIVE_EXPERIMENTAL",
    "PERSONAL_STORY_EXPERIMENTAL",
    "AFFINITY_EXPERIMENTAL",
    "COMPASSION_EXPERIMENTAL",
    "RESPECT_EXPERIMENTAL",
    "CURIOSITY_EXPERIMENTAL",
    "TOXICITY",
    "IDENTITY_ATTACK",
    "INSULT",
    "THREAT",
]

arms = [perspective_baseline, perspective_outrage, perspective_toxicity]


class PerspectiveRanker:
    ScoredStatement = namedtuple(
        "ScoredStatement", "statement attr_scores statement_id scorable"
    )

    def __init__(self):
        self.api_key = os.environ["PERSPECTIVE_API_KEY"]

    # Selects arm based on cohort index
    def arm_selection(self, ranking_request):
        cohort = ranking_request.session.cohort
        if cohort == "perspective_baseline":
            return perspective_baseline
        elif cohort == "perspective_outrage":
            return perspective_outrage
        elif cohort == "perspective_toxicity":
            return perspective_toxicity
        else:
            raise ValueError(f"Unknown cohort: {cohort}")

    async def score(self, attributes, statement, statement_id):
        headers = {"Content-Type": "application/json"}
        data = {
            "comment": {"text": statement},
            "languages": ["en"],
            "requestedAttributes": {attr: {} for attr in attributes},
        }

        response = httpx.post(
            f"{PERSPECTIVE_HOST}/v1alpha1/comments:analyze?key={self.api_key}",
            json=data,
            headers=headers,
        ).json()

        results = []
        scorable = True
        for attr in attributes:
            try:
                score = response["attributeScores"][attr]["summaryScore"]["value"]
            except KeyError:
                score = (
                    0  # for now, set the score to 0 if it wasn't possible get a score
                )
                scorable = False

            results.append((attr, score))

        result = self.ScoredStatement(statement, results, statement_id, scorable)

        return result

    async def ranker(self, ranking_request: RankingRequest):
        arm = self.arm_selection(ranking_request)
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(self.score(arm, item.text, item.id))
                for item in ranking_request.items
            ]

        scored_statements = await asyncio.gather(*tasks)
        return self.arm_sort(scored_statements)

    def arm_sort(self, scored_statements):
        weightings = {
            "CONSTRUCTIVE_EXPERIMENTAL": 1 / 6,
            "PERSONAL_STORY_EXPERIMENTAL": 1 / 6,
            "AFFINITY_EXPERIMENTAL": 1 / 6,
            "COMPASSION_EXPERIMENTAL": 1 / 6,
            "RESPECT_EXPERIMENTAL": 1 / 6,
            "CURIOSITY_EXPERIMENTAL": 1 / 6,
            "FEARMONGERING_EXPERIMENTAL": -1 / 3,
            "GENERALIZATION_EXPERIMENTAL": -1 / 3,
            "SCAPEGOATING_EXPERIMENTAL": -1 / 9,
            "MORAL_OUTRAGE_EXPERIMENTAL": -1 / 9,
            "ALIENATION_EXPERIMENTAL": -1 / 9,
            "TOXICITY": -1 / 4,
            "IDENTITY_ATTACK": -1 / 4,
            "INSULT": -1 / 4,
            "THREAT": -1 / 4,
        }

        reordered_statements = []

        last_score = 0

        for statement in scored_statements:
            if statement.scorable:
                combined_score = 0
                for group in statement.attr_scores:
                    attribute, score = group
                    combined_score += weightings[attribute] * score
            else:
                # if a statement is not scorable, keep it with its neighbor. this prevents us from collecting
                # all unscorable statements at one end of the ranking.
                combined_score = last_score
            reordered_statements.append((statement.statement_id, combined_score))
            last_score = combined_score

        reordered_statements.sort(key=lambda x: x[1], reverse=True)
        filtered = [x[0] for x in reordered_statements]

        result = {
            "ranked_ids": filtered,
        }
        return result


@app.post("/rank")
async def main(ranking_request: RankingRequest) -> RankingResponse:
    try:
        ranker = PerspectiveRanker()
        results = await ranker.ranker(ranking_request)
        logger.debug(f"ranking results: {results}")
        rank_calls.inc()
        return RankingResponse(ranked_ids=results["ranked_ids"])
    except Exception as e:
        exceptions_count.inc()
        logger.error("Error in rank endpoint:", exc_info=True)
        raise


@app.get("/health")
def health_check():
    return {"status": "healthy"}
