import asyncio
from collections import namedtuple
import os
import logging
import time
import aiohttp
from fastapi import FastAPI, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import dotenv

from ranking_challenge.request import RankingRequest
from ranking_challenge.response import RankingResponse
from ranking_challenge.prometheus_metrics_otel_middleware import (
    expose_metrics,
    CollectorRegistry,
)
from prometheus_client import Counter, Histogram


dotenv.load_dotenv()
PERSPECTIVE_HOST = os.getenv(
    "PERSPECTIVE_HOST", "https://commentanalyzer.googleapis.com"
)
PERSPECTIVE_URL = f'{PERSPECTIVE_HOST}/v1alpha1/comments:analyze?key={os.environ["PERSPECTIVE_API_KEY"]}'


# -- Metrics --

registry = CollectorRegistry()

rank_calls = Counter(
    "rank_calls", "Number of calls to the rank endpoint", registry=registry
)
exceptions_count = Counter(
    "exceptions_count", "Number of unhandled exceptions", registry=registry
)
ranking_latency = Histogram(
    "ranking_latency_seconds",
    "Latency of ranking operations in seconds",
    registry=registry,
)
scoring_latency = Histogram(
    "scoring_latency_seconds",
    "Latency of individual scoring operations in seconds",
    registry=registry,
)
max_scoring_latency_by_request = Histogram(
    "max_scoring_latency_by_request_seconds",
    "Latency of the slowest scoring operation in a request, in seconds",
    registry=registry,
)
scoring_timeouts = Counter(
    "scoring_timeouts", "Number of scoring operations that timed out", registry=registry
)
score_distribution = Histogram(
    "score_distribution",
    "Distribution of scores for ranked items",
    buckets=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
    registry=registry,
)
unscorable_items = Counter(
    "unscorable_items", "Number of items that could not be scored", registry=registry
)
items_per_request = Histogram(
    "items_per_request",
    "Number of items per ranking request",
    buckets=(1, 5, 10, 20, 50, 100, 200, 500),
    registry=registry,
)
cohort_distribution = Counter(
    "cohort_distribution",
    "Distribution of requests across different cohorts",
    ["cohort"],
    registry=registry,
)
text_length = Histogram(
    "text_length",
    "Length of text in characters",
    buckets=(0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000),
    registry=registry,
)

# -- Logging --
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


# Filter out /health since it's noisy
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/health") == -1


if numeric_level != logging.DEBUG:
    logging.getLogger("uvicorn.access").addFilter(EndpointFilter())


# -- Ranking weights --
perspective_baseline = {
    "REASONING_EXPERIMENTAL": 1 / 6,
    "PERSONAL_STORY_EXPERIMENTAL": 1 / 6,
    "AFFINITY_EXPERIMENTAL": 1 / 6,
    "COMPASSION_EXPERIMENTAL": 1 / 6,
    "RESPECT_EXPERIMENTAL": 1 / 6,
    "CURIOSITY_EXPERIMENTAL": 1 / 6,
}

perspective_outrage = {
    "REASONING_EXPERIMENTAL": 1 / 6,
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
}

perspective_toxicity = {
    "REASONING_EXPERIMENTAL": 1 / 6,
    "PERSONAL_STORY_EXPERIMENTAL": 1 / 6,
    "AFFINITY_EXPERIMENTAL": 1 / 6,
    "COMPASSION_EXPERIMENTAL": 1 / 6,
    "RESPECT_EXPERIMENTAL": 1 / 6,
    "CURIOSITY_EXPERIMENTAL": 1 / 6,
    "TOXICITY": -1 / 4,
    "IDENTITY_ATTACK": -1 / 4,
    "INSULT": -1 / 4,
    "THREAT": -1 / 4,
}

perspective_baseline_minus_outrage_toxic = {
    "REASONING_EXPERIMENTAL": 1 / 6,
    "PERSONAL_STORY_EXPERIMENTAL": 1 / 6,
    "AFFINITY_EXPERIMENTAL": 1 / 6,
    "COMPASSION_EXPERIMENTAL": 1 / 6,
    "RESPECT_EXPERIMENTAL": 1 / 6,
    "CURIOSITY_EXPERIMENTAL": 1 / 6,
    "FEARMONGERING_EXPERIMENTAL": -1 / 6,
    "GENERALIZATION_EXPERIMENTAL": -1 / 6,
    "SCAPEGOATING_EXPERIMENTAL": -1 / 18,
    "MORAL_OUTRAGE_EXPERIMENTAL": -1 / 18,
    "ALIENATION_EXPERIMENTAL": -1 / 18,
    "TOXICITY": -1 / 8,
    "IDENTITY_ATTACK": -1 / 8,
    "INSULT": -1 / 8,
    "THREAT": -1 / 8,
}

arms = [perspective_baseline, perspective_outrage, perspective_toxicity]


# -- Ranking Logic --
SCORING_TIMEOUT = 0.45  # seconds


class PerspectiveRanker:
    ScoredStatement = namedtuple(
        "ScoredStatement", "statement attr_scores statement_id scorable latency"
    )

    def __init__(self):
        # can't call aiohttp.ClientSession() here, fails during pytest b/c no event loop is running
        self.client = None
        self.scoring_timeout = aiohttp.ClientTimeout(total=SCORING_TIMEOUT) # timeout after 450ms to allow for a retry

    # Selects arm based on cohort index
    def arm_selection(self, ranking_request):
        cohort = ranking_request.session.cohort
        if cohort == "perspective_baseline":
            return perspective_baseline
        elif cohort == "perspective_outrage":
            return perspective_outrage
        elif cohort == "perspective_toxicity":
            return perspective_toxicity
        elif cohort == "perspective_baseline_minus_outrage_toxic":
            return perspective_baseline_minus_outrage_toxic
        else:
            raise ValueError(f"Unknown cohort: {cohort}")

    async def score(self, attributes, statement, statement_id):
        # don't try to score empty text
        if not statement.strip():
            return self.ScoredStatement(statement, [], statement_id, False, 0)

        headers = {"Content-Type": "application/json"}
        data = {
            "comment": {"text": statement},
            "languages": ["en"],
            "requestedAttributes": {attr: {} for attr in attributes},
        }

        logger.debug(
            f"Sending request to Perspective API for statement_id: {statement_id}"
        )

        if self.client is None:
            # don't limit max connections
            connector = aiohttp.TCPConnector(
                limit_per_host=0,
                limit=0,
                enable_cleanup_closed=True,
                keepalive_timeout=45,
            )
            self.client = aiohttp.ClientSession(connector=connector)

        try:
            start_time = time.time()
            response_json = None
            for _ in range(0,3):
                try:
                    response = await self.client.post(
                        url=PERSPECTIVE_URL, json=data, headers=headers, timeout=self.scoring_timeout
                    )
                    response.raise_for_status()
                    response_json = await response.json()
                except asyncio.TimeoutError:
                    scoring_timeouts.inc()
                    logger.warning(
                        f"Timeout ({SCORING_TIMEOUT}s) scoring statement_id {statement_id}"
                    )
                    continue

            latency = time.time() - start_time
            scoring_latency.observe(latency)

            if not response_json:
                raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=f"Gave up after 3 timeouts while scoring statement_id {statement_id}")

            results = []
            scorable = True
            for attr in attributes:
                try:
                    score = response_json["attributeScores"][attr]["summaryScore"][
                        "value"
                    ]

                except KeyError:
                    logger.warning(
                        f"Failed to get score for attribute {attr} in statement_id {statement_id}"
                    )
                    score = 0
                    scorable = False

                results.append((attr, score))

            result = self.ScoredStatement(statement, results, statement_id, scorable, latency)
            return result

        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP error occurred for statement_id {statement_id}: {e}, response: {e.response.text}")
            raise

        except Exception as e:
            logger.error(
                f"Unexpected error occurred for statement_id {statement_id}: {e}"
            )
            raise

    def arm_sort(self, arm_weightings, scored_statements):
        reordered_statements = []

        last_score = 0

        for statement in scored_statements:
            if statement.scorable:
                combined_score = 0
                for group in statement.attr_scores:
                    attribute, score = group
                    combined_score += arm_weightings[attribute] * score
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
            "ranked_ids_with_scores": reordered_statements,  # Include scores
        }
        return result

    async def rank(self, ranking_request: RankingRequest):
        arm_weights = self.arm_selection(ranking_request)

        # Record cohort distribution
        cohort_distribution.labels(ranking_request.session.cohort).inc()

        # Record number of items per request
        items_per_request.observe(len(ranking_request.items))

        # Record length of texts
        for item in ranking_request.items:
            text_length.observe(len(item.text))

        tasks = [
            self.score(arm_weights, item.text, item.id)
            for item in ranking_request.items
        ]
        scored_statements = await asyncio.gather(*tasks)

        # Count unscorable items
        unscorable_count = sum(
            1 for statement in scored_statements if not statement.scorable
        )
        unscorable_items.inc(unscorable_count)

        max_latency = max(statement.latency for statement in scored_statements)
        max_scoring_latency_by_request.observe(max_latency)

        result = self.arm_sort(arm_weights, scored_statements)

        # Record score distribution
        for _, score in result["ranked_ids_with_scores"]:
            score_distribution.observe(score)

        return result


# -- App ---

# Global ranking singleton, so that all calls share the same aiohttp client
ranker = PerspectiveRanker()

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


@app.post("/rank")
async def main(ranking_request: RankingRequest) -> RankingResponse:
    try:
        start_time = time.time()

        results = await ranker.rank(ranking_request)

        latency = time.time() - start_time
        logger.debug(f"ranking results: {results}")
        logger.info(f'ranking {len(results["ranked_ids"])} items took time: {latency}')

        # Record metrics
        rank_calls.inc()
        ranking_latency.observe(latency)

        return RankingResponse(ranked_ids=results["ranked_ids"])

    except Exception as e:
        exceptions_count.inc()
        logger.error("Error in rank endpoint:", exc_info=True)
        raise


@app.get("/health")
def health_check():
    return {"status": "healthy"}
