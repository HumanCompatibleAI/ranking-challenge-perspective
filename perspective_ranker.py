import asyncio
from collections import namedtuple
import os
import logging
import time
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import dotenv
import threading

from ranking_challenge.request import RankingRequest
from ranking_challenge.response import RankingResponse
from ranking_challenge.prometheus_metrics_otel_middleware import (
    expose_metrics,
    CollectorRegistry,
)
from prometheus_client import Counter, Histogram

from pyinstrument import Profiler
from pyinstrument.renderers.html import HTMLRenderer
from pyinstrument.renderers.speedscope import SpeedscopeRenderer

# Global counter to track the max number of open requests
open_requests = 0
max_open_requests = 0
lock = threading.Lock()
global request_start_time


# each post requires a single request, so see if we can do them all at once
KEEPALIVE_CONNECTIONS = 50

# keep connections a long time to save on tcp connection startup latency
KEEPALIVE_EXPIRY = 60 * 10

dotenv.load_dotenv()
PERSPECTIVE_HOST = os.getenv(
    "PERSPECTIVE_HOST", "https://commentanalyzer.googleapis.com"
)

# Create a registry
registry = CollectorRegistry()

# -- Metrics --
rank_calls = Counter(
    "rank_calls", "Number of calls to the rank endpoint", registry=registry
)
exceptions_count = Counter(
    "exceptions_count", "Number of unhandled exceptions", registry=registry
)
ranking_latency = Histogram(
    "ranking_latency_seconds", 
    "Latency of ranking operations in seconds", 
    registry=registry
)
score_distribution = Histogram(
    "score_distribution", 
    "Distribution of scores for ranked items",
    buckets=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
    registry=registry
)
unscorable_items = Counter(
    "unscorable_items", 
    "Number of items that could not be scored", 
    registry=registry
)
items_per_request = Histogram(
    "items_per_request", 
    "Number of items per ranking request",
    buckets=(1, 5, 10, 20, 50, 100, 200, 500),
    registry=registry
)
cohort_distribution = Counter(
    "cohort_distribution", 
    "Distribution of requests across different cohorts",
    ["cohort"],
    registry=registry
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

@app.middleware("http")
async def profile_request(request: Request, call_next: callable):
    """Profile the current request

    Taken from https://pyinstrument.readthedocs.io/en/latest/guide.html#profile-a-web-request-in-fastapi
    with small improvements.

    """
    # we map a profile type to a file extension, as well as a pyinstrument profile renderer
    profile_type_to_ext = {"html": "html", "speedscope": "speedscope.json"}
    profile_type_to_renderer = {
        "html": HTMLRenderer,
        "speedscope": SpeedscopeRenderer,
    }

    # if the `profile=true` HTTP query argument is passed, we profile the request
    if True: # request.query_params.get("profile", False):

        # The default profile format is speedscope
        profile_type = request.query_params.get("profile_format", "speedscope")

        # we profile the request along with all additional middlewares, by interrupting
        # the program every 1ms1 and records the entire stack at that point
        with Profiler(interval=0.001, async_mode="enabled") as profiler:
            response = await call_next(request)

        # we dump the profiling into a file
        extension = profile_type_to_ext[profile_type]
        renderer = profile_type_to_renderer[profile_type]()
        with open(f"profile.{extension}", "w") as out:
            out.write(profiler.output(renderer=renderer))
        return response

    # Proceed without profiling
    return await call_next(request)


# -- Ranking weights --
perspective_baseline = {
    "CONSTRUCTIVE_EXPERIMENTAL": 1 / 6,
    "PERSONAL_STORY_EXPERIMENTAL": 1 / 6,
    "AFFINITY_EXPERIMENTAL": 1 / 6,
    "COMPASSION_EXPERIMENTAL": 1 / 6,
    "RESPECT_EXPERIMENTAL": 1 / 6,
    "CURIOSITY_EXPERIMENTAL": 1 / 6,
}

perspective_outrage = {
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
}

perspective_toxicity = {
    "CONSTRUCTIVE_EXPERIMENTAL": 1 / 6,
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
    "CONSTRUCTIVE_EXPERIMENTAL": 1 / 6,
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

# -- Main ranker -- 
class PerspectiveRanker:
    ScoredStatement = namedtuple(
        "ScoredStatement", "statement attr_scores statement_id scorable"
    )

    def __init__(self):
        self.api_key = os.environ["PERSPECTIVE_API_KEY"]
        limits = httpx.Limits(
            max_keepalive_connections=KEEPALIVE_CONNECTIONS,
            max_connections=None,
            keepalive_expiry=KEEPALIVE_EXPIRY,
        )
        self.httpx_client = httpx.AsyncClient(limits=limits)

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

    def process_response(self, response, statement_id, attributes, statement):
            response.raise_for_status()
            response_json = response.json()

            #logger.debug(f"Response for statement_id {statement_id}: {response_json}")

            results = []
            scorable = True
            for attr in attributes:
                try:
                    score = response_json["attributeScores"][attr]["summaryScore"]["value"]

                except KeyError:
                    logger.warning(f"Failed to get score for attribute {attr} in statement_id {statement_id}")
                    score = 0
                    scorable = False

                results.append((attr, score))

            result = self.ScoredStatement(statement, results, statement_id, scorable)
            return result

    async def score(self, attributes, statement, statement_id):
        global request_start_time

        headers = {"Content-Type": "application/json"}
        data = {
            "comment": {"text": statement},
            "languages": ["en"],
            "requestedAttributes": {attr: {} for attr in attributes},
        }

        # don't try to score empty text
        if not statement.strip():
            return self.ScoredStatement(statement, [], statement_id, False)


        global open_requests, max_open_requests, lock
        with lock:
            open_requests += 1
            max_open_requests = max(max_open_requests, open_requests)

        logger.info(f"Sending request for {statement_id} at time {time.time() - request_start_time}, outstanding {open_requests}")
        # logger.debug(f"Request payload: {data}")  don't log text, it's sensitive

        try:
            response = await self.httpx_client.post(
                f"{PERSPECTIVE_HOST}/v1alpha1/comments:analyze?key={self.api_key}",
                json=data,
                headers=headers,
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred for statement_id {statement_id}: {e}")
            logger.error(f"Response content: {e.response.text}")
            raise
        finally:
            with lock:
                open_requests -= 1


        logger.info(f"Recevied response for {statement_id} at time {time.time() - request_start_time}, outstanding {open_requests}")

        try:
            return self.process_response(response, statement_id, attributes, statement)
        
        except Exception as e:
            logger.error(f"Unexpected error occurred for statement_id {statement_id}: {e}")
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
        
        tasks = [
            self.score(arm_weights, item.text, item.id)
            for item in ranking_request.items
        ]
        scored_statements = await asyncio.gather(*tasks)
        logger.info(f"Maximal number of simultaneous open requests: {max_open_requests}")

        # Count unscorable items
        unscorable_count = sum(1 for statement in scored_statements if not statement.scorable)
        unscorable_items.inc(unscorable_count)
        
        result = self.arm_sort(arm_weights, scored_statements)
        
        # Record score distribution
        for _, score in result['ranked_ids_with_scores']:
            score_distribution.observe(score)
        
        return result
    

# Global singleton, so that all calls share the same httpx client
ranker = PerspectiveRanker()


@app.post("/rank")
async def main(ranking_request: RankingRequest) -> RankingResponse:
    global request_start_time
    request_start_time = time.time()

    try:
        start_time = time.time() 

        results = await ranker.rank(ranking_request)

        latency = time.time() - start_time 
        logger.debug(f"ranking results for {len(results['ranked_ids'])} items : {results["ranked_ids_with_scores"]}")
        logger.debug(f"ranking took time: {latency}")
        
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
