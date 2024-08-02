import asyncio
from collections import namedtuple
import os

from googleapiclient import discovery
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import dotenv

from ranking_challenge.request import RankingRequest
from ranking_challenge.response import RankingResponse

dotenv.load_dotenv()


app = FastAPI(
    title="Prosocial Ranking Challenge Jigsaw Query",
    description="Ranks 3 separate arms of statements based on Perspective API scores.",
    version="0.1.0",
)

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
        "ScoredStatement", "statement attr_scores statement_id"
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
        # Necessary to use asyncio.to_thread to avoid blocking the event loop. googleapiclient is synchronous.
        return await asyncio.to_thread(
            self.sync_score, attributes, statement, statement_id
        )

    def sync_score(self, attributes, statement, statement_id):
        client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
        analyze_request = {
            "comment": {"text": statement},
            "languages": ["en"],
            "requestedAttributes": {attr: {} for attr in attributes},
        }

        response = client.comments().analyze(body=analyze_request).execute()
        results = [
            (attr, response["attributeScores"][attr]["summaryScore"]["value"])
            for attr in attributes
        ]

        result = self.ScoredStatement(statement, results, statement_id)

        return result

    async def ranker(self, ranking_request: RankingRequest):
        arm = self.arm_selection(ranking_request)
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(self.score(arm, item.text, item.id))
                for item in ranking_request.items
            ]

        results = await asyncio.gather(*tasks)
        return self.arm_sort(results)

    def arm_sort(self, results):
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

        for named_tuple in results:
            combined_score = 0
            for group in named_tuple.attr_scores:
                attribute, score = group
                combined_score += weightings[attribute] * score
            reordered_statements.append((named_tuple.statement_id, combined_score))

        reordered_statements.sort(key=lambda x: x[1], reverse=True)
        filtered = [x[0] for x in reordered_statements]

        result = {
            "ranked_ids": filtered,
        }
        return result


@app.post("/rank")
async def main(ranking_request: RankingRequest) -> RankingResponse:
    ranker = PerspectiveRanker()
    results = await ranker.ranker(ranking_request)
    return results
