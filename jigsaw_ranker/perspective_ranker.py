import asyncio
import numpy as np
from typing import List
import asyncio
from googleapiclient import discovery
import json
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import sys
from ranking_challenge.request import RankingRequest
from ranking_challenge.response import RankingResponse
sys.path.append(".")
# from jigsaw_ranker.sample_data import comments


API_KEY = 'AIzaSyDNaVzTkMkrI58EgUa1ZuYQ88t4UJobDCM'

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

arm_1 = ['CONSTRUCTIVE_EXPERIMENTAL' ,
        'PERSONAL_STORY_EXPERIMENTAL',
        'AFFINITY_EXPERIMENTAL',
        'COMPASSION_EXPERIMENTAL',
        'RESPECT_EXPERIMENTAL',
        'CURIOSITY_EXPERIMENTAL']

arm_2 = ['CONSTRUCTIVE_EXPERIMENTAL' ,
        'PERSONAL_STORY_EXPERIMENTAL',
        'AFFINITY_EXPERIMENTAL',
        'COMPASSION_EXPERIMENTAL',
        'RESPECT_EXPERIMENTAL',
        'CURIOSITY_EXPERIMENTAL',
        'FEARMONGERING_EXPERIMENTAL',
        'GENERALIZATION_EXPERIMENTAL',
        'SCAPEGOATING_EXPERIMENTAL',
        'MORAL_OUTRAGE_EXPERIMENTAL',
        'ALIENATION_EXPERIMENTAL']

arm_3 = ['CONSTRUCTIVE_EXPERIMENTAL' ,
        'PERSONAL_STORY_EXPERIMENTAL',
        'AFFINITY_EXPERIMENTAL',
        'COMPASSION_EXPERIMENTAL',
        'RESPECT_EXPERIMENTAL',
        'CURIOSITY_EXPERIMENTAL',
        'TOXICITY',
        'IDENTITY_ATTACK',
        'INSULT',
        'THREAT']

arms = [arm_1, arm_2, arm_3]


### Rewritten

class PerspectiveRanker:
    def __init__(self, data):
        self.api_key = API_KEY
        self.data = data  
        
    # Selects arm based on cohort index
    def arm_selection(self, data):
        cohort = data.session.cohort_index
        if cohort >0 and cohort < 1000: # These are arbitrary values until I know the cohort index split
            return arm_1
        elif cohort > 1000 and cohort < 2000:
            return arm_2
        else:
            return arm_3
    
    async def score(self, attributes, statement, statement_id):
        # Necessary to use asyncio.to_thread to avoid blocking the event loop. googleapiclient is synchronous.
        return await asyncio.to_thread(self.sync_score, attributes, statement, statement_id)

    def sync_score(self, attributes, statement, statement_id):
        client = discovery.build(
            'commentanalyzer', 
            'v1alpha1', 
            developerKey=self.api_key,
            discoveryServiceUrl='https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1',
            static_discovery=False)  
        
        analyze_request = {
            'comment': {'text': statement},
            'requestedAttributes': {attr: {} for attr in attributes}
        }
        
        response = client.comments().analyze(body=analyze_request).execute()
        results = [(attr, response['attributeScores'][attr]['summaryScore']['value']) for attr in attributes]
        return statement, results, statement_id
            
    async def ranker(self):
        arm = self.arm_selection(self.data)
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(self.score(arm, item.text, item.id)) for item in self.data.items]

        results = await asyncio.gather(*tasks)
        return self.arm_sort(results)
        
    def arm_sort(self, results):
        weightings = {'CONSTRUCTIVE_EXPERIMENTAL': 1/6,
                'PERSONAL_STORY_EXPERIMENTAL': 1/6,
                'AFFINITY_EXPERIMENTAL': 1/6,
                'COMPASSION_EXPERIMENTAL' : 1/6,
                'RESPECT_EXPERIMENTAL': 1/6,
                'CURIOSITY_EXPERIMENTAL': 1/6,
                'FEARMONGERING_EXPERIMENTAL' : -1/3,
                'GENERALIZATION_EXPERIMENTAL' : -1/3,
                'SCAPEGOATING_EXPERIMENTAL' : -1/9,
                'MORAL_OUTRAGE_EXPERIMENTAL' : -1/9,
                'ALIENATION_EXPERIMENTAL' : -1/9,
                'TOXICITY' : -1/4,
                'IDENTITY_ATTACK' : -1/4,
                'INSULT': -1/4,
                'THREAT': -1/4}    
        
        reordered_statements = []
        
        for result in results:
            combined_score = 0
            for group in result[1]: 
                attribute, score = group
                combined_score += weightings[attribute] * score
            reordered_statements.append((result[2], combined_score)) 
        
        
        reordered_statements.sort(key=lambda x: x[1], reverse=True)
        filtered = [x[0] for x in reordered_statements]
        
        result = {
        "ranked_ids": filtered,
    }
        return result

@app.post("/jigsaw")
async def main(ranking_request: RankingRequest) -> RankingResponse:
    ranker = PerspectiveRanker(ranking_request)
    results = await ranker.ranker()
    return results


# if __name__ == '__main__':
#     asyncio.run(main())

