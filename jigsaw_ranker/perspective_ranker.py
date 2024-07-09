import asyncio
import numpy as np
from typing import List
import asyncio
from googleapiclient import discovery
import json
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import sys
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

arm_1 = ['CONSTRUCTIVE_EXPERIMENTAL',
    'TOXICITY',
    'INSULT',
    'THREAT']

arm_2 = [
    'TOXICITY',
    'THREAT',]

arm_3 = ['CONSTRUCTIVE_EXPERIMENTAL',
    'TOXICITY']

arms = [arm_1, arm_2, arm_3]

def score(attributes, statement):
    
    store = []
    client = discovery.build('commentanalyzer', 
                             'v1alpha1', 
                             developerKey=API_KEY,
                             discoveryServiceUrl='https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1',
                             static_discovery=False)   

    
    analyze_request = {
        'comment': {'text': statement},
        'requestedAttributes': {attr: {} for attr in attributes}}
    
    response = client.comments().analyze(body=analyze_request).execute()
    
    results = [(attr, response['attributeScores'][attr]['summaryScore']['value']) for attr in attributes]
    return statement, results

async def ranker(attributes, statement):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, score, attributes, statement)
    return result 

async def process_request(attributes, statements):
    tasks = []
    for statement in statements:
        tasks.append(ranker(attributes, statement))

    results = await asyncio.gather(*tasks)
    
    return results

async def process_all_arms(arms, statements):
    tasks = [process_request(arm, statements) for arm in arms]
    results = await asyncio.gather(*tasks)
    sorted_results = [arm_sort(result) for result in results]
    return sorted_results[0], sorted_results[1], sorted_results[2]


def arm_sort(results):
    weightings = {'CONSTRUCTIVE_EXPERIMENTAL': 1/6,
              'PERSONAL_STORY': 1/6,
              'AFFINITY': 1/6,
              'COMPASSION' : 1/6,
              'RESPECT': 1/6,
              'CURIOSITY': 1/6,
              'FEARMONGERING' : -1/3,
              'GENERALIZATION' : -1/3,
              'SCAPEGOATING' : -1/9,
              'MORAL_OUTRAGE' : -1/9,
              'ALIENATION' : -1/9,
              'TOXICITY' : -1/4,
              'IDENTITY_ATTACK' : -1/4,
              'INSULT': -1/4,
              'THREAT': -1/4}    
    
    reordered_statements = []
    
    for statement, scoring in results:
        combined_score = 0
        for attribute, score in scoring: 
            combined_score += weightings[attribute] * score
        reordered_statements.append((statement, combined_score)) 
        
    reordered_statements.sort(key=lambda x: x[1], reverse=True)
    
    return reordered_statements

@app.post("/jigsaw")
async def jigsaw_endpoint(statements: List[str] = Body(...)):
    results = await process_all_arms(arms, statements)
    return {"results": results}


# if __name__ == '__main__':
#     scores = asyncio.run(process_all_arms(arms, comments))
#     print(scores)

