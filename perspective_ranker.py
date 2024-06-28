import asyncio
import numpy as np
# import aiohttp
import asyncio
from googleapiclient import discovery
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


API_KEY = 'AIzaSyDNaVzTkMkrI58EgUa1ZuYQ88t4UJobDCM'

# cohort_list = [np.random.randint(0, 4000) for _ in range(3)]

statements = [
    "I love everything and everyone!",
    "I am threatening but not very profane. I do not like swearing",
    "Im very toxic but also profane. Toxic Toxic Toxic!",
    ]

attributes = ['TOXICITY',
    'PROFANITY',
    'THREAT']


def score(attribute, statement):
    
    store = []
    client = discovery.build('commentanalyzer', 
                             'v1alpha1', 
                             developerKey=API_KEY,
                             discoveryServiceUrl='https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1',
                             static_discovery=False)   

    
    analyze_request = {
        'comment': {'text': statement},
        'requestedAttributes': {attribute: {}},
    }
    response = client.comments().analyze(body=analyze_request).execute()
    result = response['attributeScores'][attribute]['summaryScore']['value']
    return attribute, statement, result

async def ranker(attribute, statement):
    loop = asyncio.get_running_loop()
    task = loop.run_in_executor(None, score, attribute, statement)
    results = await asyncio.gather(task)
    return results


async def process_request(attributes, statements):
    tasks = []
    for attribute in attributes:
        for statement in statements:
            # print(attribute, statement)
            tasks.append(ranker(attribute, statement))


    results = await asyncio.gather(*tasks)
    return results

def combination(results):
    attr1= 0.2
    attr2 = 0.2
    attr3 = 0.1
    
    
    combination = {}

    for i in results:
        # print(i)
        for attribute, statement, score in i:
            if statement not in combination:
                combination[statement] = []
            combination[statement].append(score)
        
    scoring = []
    
    for statement, scores in combination.items():
        combined_score = (attr1 * scores[0]) + (attr2 * scores[1]) + (attr3 * scores[2])
        scoring.append((statement, combined_score))
        
    scoring.sort(key=lambda x: x[1], reverse=True)
    return scoring


if __name__ == '__main__':
    attributes = attributes
    statements = statements
    scores = asyncio.run(process_request(attributes, statements))
    print(combination(scores))
    
