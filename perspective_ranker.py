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
    "I am a neutral statement.",
    "I've killed before and I'll kill again. Toxic Toxic Toxic!",
    ]

def score(attribute, statement):
    
    store = []
    client = discovery.build('commentanalyzer', 
                             'v1alpha1', 
                             developerKey=API_KEY,
                             discoveryServiceUrl='https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1',
                             static_discovery=False)   

    
    analyze_request = {
        'comment': {'text': statement},
        'requestedAttributes': {    attribute : {} },
    }

    response = client.comments().analyze(body=analyze_request).execute()

    result = response['attributeScores'][attribute]['summaryScore']['value']
    return statement, result

async def ranker(attribute, statement, iterations=3):
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(None, score, attribute, statement) for i in range(iterations)
    ]
    results = await asyncio.gather(*tasks)
    return statement, results


async def process_request(attribute, statements):
    tasks = [ranker(attribute, statement) for statement in statements]
    results = await asyncio.gather(*tasks)
    return results

def combination(results):
    formula_1 = 0.5
    formula_2 = 0.4
    formula_3 = 0.1
    
    combination = {}

    for statement, scores in results:
        temp = []
        for _, i in scores:
            temp.append(i)
        score = formula_1 * temp[0] + formula_2 * temp[1]+ formula_3 * temp[2]
        combination.update({statement:score})
        
    return sorted(combination, key=combination.get, reverse=True)


if __name__ == '__main__':
    attribute = 'TOXICITY'
    statements = statements

    scores = asyncio.run(process_request(attribute, statements))
    # print(scores)
    print(combination(scores))
    
