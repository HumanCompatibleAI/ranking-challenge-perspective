import time
from pprint import pprint

from fastapi.encoders import jsonable_encoder

import requests
from ranking_challenge.fake import fake_request

# This is a simple script that sends a POST request to the API and prints the response.
# Your server should be running on localhost:8000

# Wait for the app to start up
time.sleep(2)

# make some fake request data
items = fake_request(n_posts=1, n_comments=2)
items.session.cohort = "arm1"

# Send POST request to the API
response = requests.post("http://localhost:8000/rank", json=jsonable_encoder(items))

# Check if the request was successful (status code 200)
if response.status_code == 200:
    try:
        # Attempt to parse the JSON response
        json_response = response.json()
        pprint(json_response)
    except requests.exceptions.JSONDecodeError:
        print("Failed to parse JSON response. Response may be empty.")
else:
    print(f"Request failed with status code: {response.status_code}")
    print(response.text)
