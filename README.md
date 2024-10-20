# Prosocial Ranking Challenge Perspective Ranker

To install deps:
```bash
poetry install
```

To start in dev:    
```bash
poetry run uvicorn perspective_ranker:app --reload
```

To test:
```bash
poetry run pytest
```

To build and deploy:

- commit changes to rc-release branch
- then run
```
docker buildx bake --set '*.platform=linux/amd64' -f docker-compose.yml --no-cache --load
cd ../ranking-challenge-submission
poetry run python push_to_ecr.py --team perspective
```
- manual "update service" on prod-prc-perspective-app-service in AWS
- PERSPECTIVE_API_KEY and PERSPECTIVE_HOST need to be set in ECS task definition

## API Key

You'll need a Perspective API key, which should be in the `PERSPECTIVE_API_KEY` environment variable. You can put it in `.env` for local use.
