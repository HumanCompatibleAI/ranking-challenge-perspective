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

## API Key

You'll need a Perspective API key, which should be in the `PERSPECTIVE_API_KEY` environment variable. You can put it in `.env` for local use.
