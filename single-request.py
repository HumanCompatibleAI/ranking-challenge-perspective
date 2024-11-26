import asyncio
from perspective_ranker import PerspectiveRanker,perspective_baseline

# a simple script for running a single request against the real API, so it's possible to
# easily examine the result

async def main():
    ranker = PerspectiveRanker()

    statement = 'foo bar 😅 ' * 5000

    result = await ranker.score(perspective_baseline, statement, 'some-id')

    breakpoint()
    return result


asyncio.run(main())
