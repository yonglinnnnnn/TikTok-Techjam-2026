import asyncio
from dotenv import load_dotenv
load_dotenv()

from truesight.vlm import GeminiProvider, run_vlm_tier

async def main():
    image_path = "tests/fixtures/sample_images/full_synthetic_full_synthetic_004934.jpg"

    result = await run_vlm_tier(image_path, [GeminiProvider()])

    print("=== Merged tier result ===")
    print("source:       ", result.source)
    print("ai_coverage:  ", result.ai_coverage)
    print("confidence:   ", result.confidence)
    print("reasoning:    ", result.reasoning)
    print("evidence:     ", result.evidence)

    print("\n=== Raw per-provider result ===")
    for r in result.per_provider:
        print("provider:", r.provider)
        print("error:   ", r.error)
        print("latency: ", r.latency_ms, "ms")
        print("raw:     ", r.raw_response)

asyncio.run(main())