RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "source_estimate": {
            "type": "string",
            "description": (
                "Best single-label guess for how this image was produced. "
                "One of: 'DALL-E', 'Midjourney', 'Stable Diffusion', "
                "'Other AI', 'Real', or 'Uncertain'."
            ),
        },
        "ai_coverage": {
            "type": "number",
            "description": (
                "Estimated fraction (0.0-1.0) of the image canvas that "
                "appears AI-generated or AI-modified."
            ),
        },
        "confidence": {
            "type": "number",
            "description": "Your confidence (0.0-1.0) in source_estimate.",
        },
        "reasoning": {
            "type": "string",
            "description": "1-3 sentence plain-language explanation citing evidence.",
        },
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short specific observations (max ~8 words each).",
        },
    },
    "required": ["source_estimate", "ai_coverage", "confidence", "reasoning", "evidence"],
}

CONSISTENCY_CHECK_PROMPT = """You are a forensic image analyst helping detect AI-generated \
or AI-edited images. You will be shown one image. Examine it carefully along \
these four dimensions before answering:

1. LIGHTING & SHADOWS: Do shadow directions match a single consistent light \
source? Are reflections physically plausible?
2. ANATOMY & STRUCTURE: Check hands, ears, teeth, fingers, limb proportions, \
and text/patterns for garbling or impossible geometry.
3. BACKGROUND & CONTEXT: Look for melted/repeating background elements or \
objects that don't obey perspective.
4. COMPRESSION & NOISE: Distinguish ordinary JPEG/social-media compression \
from generation artifacts.

Do not assume real just because it looks high quality. Do not assume fake \
just because it's compressed or resized — that's normal reposting, not \
evidence of AI generation.

Respond ONLY with a JSON object matching this schema (no markdown fences, no \
preamble):

{schema}
""".format(schema=RESPONSE_SCHEMA)