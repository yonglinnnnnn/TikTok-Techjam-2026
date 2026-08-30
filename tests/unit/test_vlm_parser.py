from truesight.vlm.parser import parse_vlm_response


def test_parses_well_formed_response():
    raw = {"source_estimate": "DALL-E", "ai_coverage": 0.9, "confidence": 0.85,
           "reasoning": "Hands show extra fingers.", "evidence": ["six fingers"]}
    result = parse_vlm_response(raw, provider="openai")
    assert result.is_valid()
    assert result.source_estimate == "DALL-E"


def test_handles_missing_fields_gracefully():
    result = parse_vlm_response({"source_estimate": "Real"}, provider="gemini")
    assert result.is_valid()
    assert result.ai_coverage is None


def test_rejects_invalid_source_label():
    raw = {"source_estimate": "trust me bro", "ai_coverage": 0.5,
           "confidence": 0.5, "reasoning": "n/a", "evidence": []}
    result = parse_vlm_response(raw, provider="openai")
    assert result.source_estimate == "Uncertain"


def test_handles_non_dict_payload():
    result = parse_vlm_response("not a dict", provider="openai")
    assert result.source_estimate == "Uncertain"
    assert result.confidence == 0.0


def test_clamps_out_of_range_scores():
    raw = {"source_estimate": "Real", "ai_coverage": 1.7, "confidence": -0.3,
           "reasoning": "n/a", "evidence": []}
    result = parse_vlm_response(raw, provider="openai")
    assert result.ai_coverage == 1.0
    assert result.confidence == 0.0
