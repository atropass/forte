"""OpenAI-compatible remote vLLM backend against a fake server (httpx.MockTransport) — no network, no model."""
import json
import math
import sys
from pathlib import Path

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.vlm_backends import OpenAICompatBackend, VLMRequest, score_from_top_logprobs  # noqa: E402


def fake_backend(handler):
    b = OpenAICompatBackend("Qwen/Qwen3-VL-4B-Instruct", base_url="http://fake/v1", api_key="k")
    b.client = httpx.Client(base_url="http://fake/v1", transport=httpx.MockTransport(handler),
                            headers={"Authorization": "Bearer k"})
    return b


def lp(pairs):
    return {"choices": [{"message": {"content": pairs[0][0]},
                         "logprobs": {"content": [{"token": pairs[0][0], "logprob": math.log(pairs[0][1]),
                                                   "top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in pairs]}]}}]}


def test_score_sums_spellings_and_normalizes():
    r = score_from_top_logprobs(lp([("Yes", 0.6), ("yes", 0.2), ("No", 0.1), ("The", 0.05)]), ["yes", "no"])
    assert abs(r["yes"] - 0.8 / 0.9) < 1e-9


def test_score_letters():
    r = score_from_top_logprobs(lp([("C", 0.7), ("F", 0.2), ("G", 0.1)]), list("ABCDEFG"))
    assert max(r, key=r.get) == "C" and abs(sum(r.values()) - 1) < 1e-9


def test_request_shape_retry_and_parsing():
    seen = {"n": 0}

    def handler(req: httpx.Request):
        seen["n"] += 1
        assert req.headers["authorization"] == "Bearer k"
        body = json.loads(req.content)
        if seen["n"] == 1:
            return httpx.Response(503, text="busy")  # first call fails -> retried
        content = body["messages"][0]["content"]
        assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        if body["max_tokens"] == 1:
            assert body["logprobs"] is True and body["top_logprobs"] == 20
            return httpx.Response(200, json=lp([("no", 0.7), ("yes", 0.3)]))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"inside": 1}'}}]})

    b = fake_backend(handler)
    img = Image.new("RGB", (64, 64), (128, 128, 128))
    out = b.score([VLMRequest("q", [img], 1)], [["yes", "no"]])
    assert abs(out[0]["yes"] - 0.3) < 1e-9 and seen["n"] == 2
    assert b.generate([VLMRequest("q", [img], 60), VLMRequest("q", [img], 60)]) == ['{"inside": 1}'] * 2
