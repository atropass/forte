"""VLM backends behind one interface. Select with env VLM_BACKEND=mlx|vllm|openai and VLM_MODEL=<hf id>.
  mlx     Apple silicon, in-process (development)
  vllm    NVIDIA, in-process vLLM engine
  openai  our own vLLM server over its OpenAI-compatible HTTP API (VLM_API_BASE, VLM_API_KEY)

Both backends receive exactly the same chat messages (text + PIL images) built by pipeline.vlm,
so prompts and parsing are shared. Decoding is greedy (temperature 0) everywhere.

Two call types:
  generate(reqs)          free text (JSON answers: scene state, grounding)
  score(reqs, options)    decision mode: ONE forward pass, returns the probability of each option from the
                          first answer token (e.g. {"yes": .93, "no": .07}). No text is generated, so it is
                          faster and gives a calibrated confidence instead of a hard "yes/no/unclear".
"""
from __future__ import annotations

import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from PIL import Image

DEFAULT_MODELS = {
    "mlx": "mlx-community/Qwen3-VL-4B-Instruct-4bit",
    "vllm": "Qwen/Qwen3-VL-4B-Instruct",
    "openai": "Qwen/Qwen3-VL-4B-Instruct",
}


@dataclass
class VLMRequest:
    prompt: str
    images: list[Image.Image] = field(default_factory=list)
    max_tokens: int = 64


def option_token_ids(tokenizer, options: list[str]) -> dict[int, str]:
    """First-token ids for each option and its common spellings ("yes", "Yes", " yes", "YES")."""
    ids: dict[int, str] = {}
    for opt in options:
        for variant in {opt, opt.lower(), opt.upper(), opt.capitalize(), " " + opt, " " + opt.capitalize()}:
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if toks:
                ids.setdefault(int(toks[0]), opt)
    return ids


def normalize(logprobs: dict[int, float], id2opt: dict[int, str], options: list[str]) -> dict[str, float]:
    """Sum probability mass per option over its token variants and renormalize over the options."""
    mass = {o: 0.0 for o in options}
    for tid, lp in logprobs.items():
        if tid in id2opt and lp is not None and lp > -1e9:
            mass[id2opt[tid]] += math.exp(lp)
    total = sum(mass.values())
    if total <= 0:
        return {o: 1.0 / len(options) for o in options}
    return {o: m / total for o, m in mass.items()}


class VLMBackend(ABC):
    name: str = "base"

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.calls = 0
        self.total_time = 0.0

    @abstractmethod
    def _generate(self, reqs: list[VLMRequest]) -> list[str]: ...

    @abstractmethod
    def _score(self, reqs: list[VLMRequest], options: list[list[str]]) -> list[dict[str, float]]: ...

    def generate(self, reqs: list[VLMRequest]) -> list[str]:
        t = time.time()
        out = self._generate(reqs)
        self.total_time += time.time() - t
        self.calls += len(reqs)
        return out

    def score(self, reqs: list[VLMRequest], options: list[list[str]]) -> list[dict[str, float]]:
        t = time.time()
        out = self._score(reqs, options)
        self.total_time += time.time() - t
        self.calls += len(reqs)
        return out


class MLXBackend(VLMBackend):
    """Apple-silicon backend (development). Sequential; mlx-vlm has no batched multi-image API."""
    name = "mlx"

    def __init__(self, model_id: str):
        super().__init__(model_id)
        from mlx_vlm import load
        self.model, self.processor = load(model_id)
        self.config = self.model.config
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

    def _generate(self, reqs):
        from mlx_vlm import apply_chat_template, generate
        outs = []
        for r in reqs:
            prompt = apply_chat_template(self.processor, self.config, r.prompt, num_images=len(r.images))
            res = generate(self.model, self.processor, prompt, image=r.images or None,
                           max_tokens=r.max_tokens, temperature=0.0, verbose=False)
            outs.append(res.text if hasattr(res, "text") else str(res))
        return outs

    def _score(self, reqs, options):
        import numpy as np
        from mlx_vlm import apply_chat_template, stream_generate
        outs = []
        for r, opts in zip(reqs, options):
            id2opt = option_token_ids(self.tokenizer, opts)
            prompt = apply_chat_template(self.processor, self.config, r.prompt, num_images=len(r.images))
            first = next(iter(stream_generate(self.model, self.processor, prompt, image=r.images or None,
                                              max_tokens=1, temperature=0.0)))
            lp = np.array(first.logprobs, dtype=np.float32).reshape(-1)
            outs.append(normalize({tid: float(lp[tid]) for tid in id2opt if tid < lp.shape[0]}, id2opt, opts))
        return outs


class VLLMBackend(VLMBackend):
    """NVIDIA backend (deployment). Offline engine, batched requests, prefix caching for the shared preamble."""
    name = "vllm"

    def __init__(self, model_id: str, gpu_util: float | None = None):
        super().__init__(model_id)
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=model_id,
            max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")),
            gpu_memory_utilization=gpu_util if gpu_util is not None else float(os.getenv("VLLM_GPU_UTIL", "0.80")),
            limit_mm_per_prompt={"image": 6},
            mm_processor_kwargs={"max_pixels": int(os.getenv("VLM_MAX_PIXELS", str(640 * 640)))},
            enable_prefix_caching=True,
            enforce_eager=os.getenv("VLLM_EAGER", "0") == "1",
        )
        self.SamplingParams = SamplingParams
        self.tokenizer = self.llm.get_tokenizer()

    @staticmethod
    def _messages(reqs):
        msgs = []
        for r in reqs:
            content = [{"type": "image_pil", "image_pil": im} for im in r.images]
            content.append({"type": "text", "text": r.prompt})
            msgs.append([{"role": "user", "content": content}])
        return msgs

    def _generate(self, reqs):
        sp = [self.SamplingParams(temperature=0.0, max_tokens=r.max_tokens) for r in reqs]
        outs = self.llm.chat(self._messages(reqs), sampling_params=sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def _score(self, reqs, options):
        maps = [option_token_ids(self.tokenizer, opts) for opts in options]
        # allowed_token_ids forces the answer token to be one of the options; logprobs may be reported before
        # that restriction (vLLM "raw" logprobs), so ask for the top 20 (vLLM's default max) and sum the option
        # spellings found there. With "Answer with exactly one word/letter" the options dominate the top 20.
        sp = [self.SamplingParams(temperature=0.0, max_tokens=1, logprobs=max(20, len(m)), allowed_token_ids=list(m))
              for m in maps]
        outs = self.llm.chat(self._messages(reqs), sampling_params=sp, use_tqdm=False)
        res = []
        for o, m, opts in zip(outs, maps, options):
            step = (o.outputs[0].logprobs or [{}])[0]
            res.append(normalize({int(t): float(v.logprob) for t, v in step.items()}, m, opts))
        return res


class OpenAICompatBackend(VLMBackend):
    """Our own vLLM server reached over its OpenAI-compatible HTTP API (`vllm serve`, e.g. the team's GPU pod).

    The model still runs on our own GPU — this is not a third-party AI service — only the transport differs.
    Env: VLM_API_BASE (…/v1), VLM_API_KEY (gateway bearer token), VLM_CONCURRENCY (parallel requests; the
    server batches them), VLM_TIMEOUT, VLM_JPEG_QUALITY.
    Decision mode: max_tokens=1 + top_logprobs=20, option probabilities summed over spellings.
    """
    name = "openai"

    def __init__(self, model_id: str, base_url: str | None = None, api_key: str | None = None):
        super().__init__(model_id)
        import httpx
        self.base = (base_url or os.getenv("VLM_API_BASE", "")).rstrip("/")
        if not self.base:
            raise RuntimeError("VLM_API_BASE is not set (e.g. https://<pod>-8000.proxy.runpod.net/v1)")
        key = api_key if api_key is not None else os.getenv("VLM_API_KEY", "")
        self.concurrency = int(os.getenv("VLM_CONCURRENCY", "16"))
        self.quality = int(os.getenv("VLM_JPEG_QUALITY", "90"))
        self.client = httpx.Client(
            base_url=self.base, timeout=float(os.getenv("VLM_TIMEOUT", "120")),
            headers={"Authorization": f"Bearer {key}"} if key else {},
            limits=httpx.Limits(max_connections=self.concurrency, max_keepalive_connections=self.concurrency))

    def served_models(self) -> list[str]:
        """Model ids and checkpoint paths the server actually serves (vLLM: id = served name, root = weights)."""
        data = self.client.get("/models").json().get("data", [])
        return sorted({x for d in data for x in (d.get("id"), d.get("root")) if x})

    def _content(self, r: VLMRequest) -> list[dict]:
        import base64
        import io
        parts = []
        for im in r.images:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=self.quality)
            parts.append({"type": "image_url",
                          "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}})
        parts.append({"type": "text", "text": r.prompt})
        return parts

    def _post(self, body: dict) -> dict:
        delay = 1.0
        for attempt in range(6):
            try:
                resp = self.client.post("/chat/completions", json=body)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                return resp.json()
            except Exception:  # noqa: BLE001 - transient network / gateway errors are retried
                if attempt == 5:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 15)
        raise RuntimeError("unreachable")

    def _map(self, fn, items):
        from concurrent.futures import ThreadPoolExecutor
        if len(items) <= 1:
            return [fn(x) for x in items]
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(fn, items))

    def _generate(self, reqs):
        def one(r):
            j = self._post({"model": self.model_id, "temperature": 0.0, "max_tokens": r.max_tokens,
                            "messages": [{"role": "user", "content": self._content(r)}]})
            return j["choices"][0]["message"]["content"] or ""
        return self._map(one, reqs)

    def _score(self, reqs, options):
        def one(args):
            r, opts = args
            j = self._post({"model": self.model_id, "temperature": 0.0, "max_tokens": 1, "logprobs": True,
                            "top_logprobs": 20, "messages": [{"role": "user", "content": self._content(r)}]})
            return score_from_top_logprobs(j, opts)
        return self._map(one, list(zip(reqs, options)))


def score_from_top_logprobs(resp: dict, options: list[str]) -> dict[str, float]:
    """Option probabilities from an OpenAI-style chat response with logprobs (first generated token)."""
    ch = resp["choices"][0]
    content = ((ch.get("logprobs") or {}).get("content") or [])
    tops = content[0].get("top_logprobs", []) if content else []
    lookup = {o.lower(): o for o in options}
    mass = {o: 0.0 for o in options}
    for t in tops:
        key = str(t.get("token", "")).strip().lower()
        if key in lookup and t.get("logprob") is not None:
            mass[lookup[key]] += math.exp(float(t["logprob"]))
    total = sum(mass.values())
    if total <= 0:  # no option in the top 20: fall back to the generated text
        txt = (ch.get("message", {}).get("content") or "").strip().lower()
        hit = next((o for o in options if txt.startswith(o.lower())), None)
        return {o: (1.0 if o == hit else 0.0) for o in options} if hit else {o: 1.0 / len(options) for o in options}
    return {o: m / total for o, m in mass.items()}


def make_backend(name: str | None = None, model_id: str | None = None, gpu_util: float | None = None) -> VLMBackend:
    name = (name or os.getenv("VLM_BACKEND") or "mlx").lower()
    model_id = model_id or os.getenv("VLM_MODEL") or DEFAULT_MODELS[name]
    if name == "mlx":
        return MLXBackend(model_id)
    if name == "vllm":
        return VLLMBackend(model_id, gpu_util)
    if name in ("openai", "remote"):
        return OpenAICompatBackend(model_id)
    raise ValueError(f"unknown VLM backend {name}")
