"""VLM backends behind one interface. Select with env VLM_BACKEND=mlx|vllm and VLM_MODEL=<hf id>.

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


def make_backend(name: str | None = None, model_id: str | None = None, gpu_util: float | None = None) -> VLMBackend:
    name = (name or os.getenv("VLM_BACKEND") or "mlx").lower()
    model_id = model_id or os.getenv("VLM_MODEL") or DEFAULT_MODELS[name]
    if name == "mlx":
        return MLXBackend(model_id)
    if name == "vllm":
        return VLLMBackend(model_id, gpu_util)
    raise ValueError(f"unknown VLM backend {name}")
