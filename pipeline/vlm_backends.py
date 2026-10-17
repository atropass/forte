"""VLM backends behind one interface. Select with env VLM_BACKEND=mlx|vllm and VLM_MODEL=<hf id>.

Both backends receive exactly the same chat messages (text + PIL images) built by pipeline.vlm,
so prompts and parsing are shared. Decoding is greedy (temperature 0) everywhere.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from PIL import Image

DEFAULT_MODELS = {
    "mlx": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
    "vllm": "Qwen/Qwen3-VL-8B-Instruct",
}


@dataclass
class VLMRequest:
    prompt: str
    images: list[Image.Image] = field(default_factory=list)
    max_tokens: int = 64


class VLMBackend(ABC):
    name: str = "base"

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.calls = 0
        self.total_time = 0.0

    @abstractmethod
    def _generate(self, reqs: list[VLMRequest]) -> list[str]: ...

    def generate(self, reqs: list[VLMRequest]) -> list[str]:
        t = time.time()
        out = self._generate(reqs)
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

    def _generate(self, reqs):
        from mlx_vlm import apply_chat_template, generate
        outs = []
        for r in reqs:
            prompt = apply_chat_template(self.processor, self.config, r.prompt, num_images=len(r.images))
            res = generate(self.model, self.processor, prompt, image=r.images or None,
                           max_tokens=r.max_tokens, temperature=0.0, verbose=False)
            outs.append(res.text if hasattr(res, "text") else str(res))
        return outs


class VLLMBackend(VLMBackend):
    """NVIDIA backend (deployment). Offline engine, batched requests."""
    name = "vllm"

    def __init__(self, model_id: str):
        super().__init__(model_id)
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=model_id,
            max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")),
            gpu_memory_utilization=float(os.getenv("VLLM_GPU_UTIL", "0.80")),
            limit_mm_per_prompt={"image": 6},  # handover check sends 6 frames
            mm_processor_kwargs={"max_pixels": int(os.getenv("VLM_MAX_PIXELS", str(640 * 640)))},
            enforce_eager=os.getenv("VLLM_EAGER", "0") == "1",
        )
        self.SamplingParams = SamplingParams

    def _generate(self, reqs):
        msgs = []
        for r in reqs:
            content = [{"type": "image_pil", "image_pil": im} for im in r.images]
            content.append({"type": "text", "text": r.prompt})
            msgs.append([{"role": "user", "content": content}])
        sp = [self.SamplingParams(temperature=0.0, max_tokens=r.max_tokens) for r in reqs]
        outs = self.llm.chat(msgs, sampling_params=sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


def make_backend(name: str | None = None, model_id: str | None = None) -> VLMBackend:
    name = (name or os.getenv("VLM_BACKEND") or "mlx").lower()
    model_id = model_id or os.getenv("VLM_MODEL") or DEFAULT_MODELS[name]
    if name == "mlx":
        return MLXBackend(model_id)
    if name == "vllm":
        return VLLMBackend(model_id)
    raise ValueError(f"unknown VLM backend {name}")
