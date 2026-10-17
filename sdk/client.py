"""Python clients (stdlib only).

ForteVisionClient  the public API of the service (what the jury / a bank system integrates with)
VLLMClient         the team's vLLM server (OpenAI-compatible), server-side use only
CVClient           the RT-DETRv2 CV service on the GPU pod, server-side use only

No URLs or keys are built in: pass them explicitly or set the environment variables
FV_API_BASE / FV_API_KEY, VLM_API_BASE / VLM_API_KEY, DET_API_BASE / DET_API_KEY.

    from sdk.client import ForteVisionClient
    fv = ForteVisionClient("https://forte.example.kz", api_key="fv_...")
    result = fv.analyze_and_wait("video_017.mov")      # {"job_id", "status": "done", "verdict", "violations"}
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def _required(value: str | None, env: str) -> str:
    v = value or os.getenv(env, "")
    if not v:
        raise ValueError(f"{env} is not set and no value was passed")
    return v


class _Http:
    def __init__(self, base_url: str, headers: dict, timeout: float, user_agent: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"User-Agent": user_agent, **headers}
        self.timeout = timeout

    def request(self, method: str, path: str, body: bytes | None = None, content_type: str | None = None,
                retries: int = 3) -> dict:
        headers = dict(self.headers)
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(f"{self.base_url}{path}", data=body, method=method, headers=headers)
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code in (429, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')[:500]}") from e
            except (urllib.error.URLError, TimeoutError):
                if attempt == retries - 1:
                    raise
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError("request failed after retries")


class ForteVisionClient:
    """Public API: POST /api/v1/analyze (multipart `file`) -> job_id; GET /api/v1/jobs/{job_id}."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 120.0):
        self.http = _Http(_required(base_url, "FV_API_BASE"), {"X-API-Key": _required(api_key, "FV_API_KEY")},
                          timeout, "ForteVision-SDK/2.0")

    def analyze(self, video: str | Path) -> str:
        p = Path(video)
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{p.name}\"\r\n"
                "Content-Type: video/quicktime\r\n\r\n").encode() + p.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        return self.http.request("POST", "/api/v1/analyze", body, f"multipart/form-data; boundary={boundary}",
                                 retries=1)["job_id"]

    def get_job(self, job_id: str) -> dict:
        return self.http.request("GET", f"/api/v1/jobs/{job_id}")

    def wait(self, job_id: str, timeout_s: float = 300, poll_s: float = 3) -> dict:
        deadline = time.time() + timeout_s
        while True:
            res = self.get_job(job_id)
            if res.get("status") != "processing":
                return res
            if time.time() > deadline:
                raise TimeoutError(f"job {job_id} still processing after {timeout_s} s")
            time.sleep(poll_s)

    def analyze_and_wait(self, video: str | Path, timeout_s: float = 300) -> dict:
        return self.wait(self.analyze(video), timeout_s)


class VLLMClient:
    """OpenAI-compatible vLLM server. Server-side only: never ship its token to a browser."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str = "Qwen/Qwen3-VL-4B-Instruct", timeout: float = 60.0):
        self.http = _Http(_required(base_url, "VLM_API_BASE"),
                          {"Authorization": f"Bearer {_required(api_key, 'VLM_API_KEY')}"}, timeout,
                          "ForteVision-Client/2.0")
        self.model = model

    def _chat(self, messages: list, max_tokens: int) -> str:
        res = self.http.request("POST", "/chat/completions", json.dumps({
            "model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}).encode(),
            "application/json")
        choices = res.get("choices") or []
        return (choices[0].get("message") or {}).get("content", "") if choices else ""

    def chat(self, prompt: str, system_prompt: str | None = None, max_tokens: int = 256) -> str:
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        return self._chat(messages + [{"role": "user", "content": prompt}], max_tokens)

    def analyze_image(self, prompt: str, image_path: str | Path, max_tokens: int = 256) -> str:
        p = Path(image_path)
        mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        url = f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
        return self._chat([{"role": "user", "content": [{"type": "text", "text": prompt},
                                                        {"type": "image_url", "image_url": {"url": url}}]}], max_tokens)

    def health(self) -> bool:
        root = self.http.base_url[:-3] if self.http.base_url.endswith("/v1") else self.http.base_url
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{root}/health", headers=self.http.headers), timeout=5):
                return True
        except Exception:  # noqa: BLE001
            return False


class CVClient:
    """RT-DETRv2 CV service. Server-side only. Images are sent as base64 (the service never fetches URLs)."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 30.0):
        self.http = _Http(_required(base_url, "DET_API_BASE"),
                          {"Authorization": f"Bearer {_required(api_key, 'DET_API_KEY')}"}, timeout,
                          "ForteVision-CV-Client/2.0")

    def health(self) -> bool:
        try:
            return bool(self.http.request("GET", "/health", retries=1).get("ready"))
        except Exception:  # noqa: BLE001
            return False

    def detect(self, image: str | Path, threshold: float | None = None, classes: list[str] | None = None) -> dict:
        p = Path(image)
        if not p.is_file():
            raise FileNotFoundError(f"image not found: {image}")
        payload: dict = {"image_base64": base64.b64encode(p.read_bytes()).decode()}
        if threshold is not None:
            payload["threshold"] = threshold
        if classes:
            payload["classes"] = classes
        return self.http.request("POST", "/detect", json.dumps(payload).encode(), "application/json")

    def track(self, frames: list[dict], iou_threshold: float = 0.25, max_misses: int = 4) -> dict:
        payload = {"frames": frames, "iou_threshold": iou_threshold, "max_misses": max_misses}
        return self.http.request("POST", "/track", json.dumps(payload).encode(), "application/json")
