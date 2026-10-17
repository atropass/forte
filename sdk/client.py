"""Production Python Client for ForteVision vLLM Service."""
from __future__ import annotations

import base64
import json
import time
import urllib.request
from pathlib import Path


class VLLMClient:
    def __init__(self, base_url: str = "https://tw6jhre43eeunm-8000.proxy.runpod.net/v1",
                 api_key: str = "fv-prod-secret-forte-2026",
                 model: str = "Qwen/Qwen3-VL-4B-Instruct",
                 timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "ForteVision-Client/1.0"
            }
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"HTTP {e.code}: {err_body}") from e
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError("Request failed after retries")

    def chat(self, prompt: str, system_prompt: str | None = None, max_tokens: int = 256) -> str:
        """Simple text chat completion."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        res = self._post("/chat/completions", {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0
        })
        return res["choices"][0]["message"]["content"]

    def analyze_image(self, prompt: str, image_path_or_url: str | Path, max_tokens: int = 256) -> str:
        """Vision prompt analyzing a local file or remote image URL."""
        if str(image_path_or_url).startswith(("http://", "https://")):
            img_url = str(image_path_or_url)
        else:
            p = Path(image_path_or_url)
            mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
            img_url = f"data:{mime};base64,{b64}"

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": img_url}}
            ]
        }]

        res = self._post("/chat/completions", {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0
        })
        return res["choices"][0]["message"]["content"]

    def health(self) -> bool:
        """Returns True if the vLLM engine is responsive."""
        root_url = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        req = urllib.request.Request(
            f"{root_url}/health",
            headers={"User-Agent": "ForteVision-Client/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return resp.status == 200
        except Exception:
            return False


class CVClient:
    """Production Python Client for ForteVision Computer Vision Microservice (RT-DETRv2 & Tracking)."""

    def __init__(
        self,
        base_url: str = "https://ydfg26qe73blrr-8000.proxy.runpod.net",
        api_key: str = "fv-prod-secret-forte-2026",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "ForteVision-CV-Client/1.0",
            },
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"HTTP {e.code}: {err_body}") from e
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError("Request failed after retries")

    def health(self) -> dict:
        """Check status, readiness, device, and supported COCO classes."""
        req = urllib.request.Request(
            f"{self.base_url}/health",
            headers={"User-Agent": "ForteVision-CV-Client/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def detect(
        self,
        image: str | Path,
        threshold: float | None = None,
        classes: list[str] | None = None,
    ) -> dict:
        """Run RT-DETRv2 object detection on a local file, base64 string, or remote image URL.

        Returns:
            dict with 'elapsed_ms' and list of 'detections' (label, confidence, box: [x1, y1, x2, y2]).
        """
        payload: dict = {}
        if threshold is not None:
            payload["threshold"] = threshold
        if classes:
            payload["classes"] = classes

        str_img = str(image)
        if str_img.startswith(("http://", "https://")):
            payload["url"] = str_img
        elif Path(str_img).exists():
            raw_bytes = Path(str_img).read_bytes()
            payload["image_base64"] = base64.b64encode(raw_bytes).decode("utf-8")
        else:
            payload["image_base64"] = str_img

        return self._post("/detect", payload)

    def track(
        self,
        frames: list[dict],
        iou_threshold: float = 0.25,
        max_misses: int = 4,
    ) -> dict:
        """Run ByteTrack-style IoU multi-object tracking across video frames.

        Args:
            frames: list of dicts with 'frame_index' (int) and 'person_boxes' (list of [x1, y1, x2, y2]).
            iou_threshold: minimum IoU overlap to associate tracks across frames.
            max_misses: maximum consecutive missing frames before retiring track.
        """
        payload = {
            "frames": frames,
            "iou_threshold": iou_threshold,
            "max_misses": max_misses,
        }
        return self._post("/track", payload)
