"""UI preview without models: runs the real FastAPI app, but the analyzer replays a saved full result
(with evidence frames) after a short fake delay. For frontend QA while the GPU is busy.

API_KEY=dev-key python scripts/ui_preview_server.py path/to/full_result.json
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("API_KEY", "dev-key")
os.environ.setdefault("DATA_DIR", str(ROOT / "data_service_preview"))
import service.app as app_mod  # noqa: E402

RESULT = json.loads(Path(sys.argv[1]).read_text())


class Replay:
    def analyze(self, path, job_id="", progress=None):
        for stage, p in [("perception", .1), ("grounding", .2), ("scene_state", .4), ("checks", .7), ("evidence", .9)]:
            progress(stage, p)
            time.sleep(0.8)
        return json.loads(json.dumps(RESULT))


def fake_load():
    app_mod.state.update(analyzer=Replay(), models={"vlm": "mlx-community/Qwen3-VL-4B-Instruct-4bit", "vlm_backend": "replay",
                                                    "detector": "PekingU/rtdetr_v2_r18vd", "device": "-"})
    app_mod.worker.ready.set()


app_mod.load_models = fake_load
import uvicorn  # noqa: E402

uvicorn.run(app_mod.app, host="127.0.0.1", port=int(os.getenv("PORT", "8010")))
