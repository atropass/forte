"""UI preview without models or Redis: runs the real FastAPI app on an in-memory Redis (fakeredis), and an inline
worker replays a saved full result (with evidence frames) after a short fake delay. For frontend QA.

    python scripts/ui_preview_server.py path/to/full_result.json
    -> http://127.0.0.1:8010, login preview / preview-password, API key dev-key
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("API_KEY", "dev-key")
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("COOKIE_SECURE", "0")
os.environ.setdefault("VALIDATE_VIDEO", "0")
os.environ.setdefault("WORKER_MODE", "simple")
os.environ.setdefault("REAPER_IN_API", "0")
os.environ.setdefault("DATA_DIR", str(ROOT / "data_service_preview"))
os.environ.pop("VLM_API_BASE", None)
os.environ.pop("DET_API_BASE", None)

import fakeredis  # noqa: E402

from service import auth, jobqueue, redis_conn, worker  # noqa: E402

server = fakeredis.FakeServer()
redis_conn.set_clients(fakeredis.FakeRedis(server=server, decode_responses=True), fakeredis.FakeRedis(server=server))
RESULT = json.loads(Path(sys.argv[1]).read_text())


class Replay:
    def analyze(self, path, job_id="", progress=None):
        for stage, p in [("grounding", .05), ("scene_state", .15), ("perception", .4), ("checks", .6), ("evidence", .9)]:
            progress(stage, p)
            time.sleep(0.8)
        return json.loads(json.dumps(RESULT))


def inline_worker():
    worker.set_analyzer(Replay())
    while True:
        for q in jobqueue.QUEUES:
            raw = redis_conn.get_rq_redis().lpop(jobqueue.queue(q).key)
            if raw:
                jid = raw.decode().split("-")[0]
                worker.run_job(jid)
                break
        else:
            time.sleep(0.3)


auth.create_user("preview", "preview-password")
worker.state.set_info("workers", {"models": {"vlm": "replay", "vlm_backend": "replay", "detector": "-"}})
threading.Thread(target=inline_worker, daemon=True).start()

import uvicorn  # noqa: E402

from service.app import create_app  # noqa: E402

uvicorn.run(create_app(), host="127.0.0.1", port=int(os.getenv("PORT", "8010")))
