"""ANALYZER_FACTORY=tests.stub_factory:make for integration runs without models.

Behaviour by video content: b"SLOW" sleeps 600 s (timeout test), b"CRASH" kills the process, b"FAIL" raises.
"""
import json
import os
import time
from pathlib import Path

from tests.conftest import FAKE


class Stub:
    def analyze(self, path, job_id="", progress=None):
        data = Path(path).read_bytes()
        progress("checks", 0.6)
        if b"SLOW" in data:
            time.sleep(600)
        if b"CRASH" in data:
            os._exit(137)
        if b"FAIL" in data:
            raise RuntimeError("decoder error")
        time.sleep(float(os.getenv("STUB_DELAY_S", "0.5")))
        return json.loads(json.dumps(FAKE))


def make():
    return Stub()
