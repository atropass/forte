"""Key-in-lock test: locate the lock inside the safe crop, then ask about a tiny upscaled patch."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from feasibility import video_path  # noqa: E402
from pipeline.frames import frame_at  # noqa: E402
from pipeline.vlm import QUESTIONS, VLM, crop_box, parse_answer, parse_json, to_pil  # noqa: E402
from pipeline.vlm_backends import make_backend  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-VL-4B-Instruct-4bit"
g = json.loads((ROOT / "cache" / "grounding.json").read_text())
vlm = VLM(make_backend("mlx", model), max_side=896, cache_dir=ROOT / "cache" / "vlm")
PRE = QUESTIONS["preamble"].strip()
LOCATE = ('This is a close-up of a grey metal safe. Locate the lock / keyhole on the safe door. '
          'Return JSON only: {"lock": [x1, y1, x2, y2]} in 0-1000 coordinates relative to this image.')
Q_KEY = ("Strongly magnified close-up of a safe door lock (it will look blurry; judge shapes). Is something sticking out of the round lock, such as a key head, key shaft or key tag "
         "protruding from the lock), as opposed to an empty round keyhole?")
items = {"yes": [("X-22", 16), ("X-22", 15), ("X-24", 10), ("X-24", 11), ("X-46", 5), ("X-46", 12)],
         "no": [("X-23", 16), ("X-25", 29), ("X-27", 24), ("X-21", 12), ("X-25", 20), ("X-43", 10)]}
out_dir = ROOT / "reports" / "feasibility" / "lock_patches"
out_dir.mkdir(parents=True, exist_ok=True)
ok = n = 0
for exp, lst in items.items():
    for vid, t in lst:
        f = frame_at(video_path(vid), t)
        safe = crop_box(f, g[vid]["safe"], pad=0.05)
        big = np.asarray(Image.fromarray(safe).resize((safe.shape[1] * 3, safe.shape[0] * 3), Image.LANCZOS))
        j = parse_json(vlm.raw([(LOCATE, [to_pil(big, 896)])], max_tokens=60)[0]) or {}
        lock = j.get("lock")
        if isinstance(lock, list) and len(lock) == 4:
            patch = crop_box(safe, lock, pad=0.6, min_side=20)
        else:  # fallback: left-middle of the safe door, where the lock is in all training setups
            hh, ww = safe.shape[:2]
            patch = safe[int(hh * 0.3):int(hh * 0.7), 0:int(ww * 0.45)]
        patch = Image.fromarray(patch).resize((patch.shape[1] * 10, patch.shape[0] * 10), Image.LANCZOS)
        patch.save(out_dir / f"{vid}_{t}_{exp}.png")
        a = parse_answer(vlm.raw([(PRE + "\nQuestion: " + Q_KEY, [patch])])[0])
        n += 1
        ok += a.answer == exp
        print(vid, t, "exp", exp, "got", a.answer, "| lock", lock, "|", a.evidence, flush=True)
print(f"key lock-patch acc {ok}/{n}")
