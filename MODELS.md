# Models and licenses

All models run inside the service; weights are downloaded once by `scripts/download_weights.sh`.
No cloud AI API is called at any point.

| Component | Model | License | Source | Notes |
|---|---|---|---|---|
| VLM (default) | Qwen3-VL-4B-Instruct | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct | vLLM on GPU |
| VLM (dev, Mac) | mlx-community/Qwen3-VL-4B-Instruct-4bit | Apache-2.0 | https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit | MLX 4-bit conversion of the above |
| VLM (alternative) | Qwen3-VL-8B-Instruct (+ mlx 4-bit) | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct | evaluated, not default |
| VLM (evaluated) | Qwen2.5-VL-7B-Instruct (mlx 4-bit) | Apache-2.0 | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct | evaluated, not used |
| Person detector | PekingU/rtdetr_v2_r18vd (RT-DETRv2) | Apache-2.0 | https://huggingface.co/PekingU/rtdetr_v2_r18vd | COCO classes |
| Detector (alternative) | PekingU/rtdetr_v2_r50vd | Apache-2.0 | https://huggingface.co/PekingU/rtdetr_v2_r50vd | 2× slower, not needed |

Deliberately **not** used: Ultralytics YOLO (AGPL-3.0), Qwen2.5-VL-3B (Qwen Research License, non-commercial;
its HF card has no license field), any face recognition model.

Libraries: PyTorch (BSD-3), transformers (Apache-2.0), vLLM (Apache-2.0), MLX / mlx-vlm (MIT),
FFmpeg (LGPL build, invoked as an external binary), FastAPI (MIT), React (MIT), Pillow (MIT-CMU), NumPy (BSD).
Tracking: own greedy IoU tracker (no ByteTrack dependency needed).
