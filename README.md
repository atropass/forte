# ForteVision AI — контроль кассовой дисциплины по видео

Локальный сервис: загружаете `.mov` с потолочной камеры кассы → получаете вердикт
(`violation` / `no_violation`), а для каждого нарушения — тип (код каталога V01–V19), таймкод,
описание увиденного, 1–3 кадра-доказательства и норму НПА РК (документ + пункт).

* Все модели работают внутри сервиса (Qwen3-VL-4B-Instruct + RT-DETRv2, Apache-2.0). После загрузки
  весов интернет не нужен (`HF_HUB_OFFLINE=1`). Облачные AI API не используются.
* Без распознавания лиц: роли определяются только по положению и поведению.
* Результат — подсказка для проверяющего, а не юридическое заключение.

Документы: [ARCHITECTURE.md](ARCHITECTURE.md) · [VIOLATIONS.md](VIOLATIONS.md) · [LIMITATIONS.md](LIMITATIONS.md) ·
[MODELS.md](MODELS.md) · [DECISIONS.md](DECISIONS.md) · [PROGRESS.md](PROGRESS.md) · отчёты в `reports/`.

---

## 1. Запуск на GPU-сервере (продакшн, Docker)

Требования: Linux, NVIDIA GPU 24 GB (RTX 4090 / L4 / A10), драйвер + NVIDIA Container Toolkit, Docker Compose v2.

```bash
cp .env.example .env
```
Задайте в `.env` длинный случайный `API_KEY`:
```bash
python3 -c "import secrets;print(secrets.token_urlsafe(32))"
```
Скачайте веса один раз (нужен интернет, ~10 GB в `./weights`):
```bash
scripts/download_weights.sh
```
Соберите и запустите (UI + API на порту 80):
```bash
docker compose up -d --build
```
Готовность (модели загружаются ~1–3 мин): `curl http://<host>/healthz` → `"ok": true`.
UI: `http://<host>/` (API-ключ вводится в правом верхнем углу, хранится только в браузере).

Проверка бюджета времени на этом сервере (нужны веса и видео в `open/videos/`):
```bash
docker compose exec backend python3 scripts/bench.py
```

## 2. Локальный запуск на Mac (Apple silicon, MLX) — разработка/демо

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-mac.txt
```
```bash
scripts/download_weights.sh --mac
```
```bash
make serve-local
```
Откройте http://localhost:8000 и один раз введите ключ `dev-key` (кнопка API Key; ключ запоминается в браузере).
Свой ключ: `API_KEY=... make serve-local`. Если сервис запущен без `API_KEY` другим способом, он сам
сгенерирует ключ, сохранит в `data_service/api_key.txt` и выведет его в лог. На M1 Pro 16 GB анализ 20-секундного ролика
занимает ~2 мин (на GPU-сервере — в разы быстрее).

## 3. API

Все маршруты `/api/v1/*` требуют заголовок `X-API-Key`; без верного ключа — `401`.

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/api/v1/analyze` | загрузка видео (multipart, поле `file`, `.mov`, до 500 МБ) → `{"job_id": "...", "status": "processing"}` |
| GET | `/api/v1/jobs/{job_id}` | `processing` / `failed` (+`error`) / `done` + результат по схеме ТЗ |
| GET | `/api/v1/jobs/{job_id}/progress` | этап и прогресс (для UI) |
| GET | `/api/v1/jobs/{job_id}/details` | события, хронология, тайминги (для UI) |
| GET | `/api/v1/catalog` | справочник типов нарушений и что проверяется |
| GET | `/healthz` | готовность моделей (без ключа) |

Пример из ТЗ:
```bash
curl -X POST https://team.example.kz/api/v1/analyze -H "X-API-Key: <ключ>" -F "file=@video_017.mov"
```
```bash
curl -H "X-API-Key: <ключ>" https://team.example.kz/api/v1/jobs/3f2a9c1e
```
Ответ `done` содержит ровно поля ТЗ (`job_id, status, verdict, violations[type, timecode_sec,
description, evidence_frames[timestamp_sec, image_base64]]`). Дополнительные поля — только внутри
объекта нарушения: `type_code`, `subtype`, `rule` (документ, пункт, требование, URL adilet, `verified`),
`source`, `confidence`.

Задания хранятся на диске (`DATA_DIR`), перезапуск сервиса не теряет результаты; незавершённые
задания продолжаются после рестарта. Видео обрабатываются по одному (один GPU-воркер).

## 4. Воспроизводимость

| Команда | Что делает |
|---|---|
| `make audit` | Фаза 1: `data/labels.csv`, `reports/data_audit.md` |
| `make feasibility` | Фаза 2: точность VLM по вопросам, `reports/feasibility.md` |
| `make bench` | бюджет времени на ролик 145 с |
| `make eval` | Фаза 4: все обучающие видео → `reports/eval.md` |
| `python scripts/robustness.py` | Фаза 5: устойчивость к ракурсу → `reports/robustness.md` |
| `make test` | контрактные тесты API (без моделей) |

Датасет конфиденциален и в репозиторий не входит (`open/`, `*.mov`, `Yerkes.xlsx` в `.gitignore`).
