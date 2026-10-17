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

## 0. Быстрый старт: приложение здесь + Qwen на нашем GPU-поде

Модель Qwen3-VL-4B работает на нашем GPU-сервере (vLLM, OpenAI-совместимый API). Приложение (детектор,
правила, API, UI) запускается на любой машине и обращается к нему по HTTP.

```bash
cp .env.example .env
```
В `.env` укажите `VLM_API_BASE` (…/v1) и `VLM_API_KEY` (токен шлюза пода), `API_KEY` — ключ нашего API.
```bash
make setup
```
```bash
make check-vlm
```
```bash
make run
```
Откройте http://localhost:8000, введите `API_KEY` из `.env` (по умолчанию `dev-key`). Остальные команды: `make help`.

| Команда | Что делает |
|---|---|
| `make run` | UI + API на :8000, модель из `.env` |
| `make dev` | UI с горячей перезагрузкой на :3000 (вместе с `make run`) |
| `make check-vlm` | доступность пода, имя модели, задержка |
| `make calibrate` | подбор порогов на обучающих видео этой моделью → `pipeline/thresholds.yaml` |
| `make eval` | все обучающие видео → `reports/eval.md` |
| `make eval-api URL=…` | проверка запущенного приложения через его API, как у жюри |
| `make test` | тесты без моделей |

## 1. Запуск на GPU-сервере (продакшн, Docker)

Требования: Linux VM или выделенный сервер (не «pod»/контейнер), NVIDIA GPU ≥ 24 GB (рекомендуется L40S 48 GB),
драйвер + NVIDIA Container Toolkit, Docker Compose v2, диск ≥ 100 GB, публичный IP.

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

### 1.1 Проверка задеплоенной версии (порядок)

Скопируйте обучающие видео на сервер в `open/videos/` (они монтируются в контейнер только для
калибровки и бенчмарка, в образ не попадают).

**Шаг 1 — калибровка порогов** под эту модель и этот GPU. Сервис нужно остановить: вторая копия
модели рядом с работающим vLLM не поместится в память GPU.
```bash
docker compose stop backend
```
```bash
docker compose run --rm backend python3 scripts/calibrate.py
```
Пороги пишутся в `data_service/thresholds.yaml` (переживают пересборку), отчёт — в `reports/calibration.md`
внутри контейнера (выводится в консоль).
```bash
docker compose start backend
```

**Шаг 2 — прогон всех обучающих видео через публичный API**, как это сделает жюри (с ноутбука):
```bash
.venv/bin/python scripts/eval_remote.py --url http://<host> --key <API_KEY>
```
Метрики (точность, полнота, ложные срабатывания, ошибка таймкода) — в `reports/eval_remote.md`,
ответы — в `reports/eval/remote/`. Каждый ответ проверяется на соответствие схеме ТЗ.

**Шаг 3 — нагрузка как у жюри** (5 видео одновременно, у каждого должно быть ≤ 5 мин с учётом очереди):
```bash
.venv/bin/python scripts/eval_remote.py --url http://<host> --key <API_KEY> --parallel 5 --only X-14,X-39,X-4,X-38,X-13 --force
```

Отдельный бенчмарк по этапам (тоже при остановленном backend):
```bash
docker compose run --rm backend python3 scripts/bench.py
```

### 1.2 Вторая модель для удостоверений (необязательно, GPU 48 GB)

Для эпизодов передачи (V05 удостоверение клиента, V13 документы инкассатора) можно усреднять ответы
двух моделей. В `.env`: `VLM_AUX_MODEL=Qwen/Qwen2.5-VL-7B-Instruct`, веса:
`AUX_MODEL=Qwen/Qwen2.5-VL-7B-Instruct scripts/download_weights.sh`. Сравните `eval_remote.py` с ней и без неё.

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
| `make test` | тесты API и логики решений (без моделей) |
| `scripts/calibrate.py` | подбор порогов под модель → `thresholds.yaml`, `reports/calibration.md` |
| `scripts/eval_remote.py` | оценка задеплоенного сервиса через API → `reports/eval_remote.md` |

Датасет конфиденциален и в репозиторий не входит (`open/`, `*.mov`, `Yerkes.xlsx` в `.gitignore`).
