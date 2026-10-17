# ForteVision AI — контроль кассовой дисциплины по видео

Сервис: загружаете `.mov` с потолочной камеры кассы → получаете вердикт (`violation` / `no_violation`),
а для каждого нарушения — тип (код каталога V01–V19), таймкод, описание увиденного, 1–3 кадра-доказательства
и норму НПА РК (документ + пункт).

* Модели работают на собственных GPU-серверах команды (Qwen3-VL-4B-Instruct + RT-DETRv2, Apache-2.0).
  Облачные AI API не используются, весы не скачиваются во время работы (`HF_HUB_OFFLINE=1`).
* Без распознавания лиц и без звука: роли определяются только по положению и поведению.
* Результат — подсказка для проверяющего, а не юридическое заключение.

Документы: [ARCHITECTURE.md](ARCHITECTURE.md) · [VIOLATIONS.md](VIOLATIONS.md) · [LIMITATIONS.md](LIMITATIONS.md) ·
[MODELS.md](MODELS.md) · [DECISIONS.md](DECISIONS.md) · [GPU_ACCESS.md](GPU_ACCESS.md) ·
[docs/SECURITY.md](docs/SECURITY.md) · [docs/RUNBOOK.md](docs/RUNBOOK.md) · [docs/DEPLOY_ON_PREM.md](docs/DEPLOY_ON_PREM.md) ·
отчёты в `reports/`.

---

## Архитектура сервиса

```
 жюри (X-API-Key) ─┐
 веб-UI (логин)  ──┴─► frontend  nginx :8080: статический UI, прокси /api → api
                         │
                       api   FastAPI, без состояния: проверка и приём видео, статусы
                         │
            Redis ◄──────┤   очереди api (жюри, приоритет) / ui, статусы заданий, пользователи, ключи, лимиты
                         │
                       worker ×N   RQ: отдельный процесс на каждое видео, жёсткий лимит 290 с, 1 повтор при сбое GPU
                         │   видео из хранилища → анализ (pipeline/) → результат и лог в хранилище → видео удаляется
                         ▼
               vLLM (Qwen3-VL, GPU)      CV-сервис (RT-DETRv2, GPU)

 Хранилище: любой S3 API (MinIO / Ceph / AWS / R2) или общий диск (STORAGE_BACKEND=fs).
```

* Два Docker-образа: `forte-backend` (api, worker, admin; Python 3.11 + ffmpeg 5.1) и `forte-frontend`
  (nginx со статическим UI). Для браузера это один адрес: cookie сессии и CSRF-проверки работают без CORS.
* Компоненты стандартные (контейнеры, Redis, S3, переменные окружения) — сервис встраивается в инфраструктуру банка
  ([docs/DEPLOY_ON_PREM.md](docs/DEPLOY_ON_PREM.md)).
* API и воркеры масштабируются независимо; падение или зависание одного анализа не задевает остальные.
* Задание никогда не остаётся «processing» навсегда: сторож (watchdog) завершает его по лимиту времени, reaper
  перезапускает задания упавших воркеров. При ошибке ответ всегда `failed` с кодом, а не выдуманный вердикт.

## 0. Быстрый старт (локально, модели на GPU-серверах)

```bash
cp .env.example .env        # VLM_API_BASE / VLM_API_KEY / DET_API_BASE / DET_API_KEY — см. GPU_ACCESS.md
make up                     # то же, что продакшн: UI + API на http://localhost:8080 (раздел 1.1)
```

Без Docker (разработка; нужен ffmpeg ≥ 5 — в Ubuntu 22.04 он 4.4, анализ упадёт с `invalid_video`):
```bash
make setup                  # Python venv + зависимости UI
make redis                  # локальный Redis в Docker
make check-vlm              # доступен ли vLLM-сервер
make admin ARGS="create-user demo"     # пароль будет выведен
make run                    # UI + API на http://localhost:8000 и воркеры
```

| Команда | Что делает |
|---|---|
| `make run` | UI + API + воркеры локально |
| `make dev` | UI с горячей перезагрузкой на :3000 (вместе с `make run`) |
| `make admin ARGS="…"` | пользователи и API-ключи (см. ниже) |
| `make up` | продакшн-стек в Docker: frontend (nginx :8080) + api + worker + Redis |
| `make images` | собрать оба образа |
| `make test` | тесты без моделей, Redis и GPU |
| `make loadtest URL=… KEY=…` | одновременная загрузка N видео, как у жюри |
| `make eval-api URL=… KEY=…` | все обучающие видео через публичный API, метрики |

Без GPU-серверов UI можно посмотреть на сохранённом результате: `python scripts/ui_preview_server.py result.json`
(логин `preview` / `preview-password`).

## 1. Продакшн

### 1.1 Docker Compose (любой хост)

```bash
cp .env.example .env        # REDIS_PASSWORD, PUBLIC_ORIGIN, адреса и токены GPU-серверов
docker compose up -d --build
docker compose run --rm init admin create-api-key --name jury       # ключ показывается один раз
docker compose run --rm init admin create-user jury                 # учётная запись UI для жюри
curl http://localhost:8080/readyz
```

| Сервис | Образ | Порт |
|---|---|---|
| `frontend` | `forte-frontend` (nginx, не root) | **8080** — единственный публичный |
| `api` | `forte-backend`, роль `api` | 8000, только внутри сети compose |
| `worker` | `forte-backend`, роль `worker` (`WORKER_COUNT` процессов) | — |
| `redis` | Redis 7 (AOF) | — |

Видео (временно), результаты и логи лежат на томе `data`, общем для `api` и `worker`.

`PUBLIC_ORIGIN` должен совпадать с адресом, который открывают пользователи (иначе вход отклоняется как
межсайтовый запрос). HTTPS: reverse proxy банка перед `:8080` или встроенный Caddy —
`DOMAIN=forte.example.kz docker compose --profile tls up -d`. Свой Redis: удалите сервис `redis` и задайте
`REDIS_URL`. Объектное хранилище вместо тома (несколько хостов; Ceph / AWS / MinIO): `STORAGE_BACKEND=s3` и `S3_*`.

### 1.2 Образы и выкладка

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) на каждый пуш и PR: тесты, typecheck и сборка UI,
проверка секретов, сборка обоих образов и смоук-тест стека (frontend → api → Redis). Из `main` образы
публикуются в GHCR: `ghcr.io/atropass/forte-backend` и `ghcr.io/atropass/forte-frontend` с тегами `sha-<коммит>` и
`latest`. На сервере:

```bash
echo "REGISTRY=ghcr.io/atropass" >> .env; echo "IMAGE_TAG=sha-<коммит>" >> .env
docker compose pull && docker compose up -d --no-build
```

Откат — предыдущий `IMAGE_TAG`. Миграция пользователей из версии 1 (SQLite):
`docker compose run --rm -v /path/users.db:/users.db:ro init admin import-sqlite-users /users.db`.

### 1.3 Учётные записи и ключи

Самостоятельной регистрации нет — доступ выдаёт администратор:

```bash
python -m service.admin create-user <логин> [--password …]      # без --password пароль генерируется
python -m service.admin reset-password <логин>                  # все сессии пользователя отзываются
python -m service.admin delete-user <логин>
python -m service.admin create-api-key --name jury              # хранится только SHA-256 ключа
python -m service.admin list-api-keys
python -m service.admin revoke-api-key jury                     # действует сразу, без перезапуска
python -m service.admin purge --older-than-days 30              # результаты и логи старше N дней
```

Ключ и пароль жюри передаются лично, не через репозиторий.

### 1.4 GPU-серверы

vLLM (Qwen3-VL) и CV-сервис (RT-DETRv2) работают на отдельных GPU-машинах за nginx-шлюзами с токенами.
Настройка, ротация ключей и диагностика — [GPU_ACCESS.md](GPU_ACCESS.md).

## 2. API

Доступ по ключу в заголовке `X-API-Key` (задания, созданные по API-ключам) или по сессии веб-UI (только свои
задания). Без верного ключа — `401`.

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/api/v1/analyze` | видео (multipart, поле `file`, `.mov`, до 100 МБ) → `{"job_id": "...", "status": "processing"}` |
| GET | `/api/v1/jobs/{job_id}` | `processing` / `failed` (+ код `error`) / `done` + результат по схеме ТЗ |
| GET | `/api/v1/jobs/{job_id}/progress` | этап, прогресс, позиция в очереди (для UI) |
| GET | `/api/v1/jobs/{job_id}/details` | события, хронология, тайминги (для UI) |
| GET | `/api/v1/catalog` | справочник типов нарушений и что проверяется |
| GET | `/healthz` | процесс жив (без ключа) |
| GET | `/readyz` | Redis и хранилище доступны; `?strict=1` — ещё и GPU-серверы и воркеры (без ключа) |
| GET | `/metrics` | метрики Prometheus (`Authorization: Bearer $METRICS_TOKEN`) |

```bash
curl -X POST https://team.example.kz/api/v1/analyze -H "X-API-Key: <ключ>" -F "file=@video_017.mov"
curl -H "X-API-Key: <ключ>" https://team.example.kz/api/v1/jobs/<job_id>
```

Ответ `done` содержит ровно поля ТЗ (`job_id, status, verdict, violations[type, timecode_sec, description,
evidence_frames[timestamp_sec, image_base64]]`). Дополнительные поля — только внутри объекта нарушения:
`type_code`, `subtype`, `rule` (документ, пункт, требование, URL adilet, `verified`), `source`, `confidence`.

Коды ошибок `failed`: `timeout` (анализ не уложился в лимит), `inference_unavailable` (GPU-сервер недоступен),
`invalid_video`, `worker_lost`, `internal_error`. Невалидные файлы отклоняются сразу при загрузке:
`413` (больше лимита), `415` (не MOV/MP4), `422` (не читается, слишком длинное или большое видео).
Повторная загрузка того же файла тем же ключом возвращает тот же `job_id`.

SDK: [sdk/client.py](sdk/client.py) (`ForteVisionClient.analyze_and_wait`), [sdk/client.ts](sdk/client.ts).

## 3. Воспроизводимость

| Команда | Что делает |
|---|---|
| `make audit` | Фаза 1: `data/labels.csv`, `reports/data_audit.md` |
| `make feasibility` | Фаза 2: точность VLM по вопросам, `reports/feasibility.md` |
| `make bench` | бюджет времени на ролик 145 с |
| `make calibrate` | подбор порогов под модель → `pipeline/thresholds.yaml`, `reports/calibration.md` |
| `make eval` | Фаза 4: все обучающие видео → `reports/eval.md` |
| `python scripts/robustness.py` | Фаза 5: устойчивость к ракурсу → `reports/robustness.md` |
| `make eval-api` | оценка задеплоенного сервиса через API → `reports/eval_remote.md` |
| `make loadtest` | нагрузка как у жюри → `reports/loadtest.md` |
| `make test` | тесты API, безопасности, очереди, CV-сервиса и логики решений (без моделей) |

Датасет конфиденциален и в репозиторий не входит (`open/`, `*.mov`, `Yerkes.xlsx` в `.gitignore`). Локально на Mac
(Apple silicon) модель можно запустить в процессе: `VLM_BACKEND=mlx` в `.env` (`scripts/download_weights.sh --mac`),
воркер тогда работает в режиме одного процесса.
