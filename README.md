# ForteVision AI — контроль кассовой дисциплины по видео

Сервис: загружаете `.mov` с потолочной камеры кассы → получаете вердикт (`violation` / `no_violation`),
а для каждого нарушения — тип (код каталога V01–V19), таймкод, описание увиденного, 1–3 кадра-доказательства
и норму НПА РК (документ + пункт).

* Модели работают на собственных GPU-серверах команды (Qwen3-VL-4B-Instruct + RT-DETRv2, Apache-2.0).
  Облачные AI API не используются, веса не скачиваются во время работы (`HF_HUB_OFFLINE=1`).
* Без распознавания лиц и без звука: роли определяются только по положению и поведению.
* Результат — подсказка для проверяющего, а не юридическое заключение.

Документы: [ARCHITECTURE.md](ARCHITECTURE.md) · [VIOLATIONS.md](VIOLATIONS.md) · [LIMITATIONS.md](LIMITATIONS.md) ·
[MODELS.md](MODELS.md) · [DECISIONS.md](DECISIONS.md) · [GPU_ACCESS.md](GPU_ACCESS.md) ·
[docs/SECURITY.md](docs/SECURITY.md) · [docs/RUNBOOK.md](docs/RUNBOOK.md) · [docs/DEPLOY_ON_PREM.md](docs/DEPLOY_ON_PREM.md) ·
отчёты в `reports/`.

---

## Доступ

| | |
|---|---|
| **Веб-интерфейс** | **https://forte-vision.fly.dev** |
| **API (base URL)** | **https://forte-vision.fly.dev** (маршруты `/api/v1/...`) |
| Логин и пароль для UI | демо-аккаунты ниже; остальные выдаёт администратор (раздел 1.3), самостоятельной регистрации нет |
| API-ключ (`X-API-Key`) | выдаёт администратор; жюри получает ключ лично, в репозитории ключей нет |
| Состояние сервиса | `GET /healthz` (процесс жив), `GET /readyz?strict=1` (готов анализировать: Redis, GPU-серверы, воркеры) |

### Демо-аккаунты (https://forte-vision.fly.dev)

| Назначение | Логин | Пароль |
|---|---|---|
| Демо / команда | `demo` | `Demo-Forte-7xK2pQ9mW4` |
| Жюри (веб-интерфейс) | `jury` | `Jury-Forte-3nV8rT5cY1` |

Репозиторий приватный; если он станет публичным — сменить пароли (`admin reset-password <логин>`). Аккаунты создаются
на Fly один раз (раздел 1.3):

```bash
fly ssh console -a forte-vision -C "entrypoint admin create-user demo --password Demo-Forte-7xK2pQ9mW4"
fly ssh console -a forte-vision -C "entrypoint admin create-user jury --password Jury-Forte-3nV8rT5cY1"
```

API-ключ жюри в репозиторий не входит — он передаётся лично.

### Веб-интерфейс

1. Откройте https://forte-vision.fly.dev и войдите (демо-аккаунт выше или логин от администратора).
2. Перетащите видео `.mov` / `.mp4` (до 100 МБ) и нажмите «Запустить анализ». Видно этап обработки и позицию
   в очереди.
3. Результат: вердикт; метки событий на таймлайне видео (клик — перемотка); кадры-доказательства (клик — крупно);
   карточки нарушений с описанием и нормой НПА РК (ссылка на adilet.zan.kz); вкладка «JSON API» — ровно тот
   ответ, который отдаёт API.
4. «История» — ваши прошлые проверки (видео не хранится, только отчёт); «НПА РК · справочник» — какие типы
   нарушений проверяются.

### API для жюри

Два шага, как в ТЗ: загрузить видео → получить `job_id` → опрашивать статус до `done` / `failed`.

```bash
BASE=https://forte-vision.fly.dev
KEY=<ключ>

# 1. загрузка (поле file, .mov)
curl -X POST "$BASE/api/v1/analyze" -H "X-API-Key: $KEY" -F "file=@video_017.mov"
# -> {"job_id":"4b8bac04543dc2a6","status":"processing"}

# 2. статус и результат
curl -H "X-API-Key: $KEY" "$BASE/api/v1/jobs/4b8bac04543dc2a6"
# -> {"job_id":"4b8bac04543dc2a6","status":"processing"}        пока идёт анализ
# -> {"job_id":"...","status":"done","verdict":"...","violations":[...]}   готово
# -> {"job_id":"...","status":"failed","error":"timeout"}       ошибка (код ниже)
```

Загрузка и ожидание одной командой (опрос раз в 3 с):

```bash
JOB=$(curl -s -X POST "$BASE/api/v1/analyze" -H "X-API-Key: $KEY" -F "file=@video_017.mov" \
      | python3 -c 'import sys, json; print(json.load(sys.stdin)["job_id"])')
until R=$(curl -s -H "X-API-Key: $KEY" "$BASE/api/v1/jobs/$JOB"); ! echo "$R" | grep -q '"processing"'; do sleep 3; done
echo "$R" > result.json
```

Python (`sdk/client.py`, только стандартная библиотека):

```python
from sdk.client import ForteVisionClient
fv = ForteVisionClient("https://forte-vision.fly.dev", api_key="<ключ>")
result = fv.analyze_and_wait("video_017.mov", timeout_s=300)   # dict по схеме ТЗ
```

Ответ `done` (реальный ответ сервиса, base64 сокращён):

```json
{
  "job_id": "4b8bac04543dc2a6",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Нарушение порядка хранения наличных денег на рабочем месте кассира (неубранная наличность / открытый сейф)",
      "timecode_sec": 56.8,
      "description": "Деньги предыдущей операции на столе при подходе следующего клиента. На 56.8 с к столу подходит следующий посетитель, при этом на столе лежат наличные деньги от предыдущей операции. … Основание: п. 20 (абзац второй) «Правила осуществления кассовых операций…» (постановление Правления НБРК от 29 ноября 2019 года № 231).",
      "evidence_frames": [
        {"timestamp_sec": 56.8,  "image_base64": "/9j/4AAQSkZJRgAB…"},
        {"timestamp_sec": 55.75, "image_base64": "/9j/4AAQSkZJRgAB…"},
        {"timestamp_sec": 57.75, "image_base64": "/9j/4AAQSkZJRgAB…"}
      ],
      "type_code": "V03",
      "subtype": "Деньги предыдущей операции на столе при подходе следующего клиента",
      "rule": {"clause": "п. 20 (абзац второй)", "act": "постановление от 29 ноября 2019 года № 231 (рег. № 19680)",
               "url": "https://adilet.zan.kz/rus/docs/V1900019680", "verified": true, "…": "…"},
      "source": "timeline+vlm",
      "confidence": 0.99
    }
  ]
}
```

* Верхний уровень — ровно поля ТЗ: `job_id`, `status`, `verdict` (`violation` | `no_violation`), `violations`
  (пустой список, если нарушений нет). У нарушения обязательные поля ТЗ: `type`, `timecode_sec`, `description`,
  `evidence_frames` (1–3 JPEG в base64). Дополнительно: `type_code`, `subtype`, `type_id`, `rule` (норма НПА РК
  со ссылкой), `source`, `confidence`. Схема проверяется тестами (`tests/test_api.py`, `SPEC_SCHEMA`).
* **Время:** анализ видео ~20–60 с на GPU-серверах; жёсткий лимит на задание 290 с (лимит жюри — 5 минут). При
  превышении — `failed` с кодом `timeout`, а не неполный вердикт.
* **Коды `failed`:** `timeout`, `inference_unavailable` (GPU-сервер недоступен), `invalid_video`, `worker_lost`,
  `internal_error`.
* **Ошибки загрузки (сразу):** `401` нет/неверный ключ, `413` больше 100 МБ, `415` не MOV/MP4, `422` видео не
  читается / длиннее 15 мин / больше 4K.
* Повторная загрузка того же файла тем же ключом возвращает тот же `job_id` (без повторного анализа).
* Задания, созданные по API-ключу, видны только API-ключам; задания из UI — только их владельцу.

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/api/v1/analyze` | видео (multipart, поле `file`) → `{"job_id", "status": "processing"}` |
| GET | `/api/v1/jobs/{job_id}` | `processing` / `failed` (+ `error`) / `done` + результат по схеме ТЗ |
| GET | `/api/v1/jobs/{job_id}/progress` | этап, прогресс, позиция в очереди |
| GET | `/api/v1/jobs/{job_id}/details` | события, хронология, тайминги, отладочные оценки проверок |
| GET | `/api/v1/catalog` | справочник типов нарушений и что проверяется |
| GET | `/healthz`, `/readyz` | состояние (без ключа) |
| GET | `/metrics` | метрики Prometheus (`Authorization: Bearer $METRICS_TOKEN`; на Fly не настроено) |

---

## Архитектура сервиса

```
 жюри (X-API-Key) ─┐
 веб-UI (логин)  ──┴─► API (FastAPI) + статический UI
                         │
            Redis ◄──────┤   очереди api (жюри, приоритет) / ui, статусы заданий, пользователи, ключи, лимиты
                         │
                       worker ×N   RQ: отдельный процесс на каждое видео, жёсткий лимит 290 с, 1 повтор при сбое GPU
                         │   видео → анализ (pipeline/) → результат и лог → видео удаляется
                         ▼
               vLLM (Qwen3-VL-4B, RTX 3090)      CV-сервис (RT-DETRv2, RTX 3070)      ← GPU-поды RunPod
```

* Задание никогда не остаётся «processing» навсегда: сторож (watchdog) завершает его по лимиту, reaper
  перезапускает задания упавших воркеров. При ошибке ответ всегда `failed` с кодом, а не выдуманный вердикт.
* Компоненты стандартные (контейнеры, Redis, S3 или общий диск, переменные окружения) — сервис встраивается в
  инфраструктуру банка ([docs/DEPLOY_ON_PREM.md](docs/DEPLOY_ON_PREM.md)).

## 0. Локальный запуск

```bash
cp .env.example .env        # VLM_API_BASE / VLM_API_KEY / DET_API_BASE / DET_API_KEY — см. GPU_ACCESS.md
make up                     # Docker: UI + API на http://localhost:8080 (как в продакшне)
docker compose run --rm init admin create-user demo          # пароль будет выведен
docker compose run --rm init admin create-api-key --name dev # ключ для API
```

Без Docker (разработка; нужен ffmpeg ≥ 5 — в Ubuntu 22.04 он 4.4, анализ упадёт с `invalid_video`):
```bash
make setup && make redis && make check-vlm
make admin ARGS="create-user demo"
make run                    # UI + API на http://localhost:8000 и воркеры
```

| Команда | Что делает |
|---|---|
| `make up` / `make down` | стек в Docker: frontend (nginx :8080) + api + worker + Redis |
| `make run` | UI + API + воркеры без Docker (:8000) |
| `make dev` | UI с горячей перезагрузкой на :3000 (вместе с `make run`) |
| `make admin ARGS="…"` | пользователи и API-ключи (раздел 1.3) |
| `make test` | тесты без моделей, Redis и GPU |
| `make eval-api URL=… KEY=…` | все обучающие видео через публичный API, метрики → `reports/eval_remote.md` |
| `make loadtest URL=… KEY=… BURST=10` | одновременная загрузка N видео, как у жюри |

Без GPU-серверов UI можно посмотреть на сохранённом результате: `python scripts/ui_preview_server.py result.json`
(логин `preview` / `preview-password`).

## 1. Продакшн

### 1.1 Fly.io (публичный адрес https://forte-vision.fly.dev)

Одна машина ([fly.toml](fly.toml), [docker/fly.Dockerfile](docker/fly.Dockerfile),
[docker/fly-start.sh](docker/fly-start.sh)): Redis + API со статическим UI + воркеры, состояние на томе
`forte_data` (`/data`). Модели — на GPU-подах RunPod. Каждый пуш в `main`, прошедший CI, выкатывается
автоматически; вручную — `fly deploy --remote-only`.

```bash
# адреса и токены GPU-серверов + bootstrap-ключ жюри (значения — у администратора, см. GPU_ACCESS.md)
fly secrets set -a forte-vision API_KEY=<ключ жюри> \
    VLM_API_BASE=https://<vlm-pod>-8000.proxy.runpod.net/v1 VLM_API_KEY=<токен> \
    DET_API_BASE=https://<cv-pod>-8000.proxy.runpod.net DET_API_KEY=<токен>
# учётные записи и дополнительные ключи
fly ssh console -a forte-vision -C "entrypoint admin create-user jury"
fly ssh console -a forte-vision -C "entrypoint admin create-api-key --name jury2"
fly logs -a forte-vision
```

**Размер машины:** сейчас `shared-cpu-1x` / 1 GB — этого хватает для одиночных проверок, но не для нагрузки жюри:
перед анализом видео декодируется на CPU, и при 5 одновременных видео задания ждут CPU (измерено: до 234 с на видео,
GPU простаивают). Перед проверкой жюри: `fly scale vm shared-cpu-4x --memory 4096 -a forte-vision` и затем
`make loadtest URL=https://forte-vision.fly.dev KEY=… BURST=5`. Цель — все видео < 300 с от загрузки до результата.

### 1.2 Docker Compose (любой хост, в т. ч. инфраструктура банка)

```bash
cp .env.example .env        # REDIS_PASSWORD, PUBLIC_ORIGIN, адреса и токены GPU-серверов
docker compose up -d --build
docker compose run --rm init admin create-api-key --name jury       # ключ показывается один раз
docker compose run --rm init admin create-user jury                 # учётная запись UI
curl http://localhost:8080/readyz
```

| Сервис | Образ | Порт |
|---|---|---|
| `frontend` | `forte-frontend` (nginx, не root) | **8080** — единственный публичный |
| `api` | `forte-backend`, роль `api` | 8000, только внутри сети compose |
| `worker` | `forte-backend`, роль `worker` (`WORKER_COUNT` процессов) | — |
| `redis` | Redis 7 (AOF) | — |

`PUBLIC_ORIGIN` должен совпадать с адресом, который открывают пользователи (иначе вход отклоняется как
межсайтовый запрос). HTTPS: reverse proxy банка перед `:8080` или встроенный Caddy —
`DOMAIN=forte.example.kz docker compose --profile tls up -d`. Свой Redis: удалите сервис `redis` и задайте
`REDIS_URL`. Объектное хранилище вместо тома (несколько хостов; Ceph / AWS / MinIO): `STORAGE_BACKEND=s3` и `S3_*`.

### 1.3 Учётные записи и ключи

Самостоятельной регистрации нет — доступ выдаёт администратор. Команда та же везде, меняется только префикс:
локально `make admin ARGS="…"`, в Docker `docker compose run --rm init admin …`, на Fly
`fly ssh console -a forte-vision -C "entrypoint admin …"`.

```bash
admin create-user <логин> [--password …]      # без --password пароль генерируется и выводится один раз
admin reset-password <логин>                  # все сессии пользователя отзываются
admin delete-user <логин>
admin list-users
admin create-api-key --name jury              # ключ выводится один раз, хранится только SHA-256
admin list-api-keys
admin revoke-api-key jury                     # действует сразу, без перезапуска
admin purge --older-than-days 30              # результаты и логи старше N дней
```

Ключи из переменных `API_KEY` / `API_KEYS` принимаются как bootstrap-ключи (на Fly ключ жюри задан так).
Ключи и пароли передаются лично, не через репозиторий и не в общих чатах.

### 1.4 CI/CD

[.github/workflows/ci.yml](.github/workflows/ci.yml) на каждый пуш и PR: тесты, typecheck и сборка UI, проверка
секретов (gitleaks), сборка обоих Docker-образов и смоук-тест стека. Из `main`: образы публикуются в GHCR
(`ghcr.io/atropass/forte-backend`, `ghcr.io/atropass/forte-frontend`, теги `sha-<коммит>` и `latest`), затем
деплой на Fly.io (секрет репозитория `FLY_API_TOKEN`). Для своего сервера: `REGISTRY=ghcr.io/atropass`,
`IMAGE_TAG=sha-<коммит>` в `.env`, `docker compose pull && docker compose up -d --no-build`.

### 1.5 GPU-серверы

vLLM (Qwen3-VL-4B, RTX 3090, 330 W) и CV-сервис (RT-DETRv2, RTX 3070) — поды RunPod за nginx-шлюзами с токенами.
Установка нового пода с нуля, ротация ключей, диагностика, резерв — [GPU_ACCESS.md](GPU_ACCESS.md).

## 2. Качество (40 размеченных видео через публичный API, 5 одновременно)

| Метрика | Значение |
|---|---|
| Точность вердикта | 77,5 % (31/40) |
| Ложные срабатывания на чистых видео | 1/19 (X-16: инкассаторская сумка принята за личные вещи) |
| Полнота по нарушениям | 61,9 % (13/21) |
| Тип нарушения (подтверждённые метки) | 92,3 % (12/13) |
| Ошибка таймкода, среднее / медиана | 1,2 / 0,9 с |
| Ответы по схеме ТЗ | 40/40 |
| Время от загрузки до результата (GPU-поды, 4 воркера) | среднее 44 с, максимум 102 с, > 5 мин: 0/40 |

Подробно по каждому видео — `reports/eval_2026-10-18_rtx3090.md`; известные пропуски и их причины —
[LIMITATIONS.md](LIMITATIONS.md).

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

Датасет конфиденциален и в репозиторий не входит (`open/`, `*.mov`, `Yerkes.xlsx` в `.gitignore`).
Локально на Mac (Apple silicon) модель можно запустить в процессе: `VLM_BACKEND=mlx` в `.env`
(`scripts/download_weights.sh --mac`), воркер тогда работает в режиме одного процесса.
