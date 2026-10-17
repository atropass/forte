// Client for the ForteVision backend (service/app.py). Same origin in production (static export served by
// the backend); in `pnpm dev` set NEXT_PUBLIC_API_BASE=http://localhost:8000.
// Auth: login session in an HttpOnly cookie (sent with credentials: 'include'); the X-API-Key is for API clients only.
export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? '').replace(/\/$/, '')
export const MAX_UPLOAD_MB = Number(process.env.NEXT_PUBLIC_MAX_UPLOAD_MB ?? 100)
export const VIDEO_EXT = /\.(mov|mp4|m4v)$/i

export type EvidenceFrame = { timestamp_sec: number; image_base64: string }
export type LegalRef = {
  document: string; issuer: string; act: string; clause: string | null; requirement: string
  interpretation: string | null; url: string; verified: boolean; short: string
}
export type Violation = {
  type: string; timecode_sec: number; description: string; evidence_frames: EvidenceFrame[]
  type_code?: string; subtype?: string; type_id?: string; rule?: LegalRef | null; source?: string; confidence?: number
}
export type JobResult =
  | { job_id: string; status: 'processing' }
  | { job_id: string; status: 'failed'; error: string }
  | { job_id: string; status: 'done'; verdict: 'violation' | 'no_violation'; violations: Violation[] }
export type Progress = {
  job_id: string; status: 'processing' | 'done' | 'failed'; stage: string | null; progress: number | null
  created: number | null; started: number | null; finished: number | null; elapsed_s: number | null
  filename: string | null; size_bytes: number | null; error: string | null
  queue_position: number; models_ready: boolean
}
export type CatalogItem = {
  code: string; category: string; category_ru: string | null; type_ru: string
  status: 'detected' | 'low_confidence' | 'untested' | 'disabled' | 'not_detected'; note: string | null
  clause: string | null; verified: boolean
}
export type Models = Record<string, string | null>
export type Catalog = { items: CatalogItem[]; categories: Record<string, string>; models: Models }
export type HistoryItem = {
  job_id: string; filename: string | null; size_bytes: number | null; created: number | null
  finished: number | null; elapsed_s: number | null
  status: 'processing' | 'done' | 'failed'; stage: string | null; error: string | null
  verdict: 'violation' | 'no_violation' | null; n_violations: number | null
}
export type Details = {
  job: Record<string, unknown>
  debug: { timing_sec?: Record<string, number>; duration_sec?: number; events?: { kind: string; t: number }[] }
  models: Models
}

/** Job failure codes from the API -> text for the user. */
export const JOB_ERROR_RU: Record<string, string> = {
  timeout: 'Анализ не уложился в отведённое время (5 минут). Попробуйте ещё раз.',
  inference_unavailable: 'GPU-сервер анализа временно недоступен. Попробуйте позже.',
  invalid_video: 'Видео не удалось прочитать. Проверьте файл .MOV/.MP4.',
  worker_lost: 'Обработчик был перезапущен во время анализа. Загрузите видео ещё раз.',
  result_unavailable: 'Результат анализа недоступен.',
  internal_error: 'Внутренняя ошибка анализа. Попробуйте ещё раз.',
}
export const jobErrorText = (code: string | null | undefined) => (code && JOB_ERROR_RU[code]) || 'Неизвестная ошибка анализа'

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message) }
}

const STATUS_RU: Record<number, string> = {
  0: 'Сервис недоступен. Проверьте подключение.',
  401: 'Требуется вход',
  403: 'Запрос отклонён',
  404: 'Не найдено',
  413: `Файл больше ${MAX_UPLOAD_MB} МБ`,
  415: 'Поддерживаются только видео .MOV / .MP4',
  422: 'Видео не прошло проверку',
  429: 'Слишком много запросов, попробуйте позже',
  503: 'Сервис временно недоступен',
}

function errorText(status: number, body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (status === 422 && typeof detail === 'string') return `Видео не прошло проверку: ${detail}`
  if (status === 401 && typeof detail === 'string' && detail.startsWith('invalid username')) return 'Неверный логин или пароль'
  if (status === 429 && typeof detail === 'string' && detail.includes('in progress')) return 'Дождитесь завершения текущих проверок'
  return STATUS_RU[status] ?? (status >= 500 ? 'Ошибка сервера, попробуйте позже' : `Ошибка ${status}`)
}

async function request<T>(path: string, init?: { method: 'POST'; body?: unknown }): Promise<T> {
  let r: Response
  try {
    r = await fetch(`${API_BASE}${path}`, {
      method: init?.method ?? 'GET', credentials: 'include',
      headers: init?.body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: init?.body !== undefined ? JSON.stringify(init.body) : undefined,
    })
  } catch {
    throw new ApiError(0, STATUS_RU[0])
  }
  if (!r.ok) {
    let body: unknown = null
    try { body = await r.json() } catch { /* not JSON */ }
    throw new ApiError(r.status, errorText(r.status, body))
  }
  return r.json() as Promise<T>
}

export const getJob = (id: string) => request<JobResult>(`/api/v1/jobs/${encodeURIComponent(id)}`)
export const getProgress = (id: string) => request<Progress>(`/api/v1/jobs/${encodeURIComponent(id)}/progress`)
export const getDetails = (id: string) => request<Details>(`/api/v1/jobs/${encodeURIComponent(id)}/details`)
export const getCatalog = () => request<Catalog>('/api/v1/catalog')
export const getHistory = () => request<{ items: HistoryItem[] }>('/api/v1/jobs').then((r) => r.items)

type User = { username: string }
export const getMe = () => request<User>('/api/v1/auth/me')
export const login = (username: string, password: string) => request<User>('/api/v1/auth/login', { method: 'POST', body: { username, password } })
export const logout = () => request<{ ok: boolean }>('/api/v1/auth/logout', { method: 'POST' })

export type Upload = { promise: Promise<{ job_id: string }>; abort: () => void }

/** Upload with progress (fetch has no upload progress events); cancellable. */
export function uploadVideo(file: File, onProgress: (fraction: number) => void): Upload {
  const xhr = new XMLHttpRequest()
  const promise = new Promise<{ job_id: string }>((resolve, reject) => {
    xhr.open('POST', `${API_BASE}/api/v1/analyze`)
    xhr.withCredentials = true
    xhr.timeout = 15 * 60 * 1000
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total) }
    xhr.onload = () => {
      let body: { job_id?: string; detail?: unknown } | null = null
      try { body = JSON.parse(xhr.responseText) } catch { /* nginx/HTML error page */ }
      if (xhr.status >= 200 && xhr.status < 300 && body?.job_id) resolve({ job_id: body.job_id })
      else reject(new ApiError(xhr.status, errorText(xhr.status, body)))
    }
    xhr.onerror = () => reject(new ApiError(0, STATUS_RU[0]))
    xhr.ontimeout = () => reject(new ApiError(0, 'Загрузка заняла слишком много времени'))
    xhr.onabort = () => reject(new ApiError(-1, 'Загрузка отменена'))
    const form = new FormData()
    form.append('file', file)
    xhr.send(form)
  })
  return { promise, abort: () => xhr.abort() }
}
