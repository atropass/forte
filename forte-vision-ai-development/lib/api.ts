// Client for the ForteVision backend (service/app.py). Same origin in production (static export served by
// the backend / nginx); in `pnpm dev` set NEXT_PUBLIC_API_BASE=http://localhost:8000.
// Auth: login session in an HttpOnly cookie (sent with credentials: 'include'); the X-API-Key is for API clients only.
export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? '').replace(/\/$/, '')

export type EvidenceFrame = { timestamp_sec: number; image_base64: string }
export type LegalRef = {
  document: string; issuer: string; act: string; clause: string; requirement: string
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
  job_id: string; status: string; stage: string; progress: number; created?: number; started?: number
  finished?: number; elapsed_s?: number; filename?: string; size_bytes?: number; error?: string
  queue_position: number; models_ready: boolean
}
export type CatalogItem = {
  code: string; category: string; category_ru: string; type_ru: string
  status: 'detected' | 'low_confidence' | 'untested' | 'disabled' | 'not_detected'; note?: string
  clause?: string | null; verified: boolean
}
export type Catalog = { items: CatalogItem[]; categories: Record<string, string>; models: Record<string, string> }
export type HistoryItem = {
  job_id: string; filename: string; size_bytes: number; created: number; finished?: number; elapsed_s?: number
  status: 'processing' | 'done' | 'failed'; stage?: string; error?: string
  verdict?: 'violation' | 'no_violation' | null; n_violations?: number | null
}
export type Details = {
  job: Record<string, unknown>
  debug: { timing_sec?: Record<string, number>; duration_sec?: number; events?: { kind: string; t: number }[] }
  models: Record<string, string>
}

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message) }
}

async function request<T>(path: string, init?: { method: 'POST'; body?: unknown }): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    method: init?.method ?? 'GET', credentials: 'include',
    headers: init?.body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
    body: init?.body !== undefined ? JSON.stringify(init.body) : undefined,
  })
  if (!r.ok) {
    let msg = r.statusText
    try { msg = (await r.json()).detail ?? msg } catch { /* not JSON */ }
    throw new ApiError(r.status, msg)
  }
  return r.json() as Promise<T>
}

export const getJob = (id: string) => request<JobResult>(`/api/v1/jobs/${id}`)
export const getProgress = (id: string) => request<Progress>(`/api/v1/jobs/${id}/progress`)
export const getDetails = (id: string) => request<Details>(`/api/v1/jobs/${id}/details`)
export const getCatalog = () => request<Catalog>('/api/v1/catalog')
export const getHistory = () => request<{ items: HistoryItem[] }>('/api/v1/jobs').then((r) => r.items)

type User = { username: string }
export const getMe = () => request<User>('/api/v1/auth/me')
export const login = (username: string, password: string) => request<User>('/api/v1/auth/login', { method: 'POST', body: { username, password } })
export const register = (username: string, password: string) => request<User>('/api/v1/auth/register', { method: 'POST', body: { username, password } })
export const logout = () => request<{ ok: boolean }>('/api/v1/auth/logout', { method: 'POST' })

/** Upload with progress (fetch has no upload progress events). */
export function uploadVideo(file: File, onProgress: (fraction: number) => void): Promise<{ job_id: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/api/v1/analyze`)
    xhr.withCredentials = true
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total) }
    xhr.onload = () => {
      let body: { job_id?: string; detail?: string } = {}
      try { body = JSON.parse(xhr.responseText) } catch { /* ignore */ }
      if (xhr.status >= 200 && xhr.status < 300 && body.job_id) resolve({ job_id: body.job_id })
      else reject(new ApiError(xhr.status, body.detail ?? `HTTP ${xhr.status}`))
    }
    xhr.onerror = () => reject(new ApiError(0, 'Сервис недоступен'))
    const form = new FormData()
    form.append('file', file)
    xhr.send(form)
  })
}
