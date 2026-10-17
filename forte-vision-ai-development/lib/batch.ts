'use client'
// Several videos at once from the web UI. Each file becomes a job; at most `maxActive` are uploading or being analysed
// at the same time (the server allows UI_MAX_ACTIVE_JOBS per user, default 3); the rest wait in the browser.
import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, getJob, getProgress, jobErrorText, uploadVideo, type Upload } from '@/lib/api'

export type BatchStatus = 'waiting' | 'uploading' | 'processing' | 'done' | 'failed'
export type BatchItem = {
  key: string; file: File; status: BatchStatus; uploadPct: number; jobId: string | null
  stage: string | null; progress: number; queuePos: number
  verdict: 'violation' | 'no_violation' | null; nViolations: number | null; error: string | null
}

export const BATCH_MAX_ACTIVE = Number(process.env.NEXT_PUBLIC_UI_MAX_ACTIVE ?? 3)
const POLL_MS = 2000
const RETRY_MS = 5000

let seq = 0

export function useBatch(onAuthLost: () => void, maxActive = BATCH_MAX_ACTIVE) {
  const [items, setItems] = useState<BatchItem[]>([])
  const itemsRef = useRef<BatchItem[]>([])
  const uploads = useRef(new Map<string, Upload>())
  const starting = useRef(new Set<string>())
  const retryAt = useRef(new Map<string, number>())
  const authLostRef = useRef(onAuthLost)
  authLostRef.current = onAuthLost
  itemsRef.current = items

  const patch = useCallback((key: string, p: Partial<BatchItem>) => {
    setItems((xs) => xs.map((x) => (x.key === key ? { ...x, ...p } : x)))
  }, [])

  const add = useCallback((files: File[]) => {
    const now = Date.now()
    setItems((xs) => [...xs, ...files.map((file): BatchItem => ({
      key: `${now}-${seq++}`, file, status: 'waiting', uploadPct: 0, jobId: null, stage: null, progress: 0, queuePos: 0,
      verdict: null, nViolations: null, error: null,
    }))])
  }, [])

  const clear = useCallback(() => {
    uploads.current.forEach((u) => u.abort())
    uploads.current.clear(); starting.current.clear(); retryAt.current.clear()
    setItems([])
  }, [])

  const removeFinished = useCallback(() => {
    setItems((xs) => xs.filter((x) => x.status !== 'done' && x.status !== 'failed'))
  }, [])

  const startUpload = useCallback(async (item: BatchItem) => {
    starting.current.add(item.key)
    patch(item.key, { status: 'uploading', uploadPct: 0, error: null })
    const up = uploadVideo(item.file, (f) => patch(item.key, { uploadPct: f }))
    uploads.current.set(item.key, up)
    try {
      const { job_id } = await up.promise
      patch(item.key, { status: 'processing', jobId: job_id, stage: 'queued' })
    } catch (e) {
      if (e instanceof ApiError && e.status === -1) return // cancelled (cleared)
      if (e instanceof ApiError && e.status === 401) { authLostRef.current(); return }
      if (e instanceof ApiError && e.status === 429) { // server-side cap on videos in progress: retry later
        retryAt.current.set(item.key, Date.now() + RETRY_MS)
        patch(item.key, { status: 'waiting', uploadPct: 0 })
        return
      }
      patch(item.key, { status: 'failed', error: e instanceof ApiError ? e.message : 'Ошибка загрузки' })
    } finally {
      uploads.current.delete(item.key)
      starting.current.delete(item.key)
    }
  }, [patch])

  // start waiting files while fewer than maxActive are uploading / being analysed
  useEffect(() => {
    const active = items.filter((x) => x.status === 'uploading' || x.status === 'processing').length
    let free = maxActive - active
    const now = Date.now()
    let nextRetry = Infinity
    for (const x of items) {
      if (free <= 0) break
      if (x.status !== 'waiting' || starting.current.has(x.key)) continue
      const at = retryAt.current.get(x.key) ?? 0
      if (at > now) { nextRetry = Math.min(nextRetry, at); continue }
      retryAt.current.delete(x.key)
      free--
      void startUpload(x)
    }
    if (nextRetry !== Infinity) {
      const t = window.setTimeout(() => setItems((xs) => [...xs]), nextRetry - now + 50)
      return () => window.clearTimeout(t)
    }
  }, [items, maxActive, startUpload])

  // poll every job that is being analysed
  const anyProcessing = items.some((x) => x.status === 'processing')
  useEffect(() => {
    if (!anyProcessing) return
    let stopped = false
    let running = false
    const tick = async () => {
      if (running) return // a slow round must not overlap the next one
      running = true
      try { await round() } finally { running = false }
    }
    const round = async () => {
      for (const x of itemsRef.current.filter((i) => i.status === 'processing' && i.jobId)) {
        try {
          const p = await getProgress(x.jobId as string)
          if (stopped) return
          if (p.status === 'done') {
            const j = await getJob(x.jobId as string)
            if (j.status === 'done') patch(x.key, { status: 'done', stage: 'done', progress: 1, verdict: j.verdict, nViolations: j.violations.length })
            else patch(x.key, { status: 'failed', error: jobErrorText(j.status === 'failed' ? j.error : null) })
          } else if (p.status === 'failed') {
            patch(x.key, { status: 'failed', error: jobErrorText(p.error) })
          } else {
            patch(x.key, { stage: p.stage, progress: p.progress ?? 0, queuePos: p.queue_position })
          }
        } catch (e) {
          if (e instanceof ApiError && e.status === 401) { authLostRef.current(); return }
          if (e instanceof ApiError && e.status === 404) patch(x.key, { status: 'failed', error: 'Задача не найдена' })
          // network errors: try again on the next tick
        }
      }
    }
    void tick()
    const t = window.setInterval(() => { void tick() }, POLL_MS)
    return () => { stopped = true; window.clearInterval(t) }
  }, [anyProcessing, patch])

  // warn before leaving while files are still being uploaded
  const anyUploading = items.some((x) => x.status === 'uploading' || x.status === 'waiting')
  useEffect(() => {
    if (!anyUploading) return
    const warn = (e: BeforeUnloadEvent) => { e.preventDefault() }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [anyUploading])

  useEffect(() => () => { uploads.current.forEach((u) => u.abort()) }, [])

  return { items, add, clear, removeFinished }
}
