'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Activity, AlertTriangle, ArrowUpRight, Check, CheckCircle2, ChevronRight, Clock3, Copy, FileVideo, FolderOpen, Gauge, History, ImageIcon, Info, Loader2, LogOut, Maximize2, Minus, Play, Plus, ScanLine, ShieldCheck, Upload, UserRound, Users, X, XCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { BatchList } from '@/components/batch-list'
import { type BatchItem, useBatch } from '@/lib/batch'
import { ApiError, MAX_UPLOAD_MB, VIDEO_EXT, type Catalog, type CatalogItem, type Details, type HistoryItem, type JobResult, type Progress, type Upload as UploadHandle, type Violation, getCatalog, getDetails, getHistory, getJob, getMe, getProgress, jobErrorText, login, logout, uploadVideo } from '@/lib/api'

type Phase = 'idle' | 'ready' | 'uploading' | 'processing' | 'done' | 'failed'
type Done = Extract<JobResult, { status: 'done' }>

// same order as the backend emits them (pipeline/analyze.py); 'started' is shown as the first analysis stage
const STAGES: [string, string][] = [
  ['queued', 'Видео получено, в очереди'], ['grounding', 'Разметка сцены: стол, сейф, дверь'], ['scene_state', 'Анализ присутствия по времени'],
  ['perception', 'Детекция людей и объектов'], ['checks', 'Проверка правил НБРК'], ['evidence', 'Подготовка кадров-доказательств'], ['done', 'Отчёт сформирован'],
]
const POLL_MS = 1500
const MAX_POLL_MS = 30000
const STATUS_RU: Record<CatalogItem['status'], string> = {
  detected: 'проверяется', low_confidence: 'низкая уверенность', untested: 'без примеров', disabled: 'отключено', not_detected: 'не проверяется',
}
const CHECKED = new Set<CatalogItem['status']>(['detected', 'low_confidence', 'untested'])

function formatTime(value: number) {
  const t = Number.isFinite(value) && value > 0 ? Math.round(value * 10) / 10 : 0
  const m = Math.floor(t / 60)
  return `${String(m).padStart(2, '0')}:${(t - m * 60).toFixed(1).padStart(4, '0')}`
}
function fmtClock(ts?: number) { return ts ? new Date(ts * 1000).toLocaleTimeString('ru-RU') : '—' }
function fmtDate(ts?: number) { return ts ? new Date(ts * 1000).toLocaleString('ru-RU', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—' }
function shortJson(res: unknown) {
  return JSON.stringify(res, (k, v) => (k === 'image_base64' && typeof v === 'string' ? `${v.slice(0, 24)}… (${v.length} символов)` : v), 2)
}

export default function Page() {
  const inputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const pollRef = useRef<number | null>(null)
  const genRef = useRef(0) // bumped on every reset: async work from an older generation must not touch state
  const uploadRef = useRef<UploadHandle | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [phase, setPhase] = useState<Phase>('idle')
  const [uploadPct, setUploadPct] = useState(0)
  const [jobId, setJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState<Progress | null>(null)
  const [result, setResult] = useState<Done | null>(null)
  const [details, setDetails] = useState<Details | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState(0)
  const [activeTab, setActiveTab] = useState('events')
  const [lightbox, setLightbox] = useState<{ src: string; t: number } | null>(null)
  const [catalogOpen, setCatalogOpen] = useState(false)
  const [user, setUser] = useState<string | null | undefined>(undefined) // undefined: checking the session
  const [historyOpen, setHistoryOpen] = useState(false)
  const [history, setHistory] = useState<HistoryItem[] | null>(null)
  const [opened, setOpened] = useState<{ name: string; size: number } | null>(null) // report opened from history (no video)
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [catalogError, setCatalogError] = useState(false)
  const [historyError, setHistoryError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [duration, setDuration] = useState(0)
  const [current, setCurrent] = useState(0)
  const batch = useBatch(() => authLost())

  const videoUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file])
  useEffect(() => () => { if (videoUrl) URL.revokeObjectURL(videoUrl) }, [videoUrl])
  useEffect(() => { getMe().then((u) => setUser(u.username)).catch(() => setUser(null)) }, [])
  useEffect(() => {
    if (!user) return
    setCatalogError(false)
    getCatalog().then(setCatalog).catch((e) => { if (e instanceof ApiError && e.status === 401) setUser(null); else setCatalogError(true) })
  }, [user])
  useEffect(() => () => { genRef.current++; if (pollRef.current) window.clearTimeout(pollRef.current); uploadRef.current?.abort() }, [])
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { setLightbox(null); setCatalogOpen(false); setHistoryOpen(false) } }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const violations: Violation[] = result?.violations ?? []
  const byCode = useMemo(() => Object.fromEntries((catalog?.items ?? []).map((i) => [i.code, i])), [catalog])
  const groups = useMemo(() => {
    const g: { key: string; name: string; codes: CatalogItem[] }[] = []
    for (const it of catalog?.items ?? []) {
      let row = g.find((x) => x.key === it.category)
      if (!row) { row = { key: it.category, name: it.category_ru ?? it.category, codes: [] }; g.push(row) }
      row.codes.push(it)
    }
    return g
  }, [catalog])
  const models = catalog?.models ?? details?.models ?? {}
  const modelLabel = models.vlm ? `${models.vlm.split('/').pop()?.replace(/-Instruct.*/, '')} + RT-DETRv2` : 'Qwen3-VL + RT-DETRv2'
  const videoDuration = duration || details?.debug.duration_sec || 0

  const reset = () => {
    genRef.current++
    if (pollRef.current) window.clearTimeout(pollRef.current)
    pollRef.current = null
    uploadRef.current?.abort(); uploadRef.current = null
    setPhase('idle'); setFile(null); setJobId(null); setProgress(null); setResult(null); setDetails(null); setError(null)
    setSelected(0); setUploadPct(0); setDuration(0); setCurrent(0); setOpened(null); setNotice(null)
  }
  const authLost = () => { reset(); batch.clear(); setCatalog(null); setUser(null) }
  const fileProblem = (f: File) => !VIDEO_EXT.test(f.name) ? 'не .MOV/.MP4' : f.size > MAX_UPLOAD_MB * 1024 * 1024 ? `больше ${MAX_UPLOAD_MB} МБ` : f.size === 0 ? 'пустой файл' : null
  const handleFiles = (list: FileList | null | undefined) => {
    const files = Array.from(list ?? [])
    if (!files.length) return
    if (files.length === 1) { if (!busy) handleFile(files[0]); return }
    const ok = files.filter((f) => !fileProblem(f))
    const bad = files.filter((f) => fileProblem(f)).map((f) => `${f.name} (${fileProblem(f)})`)
    setError(null)
    setNotice(bad.length ? `Пропущены: ${bad.join(', ')}` : null)
    if (ok.length) batch.add(ok)
  }
  const openBatch = (item: BatchItem) => {
    if (!item.jobId) return
    reset(); setFile(item.file); setJobId(item.jobId)
    if (item.status === 'failed') { setError(item.error ?? 'Ошибка анализа'); setPhase('failed'); return }
    setPhase('processing'); poll(item.jobId) // done jobs resolve on the first tick
  }
  const stageLabel = (stage: string | null) => STAGES.find(([k]) => k === (stage === 'started' ? 'grounding' : stage))?.[1] ?? 'Анализ'
  const handleFile = (next: File | undefined) => {
    if (!next) return
    if (!VIDEO_EXT.test(next.name)) { setError('Нужен видеофайл .MOV или .MP4'); return }
    if (next.size > MAX_UPLOAD_MB * 1024 * 1024) { setError(`Файл больше ${MAX_UPLOAD_MB} МБ`); return }
    if (next.size === 0) { setError('Файл пустой'); return }
    reset(); setFile(next); setPhase('ready')
  }

  const poll = useCallback((id: string) => {
    const gen = genRef.current
    let failures = 0
    const live = () => gen === genRef.current
    const tick = async () => {
      let delay = POLL_MS
      try {
        const p = await getProgress(id)
        if (!live()) return
        setProgress(p); setNotice(null); failures = 0
        if (p.status === 'done') {
          const j = await getJob(id)
          if (!live()) return
          if (j.status === 'done') {
            setResult(j); setPhase('done'); setSelected(0)
            getDetails(id).then((d) => { if (live()) setDetails(d) }).catch(() => undefined)
          } else {
            setError(jobErrorText(j.status === 'failed' ? j.error : null)); setPhase('failed')
          }
          return
        }
        if (p.status === 'failed') { setError(jobErrorText(p.error)); setPhase('failed'); return }
      } catch (e) {
        if (!live()) return
        if (e instanceof ApiError && e.status === 401) { authLost(); return }
        if (e instanceof ApiError && e.status === 404) { setError('Задача не найдена'); setPhase('failed'); return }
        failures++
        delay = Math.min(POLL_MS * 2 ** failures, MAX_POLL_MS)
        if (failures >= 3) setNotice('Нет связи с сервером, повторяем…')
      }
      if (live()) pollRef.current = window.setTimeout(tick, delay)
    }
    tick()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const start = async () => {
    if (!file) return
    const gen = genRef.current
    setError(null); setPhase('uploading'); setUploadPct(0)
    const up = uploadVideo(file, (f) => { if (gen === genRef.current) setUploadPct(f) })
    uploadRef.current = up
    try {
      const { job_id } = await up.promise
      if (gen !== genRef.current) return
      uploadRef.current = null
      setJobId(job_id); setPhase('processing'); poll(job_id)
    } catch (e) {
      if (gen !== genRef.current || (e instanceof ApiError && e.status === -1)) return
      uploadRef.current = null
      if (e instanceof ApiError && e.status === 401) { authLost(); return }
      setPhase('failed')
      setError(e instanceof ApiError ? e.message : 'Ошибка загрузки')
    }
  }
  const showHistory = () => {
    setHistoryOpen(true); setHistory(null); setHistoryError(null)
    getHistory().then(setHistory).catch((e) => {
      if (e instanceof ApiError && e.status === 401) { setHistoryOpen(false); authLost(); return }
      setHistoryError(e instanceof ApiError ? e.message : 'Не удалось загрузить историю')
    })
  }
  const openJob = (item: HistoryItem) => {
    reset(); setHistoryOpen(false)
    setOpened({ name: item.filename ?? 'video', size: item.size_bytes ?? 0 }); setJobId(item.job_id)
    if (item.status === 'failed') { setError(jobErrorText(item.error)); setPhase('failed'); return }
    setPhase('processing'); poll(item.job_id) // done jobs resolve on the first tick
  }
  const signOut = () => { logout().catch(() => undefined); authLost() }

  const seek = (time: number, index?: number) => {
    if (index !== undefined) setSelected(index)
    const v = videoRef.current
    if (v) { v.currentTime = time; v.pause() }
  }
  const busy = phase === 'uploading' || phase === 'processing'
  const json = useMemo(() => (result ? shortJson(result) : ''), [result])
  useEffect(() => {
    if (phase !== 'uploading') return
    const warn = (e: BeforeUnloadEvent) => { e.preventDefault() }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [phase])
  const sel = violations[Math.min(selected, Math.max(violations.length - 1, 0))]
  const stageIdx = Math.max(0, STAGES.findIndex(([s]) => s === (progress?.stage === 'started' ? 'grounding' : progress?.stage)))
  const pct = phase === 'uploading' ? uploadPct * 0.1 : phase === 'processing' ? 0.1 + 0.9 * (progress?.progress ?? 0) : phase === 'done' ? 1 : 0
  const shown = file ? { name: file.name, size: file.size } : opened
  const copyJson = () => {
    if (!result) return
    const text = JSON.stringify(result, null, 2)
    if (navigator.clipboard?.writeText) navigator.clipboard.writeText(text).catch(() => setNotice('Не удалось скопировать: браузер запретил доступ к буферу обмена'))
    else setNotice('Копирование доступно только по HTTPS')
  }

  if (user === undefined) return <main className="flex min-h-screen items-center justify-center bg-[#f0f0f7]"><Loader2 className="size-6 animate-spin text-[#a31551]" /></main>
  if (user === null) return <AuthScreen onAuth={setUser} />

  return <main className="min-h-screen bg-[#f0f0f7] text-[#1e1b39]">
    <header className="header-gradient sticky top-0 z-20 border-b border-white/10 text-white">
      <div className="mx-auto flex max-w-[1500px] items-center justify-between gap-4 px-5 py-3 lg:px-8">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2"><span className="forte-mark">F</span><span className="text-[18px] font-extrabold tracking-[-0.04em]">Forte</span></div>
          <span className="hidden h-7 w-px bg-white/20 sm:block" /><span className="hidden text-sm font-semibold tracking-wide text-[#61cfff] sm:block">Cash Compliance AI</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="hidden lg:block"><div className="status-pill"><span className="live-dot" /> Локальный AI: {modelLabel} <span className="opacity-50">·</span> On-Premise</div></div>
          <Button onClick={() => setCatalogOpen(true)} variant="ghost" className="hidden text-white hover:bg-white/10 hover:text-white sm:inline-flex"><Info data-icon="inline-start" /> НПА РК · справочник</Button>
          <Button onClick={showHistory} variant="ghost" className="text-white hover:bg-white/10 hover:text-white"><History data-icon="inline-start" /> <span className="hidden sm:inline">История</span></Button>
          <span className="hidden items-center gap-1.5 px-2 text-sm font-semibold text-white/80 md:flex"><UserRound className="size-4" /> {user}</span>
          <Button onClick={signOut} variant="ghost" size="icon" aria-label="Выйти" title="Выйти" className="text-white hover:bg-white/10 hover:text-white"><LogOut /></Button>
          <Button onClick={() => { if (busy && !window.confirm('Текущая проверка будет прервана в браузере. Продолжить?')) return; reset(); inputRef.current?.click() }} className="bg-[#a31551] text-white hover:bg-[#bf015d]"><Plus data-icon="inline-start" /> <span className="hidden sm:inline">Новая проверка</span></Button>
        </div>
      </div>
    </header>

    <div className="mx-auto max-w-[1500px] px-5 py-6 lg:px-8">
      <div className="mb-6 flex items-end justify-between"><div><p className="eyebrow">FORTEVISION AI · КОНТРОЛЬ КАССОВОЙ ДИСЦИПЛИНЫ</p><h1 className="mt-1 text-2xl font-extrabold tracking-[-0.03em] md:text-3xl">Проверка видеозаписи</h1><p className="mt-1 text-sm text-[#6e6c83]">Загрузите видео с камеры наблюдения, чтобы получить интерактивный разбор.</p></div><div className="hidden text-right sm:block"><p className="text-xs text-[#6e6c83]">СЕССИЯ ПРОВЕРКИ</p><p className="font-mono text-sm font-semibold">{jobId ? `#${jobId.toUpperCase()}` : '—'}</p></div></div>
      <div className="grid gap-6 xl:grid-cols-[1.16fr_.84fr]">
        <section className="flex min-w-0 flex-col gap-5">
          <div className={`upload-card ${file ? 'has-file' : ''}`} onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); handleFiles(e.dataTransfer.files) }}>
            <input ref={inputRef} type="file" accept="video/quicktime,video/mp4,.mov,.mp4,.m4v" multiple className="hidden" onChange={(e) => { handleFiles(e.target.files); e.target.value = '' }} />
            {!shown ? <><div className="upload-icon"><Upload /></div><h2 className="mt-4 text-lg font-bold">Перетащите видео сюда</h2><p className="mt-1 text-sm text-[#6e6c83]">или выберите один или несколько файлов .MOV / .MP4 до {MAX_UPLOAD_MB} МБ</p><Button onClick={() => inputRef.current?.click()} className="mt-5 bg-[#a31551] px-5 text-white hover:bg-[#bf015d]"><FolderOpen data-icon="inline-start" /> Выбрать файлы</Button></> : <div className="flex w-full flex-col gap-3"><div className="flex w-full items-center gap-4 text-left"><div className="flex size-12 shrink-0 items-center justify-center rounded-xl bg-[#fae7ef] text-[#a31551]"><FileVideo /></div><div className="min-w-0 flex-1"><p className="truncate text-sm font-bold">{shown.name}</p><p className="mt-1 text-xs text-[#6e6c83]">{(shown.size / 1024 / 1024).toFixed(1)} МБ · {opened && phase === 'done' ? 'Отчёт из истории' : phase === 'ready' ? 'Готово к анализу' : phase === 'uploading' ? `Загрузка ${Math.round(uploadPct * 100)}%` : phase === 'processing' ? (progress?.queue_position ? `В очереди: ${progress.queue_position}` : STAGES[stageIdx][1]) : phase === 'done' ? 'Анализ завершён' : 'Ошибка'}</p></div>{phase === 'ready' && <Button onClick={start} className="bg-[#a31551] text-white hover:bg-[#bf015d]"><ScanLine data-icon="inline-start" /> Запустить анализ</Button>}{busy && <Loader2 className="size-5 animate-spin text-[#a31551]" />}{phase === 'uploading' && <Button onClick={reset} variant="outline" size="sm">Отменить</Button>}{!busy && <Button onClick={reset} variant="ghost" size="icon" aria-label="Убрать файл"><X /></Button>}</div>{(busy || phase === 'done') && <div className="h-1.5 w-full overflow-hidden rounded-full bg-white"><div className="h-full rounded-full bg-gradient-to-r from-[#a31551] to-[#fc9385] transition-all duration-500" style={{ width: `${Math.round(pct * 100)}%` }} /></div>}</div>}
            {error && <p role="alert" className="mt-3 flex items-center gap-1.5 text-xs font-semibold text-[#a31551]"><XCircle className="size-3.5" /> {error}</p>}
            {notice && <p role="status" className="mt-3 flex items-center gap-1.5 text-xs font-semibold text-[#6e6c83]"><Info className="size-3.5" /> {notice}</p>}
          </div>
          <BatchList items={batch.items} activeJobId={jobId} stageLabel={stageLabel} onOpen={openBatch} onRemoveFinished={batch.removeFinished} onClear={batch.clear} />
          <div className="video-card"><div className="video-stage">{videoUrl ? <video ref={videoRef} controls playsInline className="h-full w-full object-contain" src={videoUrl} onLoadedMetadata={(e) => setDuration(Number.isFinite(e.currentTarget.duration) ? e.currentTarget.duration : 0)} onTimeUpdate={(e) => setCurrent(e.currentTarget.currentTime)} /> : <div className="video-placeholder"><div className="camera-grid" /><div className="play-ring">{opened ? <FileVideo /> : <Play fill="currentColor" />}</div><p>{opened ? `${opened.name} · видео не хранится на сервере, только отчёт` : 'Превью появится после загрузки видео'}</p></div>}</div>
            <div className="timeline-wrap"><div className="mb-2 flex justify-between text-[11px] font-semibold text-[#6e6c83]"><span>{formatTime(current)}</span><span>{formatTime(videoDuration)}</span></div><div className="timeline"><div className="timeline-progress" style={{ width: videoDuration ? `${Math.min(100, (current / videoDuration) * 100)}%` : '0%' }} />{videoDuration > 0 && violations.map((v, i) => <button key={i} aria-label={`Нарушение на ${v.timecode_sec} с`} title={`${v.type_code ?? ''} · ${formatTime(v.timecode_sec)}`} onClick={() => seek(v.timecode_sec, i)} className={`timeline-pin ${selected === i ? 'selected' : ''}`} style={{ left: `${Math.min(100, (v.timecode_sec / videoDuration) * 100)}%` }} />)}</div><div className="mt-3 flex items-center gap-2 text-xs text-[#6e6c83]">{phase === 'done' ? (violations.length ? <><AlertTriangle className="size-3.5 text-[#a31551]" /> {violations.length} {violations.length === 1 ? 'событие требует' : 'события требуют'} внимания · нажмите на метку</> : <><CheckCircle2 className="size-3.5 text-[#196b24]" /> Событий не обнаружено</>) : <><Clock3 className="size-3.5" /> Метки событий появятся после анализа</>}</div></div></div>
          <div><div className="mb-3 flex items-center justify-between"><h2 className="section-title"><ImageIcon /> Кадры-доказательства</h2><span className="text-xs text-[#6e6c83]">{sel ? `Событие ${String(selected + 1).padStart(2, '0')} · ${formatTime(sel.timecode_sec)}` : 'нет событий'}</span></div>
            {sel ? <div className="evidence-grid">{sel.evidence_frames.map((f, i) => { const src = `data:image/jpeg;base64,${f.image_base64}`; return <button key={i} onClick={() => { setLightbox({ src, t: f.timestamp_sec }); seek(f.timestamp_sec) }} className="evidence-frame"><div className="evidence-photo"><img src={src} alt={`Кадр ${f.timestamp_sec} с`} /></div><span>{f.timestamp_sec.toFixed(1)}с <Maximize2 className="size-3" /></span></button> })}</div> : <div className="evidence-grid">{[0, 1, 2].map((i) => <div key={i} className="evidence-frame"><div className="evidence-art"><ScanLine /><span className="evidence-label">ОЖИДАНИЕ АНАЛИЗА</span></div><span>—</span></div>)}</div>}</div>
          <div><div className="mb-3 flex items-center justify-between"><h2 className="section-title"><Gauge /> Матрица контроля</h2><span className="text-xs font-semibold text-[#196b24]">{catalog ? `${catalog.items.filter((i) => CHECKED.has(i.status)).length} из ${catalog.items.length} типов проверяется` : catalogError ? 'справочник недоступен' : 'загрузка…'}</span></div><div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">{groups.map((g, gi) => { const on = g.codes.filter((c) => CHECKED.has(c.status)); return <button onClick={() => setCatalogOpen(true)} className="check-card text-left" key={g.key}><span className="check-number">{String(gi + 1).padStart(2, '0')}</span><div className="min-w-0"><p className="truncate text-xs font-bold">{g.name}</p><p className="mt-1 text-[10px] text-[#6e6c83]">{g.codes.map((c) => c.code).join(' · ')} · {on.length}/{g.codes.length}</p></div>{on.length ? <Check className="ml-auto size-4 shrink-0 text-[#4ea72e]" /> : <Minus className="ml-auto size-4 shrink-0 text-[#a8a5ba]" />}</button> })}</div></div>
        </section>

        <aside className="flex min-w-0 flex-col gap-5">
          <Verdict phase={phase} result={result} progress={progress} stageLabel={STAGES[stageIdx][1]} error={error} duration={videoDuration} elapsed={progress?.elapsed_s ?? undefined} />
          <div className="tabs-card"><div className="tabs-nav">{[['events', `Обнаруженные события (${violations.length})`], ['history', 'Хронология'], ['json', 'JSON API']].map(([id, label]) => <button key={id} onClick={() => setActiveTab(id)} className={activeTab === id ? 'active' : ''}>{label}</button>)}</div>
            {activeTab === 'events' && <div className="flex flex-col gap-4 p-4">{phase !== 'done' && <p className="py-6 text-center text-sm text-[#6e6c83]">{busy ? 'Идёт анализ…' : 'Результаты появятся после анализа.'}</p>}{phase === 'done' && !violations.length && <p className="py-6 text-center text-sm text-[#196b24]">Признаков нарушений не обнаружено.</p>}{violations.map((item, index) => <article key={index} className={`violation-card ${selected === index ? 'selected' : ''}`} onClick={() => setSelected(index)}><div className="flex items-start gap-3"><span className="violation-number">{String(index + 1).padStart(2, '0')}</span><div className="min-w-0 flex-1"><div className="flex flex-wrap items-center gap-2">{item.type_code && <span className="code-badge">{item.type_code}</span>}<span className="category-label">{byCode[item.type_code ?? '']?.category_ru ?? ''}</span>{item.confidence !== undefined && item.confidence < 0.67 && <span className="category-label">· низкая уверенность</span>}</div><h3 className="mt-2 text-[15px] font-bold leading-snug">{item.type}</h3>{item.subtype && <p className="mt-1 text-xs font-semibold text-[#a31551]">{item.subtype}</p>}</div><button onClick={(e) => { e.stopPropagation(); seek(item.timecode_sec, index) }} className="timecode"><Play fill="currentColor" /> {formatTime(item.timecode_sec)}</button></div><p className="mt-4 text-sm leading-relaxed text-[#474464]">{item.description}</p>{item.rule && <a href={item.rule.url} target="_blank" rel="noreferrer" className="legal-box" onClick={(e) => e.stopPropagation()}><div><p className="text-[10px] font-bold uppercase tracking-wide text-[#113163]">Нормативный акт РК {item.rule.verified ? '· пункт сверен с текстом' : '· трактовка'}</p><p className="mt-1 text-xs font-semibold text-[#1e1b39]">{item.rule.clause} · {item.rule.document.split(' в банках')[0]} ({item.rule.act})</p><p className="mt-1 text-[11px] leading-relaxed text-[#474464]">{item.rule.requirement}</p>{item.rule.interpretation && <p className="mt-1 text-[11px] italic text-[#6e6c83]">{item.rule.interpretation}</p>}</div><ArrowUpRight className="size-4 shrink-0 text-[#113163]" /></a>}<button onClick={(e) => { e.stopPropagation(); seek(item.timecode_sec, index) }} className="mt-3 flex items-center gap-1 text-xs font-bold text-[#a31551]">Перейти к событию <ChevronRight className="size-3" /></button></article>)}</div>}
            {activeTab === 'history' && <div className="flex flex-col gap-0 p-5">{STAGES.map(([key, text], i) => { const reached = phase === 'done' || (busy && i < stageIdx); const now = busy && i === stageIdx; const t = (key === 'queued' ? progress?.created : key === 'done' ? progress?.finished : undefined) ?? undefined; const timing = details?.debug.timing_sec; const dur = key === 'perception' ? timing?.perception : key === 'scene_state' ? timing?.scene_state : key === 'checks' ? timing?.checks : undefined; return <div className="history-row" key={key}><div className="history-dot">{reached ? <Check /> : now ? <Loader2 className="animate-spin" /> : <Clock3 />}</div><div><p className={`text-sm font-semibold ${reached || now ? '' : 'text-[#a8a5ba]'}`}>{text}</p><p className="text-xs text-[#6e6c83]">{t ? fmtClock(t) : ''}{dur !== undefined ? ` ${dur} с` : ''}{key === 'done' && progress?.elapsed_s ? ` · всего ${progress.elapsed_s} с` : ''}</p></div></div> })}</div>}
            {activeTab === 'json' && <div className="relative p-4">{result ? <><Button size="sm" variant="outline" className="absolute right-4 top-4" onClick={copyJson}><Copy data-icon="inline-start" /> Копировать</Button><pre className="max-h-[560px] overflow-auto rounded-xl bg-[#1e1b39] p-4 pt-14 text-xs leading-relaxed text-[#d8d6eb]">{json}</pre></> : <p className="py-6 text-center text-sm text-[#6e6c83]">Ответ API появится после анализа. GET /api/v1/jobs/{'{job_id}'}</p>}</div>}
          </div>
          <div className="note-card"><ShieldCheck className="size-5 text-[#00b4c4]" /><div><p className="text-sm font-bold">Без распознавания лиц</p><p className="mt-1 text-xs leading-relaxed text-[#6e6c83]">Система отслеживает только роли и пространственные зоны кассы. Видео удаляется сразу после анализа — в истории хранится только отчёт.</p><div className="mt-2 flex flex-wrap gap-1.5">{['Кассир', 'Клиент', 'Инкассатор'].map((role) => <span className="role-pill" key={role}><Users className="size-3" /> {role}</span>)}</div></div></div>
        </aside>
      </div>
    </div>
    {catalogOpen && <div className="modal-backdrop" onClick={() => setCatalogOpen(false)}><div className="modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}><div className="flex items-start justify-between"><div><p className="eyebrow">СПРАВОЧНИК КОНТРОЛЯ</p><h2 className="mt-1 text-xl font-extrabold">НПА РК · типы нарушений</h2><p className="mt-1 text-xs text-[#6e6c83]">Постановления Правления НБРК № 231 (29.11.2019) и № 14 (24.02.2020).</p></div><Button variant="ghost" size="icon" aria-label="Закрыть" onClick={() => setCatalogOpen(false)}><X /></Button></div><div className="mt-5 flex flex-col gap-1">{catalog ? catalog.items.map((it) => <div className="catalog-row" key={it.code}><b>{it.code}</b><span><span className="block">{it.type_ru}</span><small className="block">{it.clause ? `${it.clause} · ` : ''}{it.note}</small></span><small className={`status-tag ${it.status}`}>{STATUS_RU[it.status]}</small></div>) : <p className="text-sm text-[#6e6c83]">{catalogError ? 'Справочник недоступен, попробуйте обновить страницу.' : 'Загрузка справочника…'}</p>}</div></div></div>}
    {historyOpen && <div className="modal-backdrop" onClick={() => setHistoryOpen(false)}><div className="modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}><div className="flex items-start justify-between"><div><p className="eyebrow">ИСТОРИЯ ПРОВЕРОК</p><h2 className="mt-1 text-xl font-extrabold">Мои видео</h2><p className="mt-1 text-xs text-[#6e6c83]">Сохраняются отчёты и объяснения; сами видео удаляются после анализа.</p></div><Button variant="ghost" size="icon" aria-label="Закрыть" onClick={() => setHistoryOpen(false)}><X /></Button></div>
      <div className="mt-5 flex flex-col gap-1">{historyError ? <p role="alert" className="text-sm text-[#a31551]">{historyError}</p> : history === null ? <p className="text-sm text-[#6e6c83]">Загрузка…</p> : !history.length ? <p className="text-sm text-[#6e6c83]">Вы ещё не загружали видео.</p> : history.map((h) => <button key={h.job_id} onClick={() => openJob(h)} className="catalog-row w-full text-left hover:bg-[#f7f7fb]"><FileVideo className="size-5 text-[#a31551]" /><span className="min-w-0"><span className="block truncate font-semibold">{h.filename ?? 'video'}</span><small className="block">{fmtDate(h.created ?? undefined)} · #{h.job_id.toUpperCase()}{h.elapsed_s ? ` · ${h.elapsed_s} с` : ''}</small></span><small className={`status-tag ${h.status === 'failed' ? 'not_detected' : h.status === 'processing' ? 'untested' : h.verdict === 'violation' ? 'low_confidence' : 'detected'}`}>{h.status === 'failed' ? 'ошибка' : h.status === 'processing' ? 'в работе' : h.verdict === 'violation' ? `нарушений: ${h.n_violations ?? '?'}` : 'без нарушений'}</small></button>)}</div></div></div>}
    {lightbox && <div className="modal-backdrop" onClick={() => setLightbox(null)}><div className="lightbox" onClick={(e) => e.stopPropagation()}><Button variant="ghost" size="icon" className="absolute right-3 top-3 z-10 bg-black/40 text-white hover:bg-white/10 hover:text-white" aria-label="Закрыть" onClick={() => setLightbox(null)}><X /></Button><img src={lightbox.src} alt="Кадр-доказательство" className="max-h-[85vh] max-w-[90vw] rounded-xl" /><span className="evidence-label">КАДР-ДОКАЗАТЕЛЬСТВО · {formatTime(lightbox.t)}</span></div></div>}
  </main>
}

function Verdict({ phase, result, progress, stageLabel, error, duration, elapsed }: { phase: Phase; result: Done | null; progress: Progress | null; stageLabel: string; error: string | null; duration: number; elapsed?: number }) {
  if (phase === 'done' && result) {
    const v = result.verdict === 'violation'
    return <div className={`verdict-card ${v ? '' : 'ok'}`}><div className="flex items-start gap-4"><div className="verdict-icon">{v ? <AlertTriangle /> : <CheckCircle2 />}</div><div><p className="eyebrow text-white/65">РЕЗУЛЬТАТ АНАЛИЗА</p><h2 className="mt-1 text-xl font-extrabold leading-tight">{v ? <>Обнаружены признаки<br className="hidden sm:block" /> возможных нарушений</> : 'Нарушение не обнаружено'}</h2><p className="mt-3 text-sm leading-relaxed text-white/75">{v ? <>Выявлено потенциальных нарушений: <strong className="text-white">{result.violations.length}</strong> · Нажмите на таймкод для просмотра фрагмента.</> : 'Видимых признаков нарушений кассовой дисциплины не найдено.'}</p></div></div><div className="mt-5 flex items-center gap-3 border-t border-white/20 pt-4 text-xs text-white/70"><Activity className="size-4" /> Анализ завершён · {formatTime(duration)} видео{elapsed ? ` · ${elapsed} с обработки` : ''} · результат — подсказка для проверяющего</div></div>
  }
  if (phase === 'uploading' || phase === 'processing') {
    return <div className="verdict-card pending"><div className="flex items-start gap-4"><div className="verdict-icon"><Loader2 className="animate-spin" /></div><div><p className="eyebrow text-white/65">ИДЁТ АНАЛИЗ</p><h2 className="mt-1 text-xl font-extrabold leading-tight">{phase === 'uploading' ? 'Загрузка видео' : stageLabel}</h2><p className="mt-3 text-sm leading-relaxed text-white/75">{progress && !progress.models_ready ? 'GPU-сервис анализа недоступен или перезапускается — задача ждёт в очереди.' : progress?.queue_position ? `Позиция в очереди: ${progress.queue_position}` : 'Обработка занимает до нескольких минут.'}</p></div></div></div>
  }
  if (phase === 'failed') {
    return <div className="verdict-card failed"><div className="flex items-start gap-4"><div className="verdict-icon"><XCircle /></div><div><p className="eyebrow text-white/65">ОШИБКА</p><h2 className="mt-1 text-xl font-extrabold leading-tight">Анализ не выполнен</h2><p className="mt-3 text-sm leading-relaxed text-white/75">{error ?? 'Неизвестная ошибка'}</p></div></div></div>
  }
  return <div className="verdict-card pending"><div className="flex items-start gap-4"><div className="verdict-icon"><FileVideo /></div><div><p className="eyebrow text-white/65">РЕЗУЛЬТАТ АНАЛИЗА</p><h2 className="mt-1 text-xl font-extrabold leading-tight">Ожидание видео</h2><p className="mt-3 text-sm leading-relaxed text-white/75">Загрузите запись и нажмите «Запустить анализ».</p></div></div></div>
}

function AuthScreen({ onAuth }: { onAuth: (username: string) => void }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setError(null); setBusy(true)
    try {
      const u = await login(username.trim(), password)
      onAuth(u.username)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Сервис недоступен')
    } finally { setBusy(false) }
  }
  const input = 'w-full rounded-lg border border-[#dedee9] bg-[#f0f0f7] px-3 py-2.5 text-sm text-[#1e1b39] outline-none focus:border-[#a31551]'
  return <main className="flex min-h-screen flex-col bg-[#f0f0f7] text-[#1e1b39]">
    <header className="header-gradient text-white"><div className="mx-auto flex max-w-[1500px] items-center gap-3 px-5 py-3 lg:px-8"><div className="flex items-center gap-2"><span className="forte-mark">F</span><span className="text-[18px] font-extrabold tracking-[-0.04em]">Forte</span></div><span className="hidden h-7 w-px bg-white/20 sm:block" /><span className="hidden text-sm font-semibold tracking-wide text-[#61cfff] sm:block">Cash Compliance AI</span></div></header>
    <div className="flex flex-1 items-center justify-center px-4 py-10">
      <form onSubmit={submit} className="w-full max-w-[380px] rounded-2xl border border-[#dedee9] bg-white p-6 shadow-[0_10px_30px_rgba(30,27,57,.08)]">
        <p className="eyebrow">FORTEVISION AI</p>
        <h1 className="mt-1 text-2xl font-extrabold tracking-[-0.03em]">Вход</h1>
        <p className="mt-1 text-sm text-[#6e6c83]">Войдите, чтобы загружать видео и видеть свою историю проверок. Учётные записи выдаёт администратор.</p>
        <label className="mt-5 block text-xs font-semibold text-[#6e6c83]">Логин<input className={`mt-1 ${input}`} autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} required minLength={3} maxLength={64} /></label>
        <label className="mt-3 block text-xs font-semibold text-[#6e6c83]">Пароль<input className={`mt-1 ${input}`} type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} required maxLength={256} /></label>
        {error && <p role="alert" className="mt-3 flex items-center gap-1.5 text-xs font-semibold text-[#a31551]"><XCircle className="size-3.5" /> {error}</p>}
        <Button type="submit" disabled={busy} className="mt-5 w-full bg-[#a31551] text-white hover:bg-[#bf015d]">{busy && <Loader2 className="animate-spin" data-icon="inline-start" />}Войти</Button>
      </form>
    </div>
  </main>
}
