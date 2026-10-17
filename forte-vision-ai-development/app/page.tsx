'use client'

import { useRef, useState } from 'react'
import { Activity, AlertTriangle, ArrowUpRight, Check, ChevronRight, Clock3, Copy, FileVideo, FolderOpen, Gauge, ImageIcon, Info, LockKeyhole, Maximize2, Play, Plus, ScanLine, ShieldCheck, Upload, Users, X } from 'lucide-react'
import { Button } from '@/components/ui/button'

type Violation = { code: string; type: string; time: number; category: string; description: string; quote: string }

const violations: Violation[] = [
  { code: 'V01', type: 'Личные вещи в кассовом узле', time: 11.4, category: 'Дисциплина рабочего места', description: 'Кассир использует смартфон на рабочем месте во время работы с наличностью. Средство связи имеет функцию фото- и видеофиксации.', quote: 'П. 12 Правил осуществления кассовых операций · Постановление НБРК № 231' },
  { code: 'V06', type: 'Приём наличности без пересчёта', time: 44.2, category: 'Приём наличности от клиента', description: 'Пачка банкнот принята через лоток и убрана в тумбу без полистного пересчёта и проверки подлинности на детекторе.', quote: 'П. 19, подп. 2 · Правила осуществления кассовых операций НБРК' },
]
const checks = [
  ['01', 'Дисциплина рабочего места', 'V01 — V04', 'done'], ['02', 'Приём наличности от клиента', 'V05 — V08', 'done'], ['03', 'Выдача наличности клиенту', 'V09', 'done'], ['04', 'Внутренний пересчёт и упаковка', 'V10 — V12', 'done'], ['05', 'Инкассация', 'V13 — V15', 'done'], ['06', 'Безопасность и устройство кассы', 'V16 — V19', 'done'],
]

function formatTime(value: number) { return `00:${value.toFixed(1).padStart(4, '0')}` }

export default function Page() {
  const inputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const [selected, setSelected] = useState(0)
  const [file, setFile] = useState<File | null>(null)
  const [activeTab, setActiveTab] = useState('events')
  const [lightbox, setLightbox] = useState(false)
  const [catalogOpen, setCatalogOpen] = useState(false)
  const [apiOpen, setApiOpen] = useState(false)

  const seek = (time: number, index: number) => { setSelected(index); if (videoRef.current) videoRef.current.currentTime = time }
  const handleFile = (next: File | undefined) => { if (next && (next.type.includes('video') || next.name.endsWith('.mov'))) setFile(next) }

  return <main className="min-h-screen bg-[#f0f0f7] text-[#1e1b39]">
    <header className="header-gradient sticky top-0 z-20 border-b border-white/10 text-white">
      <div className="mx-auto flex max-w-[1500px] items-center justify-between gap-4 px-5 py-3 lg:px-8">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2"><span className="forte-mark">F</span><span className="text-[18px] font-extrabold tracking-[-0.04em]">Forte</span></div>
          <span className="h-7 w-px bg-white/20" /><span className="text-sm font-semibold tracking-wide text-[#61cfff]">Cash Compliance AI</span>
        </div>
        <div className="hidden items-center gap-2 md:flex">
          <div className="status-pill"><span className="live-dot" /> Local AI: Qwen-VL + YOLO <span className="opacity-50">·</span> On-Premise</div>
          <Button onClick={() => setCatalogOpen(true)} variant="ghost" className="text-white hover:bg-white/10 hover:text-white"><Info data-icon="inline-start" /> НПА РК · 19 правил</Button>
          <div className="relative"><Button onClick={() => setApiOpen(!apiOpen)} variant="ghost" className="text-white hover:bg-white/10 hover:text-white"><LockKeyhole data-icon="inline-start" /> API Key</Button>{apiOpen && <div className="popover"><p className="mb-2 text-xs font-semibold text-[#6e6c83]">X-API-Key</p><div className="flex items-center gap-2"><input className="w-36 rounded-md border border-[#dedee9] bg-[#f0f0f7] px-2 py-1.5 text-xs text-[#1e1b39]" value="••••••••••••" readOnly /><Button size="sm" onClick={() => setApiOpen(false)}>Сохранить</Button></div></div>}</div>
          <Button onClick={() => { setFile(null); inputRef.current?.click() }} className="bg-[#a31551] text-white hover:bg-[#bf015d]"><Plus data-icon="inline-start" /> Новая проверка</Button>
        </div>
      </div>
    </header>

    <div className="mx-auto max-w-[1500px] px-5 py-6 lg:px-8">
      <div className="mb-6 flex items-end justify-between"><div><p className="eyebrow">FORTEVISION AI · КОНТРОЛЬ КАССОВОЙ ДИСЦИПЛИНЫ</p><h1 className="mt-1 text-2xl font-extrabold tracking-[-0.03em] md:text-3xl">Проверка видеозаписи</h1><p className="mt-1 text-sm text-[#6e6c83]">Загрузите видео с камеры наблюдения, чтобы получить интерактивный разбор.</p></div><div className="hidden text-right sm:block"><p className="text-xs text-[#6e6c83]">СЕССИЯ ПРОВЕРКИ</p><p className="font-mono text-sm font-semibold">#3F2A9C1E</p></div></div>
      <div className="grid gap-6 xl:grid-cols-[1.16fr_.84fr]">
        <section className="flex min-w-0 flex-col gap-5">
          <div className={`upload-card ${file ? 'has-file' : ''}`} onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); handleFile(e.dataTransfer.files[0]) }}>
            <input ref={inputRef} type="file" accept="video/quicktime,video/mp4,.mov,.mp4" className="hidden" onChange={(e) => handleFile(e.target.files?.[0])} />
            {!file ? <><div className="upload-icon"><Upload /></div><h2 className="mt-4 text-lg font-bold">Перетащите видео сюда</h2><p className="mt-1 text-sm text-[#6e6c83]">или выберите файл формата .MOV / .MP4 до 500 МБ</p><Button onClick={() => inputRef.current?.click()} className="mt-5 bg-[#a31551] px-5 text-white hover:bg-[#bf015d]"><FolderOpen data-icon="inline-start" /> Выбрать файл</Button></> : <div className="flex w-full items-center gap-4 text-left"><div className="flex size-12 shrink-0 items-center justify-center rounded-xl bg-[#fae7ef] text-[#a31551]"><FileVideo /></div><div className="min-w-0 flex-1"><p className="truncate text-sm font-bold">{file.name}</p><p className="mt-1 text-xs text-[#6e6c83]">{(file.size / 1024 / 1024).toFixed(1)} МБ · Готово к анализу</p></div><Button onClick={() => setFile(null)} variant="ghost" size="icon"><X /></Button></div>}
          </div>
          <div className="video-card"><div className="video-stage">{file ? <video ref={videoRef} controls className="h-full w-full object-contain" src={URL.createObjectURL(file)} /> : <div className="video-placeholder"><div className="camera-grid" /><div className="play-ring"><Play fill="currentColor" /></div><p>Превью появится после загрузки видео</p></div>}</div><div className="timeline-wrap"><div className="mb-2 flex justify-between text-[11px] font-semibold text-[#6e6c83]"><span>00:00.0</span><span>01:25.0</span></div><div className="timeline"><div className="timeline-progress" /><button aria-label="Нарушение в 11.4 секунд" onClick={() => seek(11.4, 0)} className={`timeline-pin ${selected === 0 ? 'selected' : ''}`} style={{ left: '13.4%' }} /><button aria-label="Нарушение в 44.2 секунд" onClick={() => seek(44.2, 1)} className={`timeline-pin ${selected === 1 ? 'selected' : ''}`} style={{ left: '52%' }} /></div><div className="mt-3 flex items-center gap-2 text-xs text-[#6e6c83]"><AlertTriangle className="size-3.5 text-[#a31551]" /> 2 события требуют внимания</div></div></div>
          <div><div className="mb-3 flex items-center justify-between"><h2 className="section-title"><ImageIcon /> Кадры-доказательства</h2><span className="text-xs text-[#6e6c83]">Событие {String(selected + 1).padStart(2, '0')} · {formatTime(violations[selected].time)}</span></div><div className="evidence-grid">{[0,1,2].map((i) => <button key={i} onClick={() => setLightbox(true)} className="evidence-frame"><div className="evidence-art"><ScanLine /><span className="evidence-label">CAM 04 · CASH DESK</span></div><span>{(violations[selected].time + (i - 1) * .4).toFixed(1)}с <Maximize2 className="size-3" /></span></button>)}</div></div>
          <div><div className="mb-3 flex items-center justify-between"><h2 className="section-title"><Gauge /> Матрица контроля</h2><span className="text-xs font-semibold text-[#196b24]">6 / 6 групп проверено</span></div><div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">{checks.map(([num, name, code]) => <div className="check-card" key={num}><span className="check-number">{num}</span><div className="min-w-0"><p className="truncate text-xs font-bold">{name}</p><p className="mt-1 text-[10px] text-[#6e6c83]">{code}</p></div><Check className="ml-auto size-4 shrink-0 text-[#4ea72e]" /></div>)}</div></div>
        </section>

        <aside className="flex min-w-0 flex-col gap-5"><div className="verdict-card"><div className="flex items-start gap-4"><div className="verdict-icon"><AlertTriangle /></div><div><p className="eyebrow text-white/65">РЕЗУЛЬТАТ АНАЛИЗА</p><h2 className="mt-1 text-xl font-extrabold leading-tight">Обнаружены признаки<br className="hidden sm:block" /> возможных нарушений</h2><p className="mt-3 text-sm leading-relaxed text-white/75">Выявлено потенциальных нарушений: <strong className="text-white">2</strong> · Нажмите на таймкод для просмотра фрагмента.</p></div></div><div className="mt-5 flex items-center gap-3 border-t border-white/20 pt-4 text-xs text-white/70"><Activity className="size-4" /> Анализ завершён · 01:25.0 видео</div></div>
          <div className="tabs-card"><div className="tabs-nav">{[['events', 'Обнаруженные события (2)'], ['history', 'Хронология'], ['json', 'JSON API']].map(([id, label]) => <button key={id} onClick={() => setActiveTab(id)} className={activeTab === id ? 'active' : ''}>{label}</button>)}</div>{activeTab === 'events' && <div className="flex flex-col gap-4 p-4">{violations.map((item, index) => <article key={item.code} className={`violation-card ${selected === index ? 'selected' : ''}`}><div className="flex items-start gap-3"><span className="violation-number">{String(index + 1).padStart(2, '0')}</span><div className="min-w-0 flex-1"><div className="flex flex-wrap items-center gap-2"><span className="code-badge">{item.code}</span><span className="category-label">{item.category}</span></div><h3 className="mt-2 text-[15px] font-bold leading-snug">{item.type}</h3></div><button onClick={() => seek(item.time, index)} className="timecode"><Play fill="currentColor" /> {formatTime(item.time)}</button></div><p className="mt-4 text-sm leading-relaxed text-[#474464]">{item.description}</p><div className="legal-box"><div><p className="text-[10px] font-bold uppercase tracking-wide text-[#113163]">Нормативный акт РК</p><p className="mt-1 text-xs font-semibold text-[#1e1b39]">{item.quote}</p></div><ArrowUpRight className="size-4 shrink-0 text-[#113163]" /></div><button onClick={() => seek(item.time, index)} className="mt-3 flex items-center gap-1 text-xs font-bold text-[#a31551]">Перейти к событию <ChevronRight className="size-3" /></button></article>)}</div>}{activeTab === 'history' && <div className="flex flex-col gap-0 p-5">{['Видео получено и проверено', 'Роли и зоны кассы определены', '19 правил НБРК применены', 'Отчёт сформирован'].map((text, i) => <div className="history-row" key={text}><div className="history-dot">{i < 3 ? <Check /> : <Clock3 />}</div><div><p className="text-sm font-semibold">{text}</p><p className="text-xs text-[#6e6c83]">{['10:42:01', '10:42:03', '10:42:18', '10:42:21'][i]}</p></div></div>)}</div>}{activeTab === 'json' && <div className="relative p-4"><Button size="sm" variant="outline" className="absolute right-4 top-4" onClick={() => navigator.clipboard?.writeText(JSON.stringify({ status: 'done', verdict: 'violation', violations }, null, 2))}><Copy data-icon="inline-start" /> Копировать</Button><pre className="overflow-auto rounded-xl bg-[#1e1b39] p-4 pt-14 text-xs leading-relaxed text-[#d8d6eb]">{JSON.stringify({ status: 'done', verdict: 'violation', violations }, null, 2)}</pre></div>}</div>
          <div className="note-card"><ShieldCheck className="size-5 text-[#00b4c4]" /><div><p className="text-sm font-bold">Без распознавания лиц</p><p className="mt-1 text-xs leading-relaxed text-[#6e6c83]">Система отслеживает только роли и пространственные зоны кассы.</p><div className="mt-2 flex flex-wrap gap-1.5">{['Кассир', 'Клиент', 'Контролирующий работник', 'Инкассатор'].map((role) => <span className="role-pill" key={role}><Users className="size-3" /> {role}</span>)}</div></div></div>
        </aside>
      </div>
    </div>
    {catalogOpen && <div className="modal-backdrop" onClick={() => setCatalogOpen(false)}><div className="modal" onClick={(e) => e.stopPropagation()}><div className="flex items-start justify-between"><div><p className="eyebrow">СПРАВОЧНИК КОНТРОЛЯ</p><h2 className="mt-1 text-xl font-extrabold">НПА РК · 19 правил</h2></div><Button variant="ghost" size="icon" onClick={() => setCatalogOpen(false)}><X /></Button></div><div className="mt-5 flex flex-col gap-2">{checks.map(([num, name, code]) => <div className="catalog-row" key={num}><b>{num}</b><span>{name}</span><small>{code}</small></div>)}</div></div></div>}
    {lightbox && <div className="modal-backdrop" onClick={() => setLightbox(false)}><div className="lightbox" onClick={(e) => e.stopPropagation()}><Button variant="ghost" size="icon" className="absolute right-3 top-3 text-white hover:bg-white/10 hover:text-white" onClick={() => setLightbox(false)}><X /></Button><div className="evidence-art large"><ScanLine /><span className="evidence-label">CAM 04 · EVIDENCE FRAME · {formatTime(violations[selected].time)}</span></div></div></div>}
  </main>
}
