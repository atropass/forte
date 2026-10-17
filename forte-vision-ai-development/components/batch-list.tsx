'use client'
import { AlertTriangle, CheckCircle2, ChevronRight, Clock3, FileVideo, Loader2, Trash2, XCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { BatchItem } from '@/lib/batch'

type Props = {
  items: BatchItem[]
  activeJobId: string | null
  stageLabel: (stage: string | null) => string
  onOpen: (item: BatchItem) => void
  onRemoveFinished: () => void
  onClear: () => void
}

function statusText(x: BatchItem, stageLabel: Props['stageLabel']) {
  switch (x.status) {
    case 'waiting': return 'Ожидает загрузки'
    case 'uploading': return `Загрузка ${Math.round(x.uploadPct * 100)}%`
    case 'processing': return x.queuePos ? `В очереди: ${x.queuePos}` : `${stageLabel(x.stage)} · ${Math.round(x.progress * 100)}%`
    case 'done': return x.verdict === 'violation' ? `Нарушений: ${x.nViolations ?? '?'}` : 'Нарушений не обнаружено'
    case 'failed': return x.error ?? 'Ошибка'
  }
}

function pct(x: BatchItem) {
  if (x.status === 'uploading') return x.uploadPct * 0.1
  if (x.status === 'processing') return 0.1 + 0.9 * x.progress
  return x.status === 'done' ? 1 : 0
}

function Icon({ x }: { x: BatchItem }) {
  if (x.status === 'done') return x.verdict === 'violation'
    ? <AlertTriangle className="size-4 shrink-0 text-[#a31551]" aria-label="нарушения" />
    : <CheckCircle2 className="size-4 shrink-0 text-[#196b24]" aria-label="без нарушений" />
  if (x.status === 'failed') return <XCircle className="size-4 shrink-0 text-[#a31551]" aria-label="ошибка" />
  if (x.status === 'waiting') return <Clock3 className="size-4 shrink-0 text-[#a8a5ba]" aria-label="ожидает" />
  return <Loader2 className="size-4 shrink-0 animate-spin text-[#a31551]" aria-label="в работе" />
}

export function BatchList({ items, activeJobId, stageLabel, onOpen, onRemoveFinished, onClear }: Props) {
  if (!items.length) return null
  const finished = items.filter((x) => x.status === 'done' || x.status === 'failed').length
  const busy = items.length - finished
  return (
    <div className="rounded-2xl border border-[#dedee9] bg-white p-4" aria-live="polite">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-sm font-bold">Пакетная проверка · {items.length} видео</p>
          <p className="text-xs text-[#6e6c83]">Готово {finished} из {items.length}{busy ? ' · одновременно анализируется до 3 видео' : ''}</p>
        </div>
        <div className="flex gap-2">
          {finished > 0 && busy > 0 && <Button size="sm" variant="outline" onClick={onRemoveFinished}>Убрать готовые</Button>}
          <Button size="sm" variant="ghost" onClick={onClear} aria-label="Очистить список"><Trash2 data-icon="inline-start" /> {busy ? 'Отменить все' : 'Очистить'}</Button>
        </div>
      </div>
      <ul className="flex flex-col gap-1.5">
        {items.map((x) => {
          const active = x.jobId !== null && x.jobId === activeJobId
          return (
            <li key={x.key}>
              <button
                type="button"
                disabled={!x.jobId}
                onClick={() => onOpen(x)}
                className={`flex w-full items-center gap-3 rounded-xl border px-3 py-2 text-left transition ${active ? 'border-[#a31551] bg-[#fae7ef]' : 'border-transparent hover:bg-[#f7f7fb]'} disabled:cursor-default disabled:hover:bg-transparent`}
              >
                <FileVideo className="size-4 shrink-0 text-[#a31551]" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-semibold">{x.file.name}</span>
                  <span className={`block truncate text-xs ${x.status === 'failed' ? 'text-[#a31551]' : 'text-[#6e6c83]'}`}>
                    {(x.file.size / 1024 / 1024).toFixed(1)} МБ · {statusText(x, stageLabel)}
                  </span>
                  {(x.status === 'uploading' || x.status === 'processing') && (
                    <span className="mt-1 block h-1 w-full overflow-hidden rounded-full bg-[#f0f0f7]">
                      <span className="block h-full rounded-full bg-gradient-to-r from-[#a31551] to-[#fc9385] transition-all duration-500" style={{ width: `${Math.round(pct(x) * 100)}%` }} />
                    </span>
                  )}
                </span>
                <Icon x={x} />
                {x.jobId && <ChevronRight className="size-4 shrink-0 text-[#a8a5ba]" aria-hidden />}
              </button>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
