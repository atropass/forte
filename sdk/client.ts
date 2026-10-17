/**
 * TypeScript client for the ForteVision public API (Node 18+ / server-side).
 *
 *   const fv = new ForteVisionClient({ baseUrl: "https://forte.example.kz", apiKey: process.env.FV_API_KEY! });
 *   const result = await fv.analyzeAndWait(file);   // { job_id, status: "done", verdict, violations }
 *
 * No URLs or keys are built in. API keys are secrets: use this client on a server, never in a browser bundle.
 */

export interface EvidenceFrame {
  timestamp_sec: number;
  image_base64: string;
}

export interface Violation {
  type: string;
  timecode_sec: number;
  description: string;
  evidence_frames: EvidenceFrame[];
  [extra: string]: unknown;
}

export type JobResult =
  | { job_id: string; status: "processing" }
  | { job_id: string; status: "failed"; error: string }
  | { job_id: string; status: "done"; verdict: "violation" | "no_violation"; violations: Violation[] };

export interface ForteVisionClientConfig {
  baseUrl: string;
  apiKey: string;
  timeoutMs?: number;
}

export class ForteVisionClient {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly timeoutMs: number;

  constructor(config: ForteVisionClientConfig) {
    if (!config.baseUrl || !config.apiKey) throw new Error("baseUrl and apiKey are required");
    this.baseUrl = config.baseUrl.replace(/\/$/, "");
    this.apiKey = config.apiKey;
    this.timeoutMs = config.timeoutMs ?? 120_000;
  }

  async analyze(video: Blob, filename = "video.mov"): Promise<string> {
    const form = new FormData();
    form.append("file", video, filename);
    const res = await this.request("/api/v1/analyze", { method: "POST", body: form });
    return (res as { job_id: string }).job_id;
  }

  async getJob(jobId: string): Promise<JobResult> {
    return (await this.request(`/api/v1/jobs/${encodeURIComponent(jobId)}`, { method: "GET" })) as JobResult;
  }

  async wait(jobId: string, timeoutMs = 300_000, pollMs = 3_000): Promise<JobResult> {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const res = await this.getJob(jobId);
      if (res.status !== "processing") return res;
      if (Date.now() > deadline) throw new Error(`job ${jobId} still processing after ${timeoutMs} ms`);
      await new Promise((r) => setTimeout(r, pollMs));
    }
  }

  async analyzeAndWait(video: Blob, filename = "video.mov", timeoutMs = 300_000): Promise<JobResult> {
    return this.wait(await this.analyze(video, filename), timeoutMs);
  }

  private async request(path: string, init: RequestInit): Promise<unknown> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: { "X-API-Key": this.apiKey, ...(init.headers ?? {}) },
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
    return res.json();
  }
}
