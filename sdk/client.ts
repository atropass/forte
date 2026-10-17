/**
 * Production TypeScript / Next.js client for ForteVision vLLM.
 */

export interface VLLMClientConfig {
  baseUrl?: string;
  apiKey?: string;
  model?: string;
  timeoutMs?: number;
}

export class VLLMClient {
  private baseUrl: string;
  private apiKey: string;
  private model: string;
  private timeoutMs: number;

  constructor(config: VLLMClientConfig = {}) {
    this.baseUrl = (config.baseUrl ?? "https://tw6jhre43eeunm-8000.proxy.runpod.net/v1").replace(/\/$/, "");
    this.apiKey = config.apiKey ?? "fv-prod-secret-forte-2026";
    this.model = config.model ?? "Qwen/Qwen2-VL-2B-Instruct";
    this.timeoutMs = config.timeoutMs ?? 60000;
  }

  async chat(prompt: string, systemPrompt?: string, maxTokens = 256): Promise<string> {
    const messages: Array<{ role: string; content: string }> = [];
    if (systemPrompt) {
      messages.push({ role: "system", content: systemPrompt });
    }
    messages.push({ role: "user", content: prompt });

    const res = await this.post("/chat/completions", {
      model: this.model,
      messages,
      max_tokens: maxTokens,
      temperature: 0.0,
    });

    return res.choices[0]?.message?.content ?? "";
  }

  async analyzeImage(prompt: string, imageUrl: string, maxTokens = 256): Promise<string> {
    const messages = [
      {
        role: "user",
        content: [
          { type: "text", text: prompt },
          { type: "image_url", image_url: { url: imageUrl } },
        ],
      },
    ];

    const res = await this.post("/chat/completions", {
      model: this.model,
      messages,
      max_tokens: maxTokens,
      temperature: 0.0,
    });

    return res.choices[0]?.message?.content ?? "";
  }

  async health(): Promise<boolean> {
    try {
      const rootUrl = this.baseUrl.endsWith("/v1") ? this.baseUrl.slice(0, -3) : this.baseUrl;
      const res = await fetch(`${rootUrl}/health`, { signal: AbortSignal.timeout(5000) });
      return res.ok;
    } catch {
      return false;
    }
  }

  private async post(path: string, payload: Record<string, unknown>) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`HTTP ${res.status}: ${errorText}`);
      }
      return await res.json();
    } finally {
      clearTimeout(timeout);
    }
  }
}
