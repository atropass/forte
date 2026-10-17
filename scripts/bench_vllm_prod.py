#!/usr/bin/env python3
"""Production Concurrency Benchmark: Simulates concurrent developers querying vLLM."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import urllib.request

PROMPT_TEXT = "Provide a concise 3-bullet summary explaining cashier security and cash drawer compliance."


async def run_single_user(user_id: int, base_url: str, api_key: str, model: str) -> dict:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a professional banking audit assistant."},
            {"role": "user", "content": f"User {user_id}: {PROMPT_TEXT}"}
        ],
        "max_tokens": 120,
        "temperature": 0.0
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "ForteVision-Bench/1.0"
        }
    )

    loop = asyncio.get_running_loop()
    t0 = time.time()
    try:
        def _fetch():
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))

        data = await loop.run_in_executor(None, _fetch)
        elapsed = time.time() - t0
        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("completion_tokens", len(content.split()))
        speed = tokens / elapsed if elapsed > 0 else 0
        return {
            "user_id": user_id,
            "status": "success",
            "elapsed_s": round(elapsed, 2),
            "tokens": tokens,
            "tokens_per_sec": round(speed, 1),
            "sample_output": content[:60] + "..."
        }
    except Exception as e:
        return {
            "user_id": user_id,
            "status": f"failed: {e}",
            "elapsed_s": round(time.time() - t0, 2),
            "tokens": 0,
            "tokens_per_sec": 0
        }


async def main():
    parser = argparse.ArgumentParser(description="Benchmark concurrent vLLM production service.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="Base URL of vLLM /v1 endpoint")
    parser.add_argument("--api-key", default="fv-prod-secret-forte-2026", help="Bearer API key")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct", help="Model ID")
    parser.add_argument("--concurrency", type=int, default=3, help="Simulated concurrent users")
    args = parser.parse_args()

    print(f"==> Launching Concurrency Benchmark against {args.base_url}")
    print(f"    Concurrent Developers: {args.concurrency}")
    print(f"    Model: {args.model}")
    print("-" * 65)

    start_total = time.time()
    tasks = [
        run_single_user(i + 1, args.base_url, args.api_key, args.model)
        for i in range(args.concurrency)
    ]
    results = await asyncio.gather(*tasks)
    total_time = round(time.time() - start_total, 2)

    total_tokens = sum(r["tokens"] for r in results)
    agg_tok_per_s = round(total_tokens / total_time, 1) if total_time > 0 else 0

    print(f"{'User':<8} {'Status':<10} {'Latency (s)':<14} {'Tokens':<8} {'Tok/s':<8}")
    print("-" * 65)
    for r in results:
        print(f"User {r['user_id']:<3} {r['status']:<10} {r['elapsed_s']:<14} {r['tokens']:<8} {r['tokens_per_sec']:<8}")

    print("-" * 65)
    print(f"Total Wall Time:          {total_time} s")
    print(f"Total Tokens Generated:   {total_tokens}")
    print(f"Aggregate Throughput:     {agg_tok_per_s} tokens/sec")
    print(f"Average Per-User Latency: {round(sum(r['elapsed_s'] for r in results)/len(results), 2)} s")


if __name__ == "__main__":
    asyncio.run(main())
