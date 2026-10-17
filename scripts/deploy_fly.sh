#!/usr/bin/env bash
# Deploy the committed HEAD to Fly.io (app forte-vision, see fly.toml) by hand.
# Normally not needed: every push to main is deployed by .github/workflows/ci.yml.
# Builds from a clean `git archive`, so uncommitted files, .env, .venv and videos never reach the image.
# Secrets live on Fly only: VLM_API_BASE, VLM_API_KEY, DET_API_BASE, DET_API_KEY, METRICS_TOKEN (fly secrets set ... -a forte-vision).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FLY="$(command -v flyctl || command -v fly || echo "$HOME/.fly/bin/flyctl")"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

git -C "$ROOT" diff --quiet HEAD || echo "warning: uncommitted changes are NOT deployed (building $(git -C "$ROOT" rev-parse --short HEAD))"
git -C "$ROOT" archive HEAD | tar -x -C "$BUILD"
"$FLY" deploy "$BUILD" --config "$BUILD/fly.toml" --remote-only "$@"
