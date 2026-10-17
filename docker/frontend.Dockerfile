# Static UI (Next.js export) served by nginx, which also proxies /api and /healthz to the backend.
FROM node:20.18-alpine AS build
WORKDIR /ui
RUN corepack enable && corepack prepare pnpm@10.13.1 --activate
COPY forte-vision-ai-development/package.json forte-vision-ai-development/pnpm-lock.yaml forte-vision-ai-development/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile || pnpm install
COPY forte-vision-ai-development/ ./
ENV NEXT_TELEMETRY_DISABLED=1
RUN pnpm build

FROM nginx:1.27-alpine
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /ui/out /usr/share/nginx/html
EXPOSE 80
