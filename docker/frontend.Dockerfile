# Frontend image: the static Next.js export served by nginx (non-root, port 8080), which also proxies /api, /healthz
# and /readyz to the backend `api` service. One origin for the browser: session cookies and CSRF checks work as is.
#   docker build -f docker/frontend.Dockerfile -t forte-frontend .
FROM node:20.18-alpine AS build
WORKDIR /ui
# corepack in node 20.18 fails pnpm's signature check ("Cannot find matching keyid"), so install pnpm via npm
RUN npm install -g pnpm@10.13.1
COPY forte-vision-ai-development/package.json forte-vision-ai-development/pnpm-lock.yaml forte-vision-ai-development/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY forte-vision-ai-development/ ./
ARG NEXT_PUBLIC_MAX_UPLOAD_MB=100
ENV NEXT_TELEMETRY_DISABLED=1 NEXT_PUBLIC_MAX_UPLOAD_MB=$NEXT_PUBLIC_MAX_UPLOAD_MB
RUN pnpm typecheck && pnpm build

FROM nginxinc/nginx-unprivileged:1.27-alpine
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /ui/out /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/nginx-health || exit 1
