# Makefile for Darwin API — Biomaterials Bridge

.PHONY: run build docker-run actions-run actions-build actions-docker-run

run:
	cd darwin/backend/kec_unified_api && uvicorn main:app --reload --host 0.0.0.0 --port 8080

build:
	docker build -t darwin-kec-biomat -f darwin/backend/kec_unified_api/Dockerfile darwin/backend/kec_unified_api

docker-run:
	docker run --rm -it -p 8080:8080 --env-file infra/api/.env.example darwin-kec-biomat

actions-run:
	uvicorn server.fastapi_app:app --reload --host 0.0.0.0 --port 8000

actions-build:
	docker build -t kec-actions-api -f server/Dockerfile server

actions-docker-run:
	docker run --rm -it -p 8000:8000 -e KEC_API_KEY=$${KEC_API_KEY:-dev-12345} kec-actions-api

.PHONY: compose-build compose-up compose-down compose-logs up-all tunnel-up tunnel-down tunnel-logs compose-up-gpu smoke-local smoke-public smoke-all smoke-q1 help

compose-build:
	docker compose -f docker-compose.local.yml build

compose-up:
	docker compose -f docker-compose.local.yml up -d

compose-down:
	docker compose -f docker-compose.local.yml down

compose-logs:
	docker compose -f docker-compose.local.yml logs -f --tail 200

up-all: compose-up tunnel-up

tunnel-up:
	docker compose -f docker-compose.yml --profile tunnel up -d cloudflared

tunnel-down:
	docker compose -f docker-compose.yml stop cloudflared || true

tunnel-logs:
	docker compose -f docker-compose.yml logs -f --tail 200 cloudflared

compose-up-gpu:
	docker compose -f darwin/docker-compose.yml --profile gpu up -d backend-gpu

smoke-q1:
	bash -lc 'set -e; PORT=$${HOST_BACKEND_PORT:-8090}; curl -fsS "http://localhost:$${PORT}/q1-scholar/health" && echo OK'

smoke-local:
	bash scripts/smoke.sh local

smoke-public:
	bash scripts/smoke.sh public

smoke-all:
	bash scripts/smoke.sh all

help:
	@echo "Targets:"
	@echo "  compose-build    - docker compose build"
	@echo "  compose-up       - docker compose up -d"
	@echo "  compose-up-gpu   - docker compose --profile gpu up -d backend-gpu"
	@echo "  compose-down     - docker compose down"
	@echo "  compose-logs     - docker compose logs -f --tail 200"
	@echo "  tunnel-up        - start cloudflared (compose profile)"
	@echo "  tunnel-down      - stop cloudflared container"
	@echo "  tunnel-logs      - tail cloudflared logs"
	@echo "  smoke-local      - run local health checks"
	@echo "  smoke-q1         - check /q1-scholar/health"
	@echo "  smoke-public     - run public health checks via tunnel"
	@echo "  smoke-all        - run both"
