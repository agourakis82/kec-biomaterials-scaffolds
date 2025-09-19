# Makefile for Darwin API — Biomaterials Bridge

.PHONY: run build docker-run actions-run actions-build actions-docker-run

run:
	cd infra/api && uvicorn main:app --reload --host 0.0.0.0 --port 8080

build:
	docker build -t darwin-kec-biomat -f infra/api/Dockerfile infra/api

docker-run:
	docker run --rm -it -p 8080:8080 --env-file infra/api/.env.example darwin-kec-biomat

actions-run:
	uvicorn server.fastapi_app:app --reload --host 0.0.0.0 --port 8000

actions-build:
	docker build -t kec-actions-api -f server/Dockerfile server

actions-docker-run:
	docker run --rm -it -p 8000:8000 -e KEC_API_KEY=$${KEC_API_KEY:-dev-12345} kec-actions-api
