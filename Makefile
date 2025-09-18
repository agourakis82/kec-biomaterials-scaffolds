# Makefile for Darwin API — Biomaterials Bridge

.PHONY: run build docker-run

run:
	cd infra/api && uvicorn main:app --reload --host 0.0.0.0 --port 8080

build:
	docker build -t darwin-kec-biomat -f infra/api/Dockerfile infra/api

docker-run:
	docker run --rm -it -p 8080:8080 --env-file infra/api/.env.example darwin-kec-biomat
