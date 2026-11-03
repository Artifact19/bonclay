.PHONY: setup dev backend frontend test verify

setup:
	./bonclay setup

dev:
	./bonclay up

backend:
	cd backend && .venv/bin/uvicorn backend:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	cd backend && .venv/bin/python -m pytest -q
	cd frontend && npm test -- --watchAll=false

verify:
	python3 sdk/verify/bonclay_verify.py $(FILE)

