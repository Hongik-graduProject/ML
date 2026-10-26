.PHONY: install test validate run openapi

install:
	python -m pip install -r requirements-dev.txt

test:
	PYTHONPATH=. pytest -q

validate:
	PYTHONPATH=. python scripts/validate_artifacts.py

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

openapi:
	PYTHONPATH=. python -c "import json; from app.main import app; open('openapi.json','w',encoding='utf-8').write(json.dumps(app.openapi(),ensure_ascii=False,indent=2))"
