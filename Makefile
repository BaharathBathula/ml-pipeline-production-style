install:
	pip install -r requirements.txt

train:
	python main.py

run-api:
	uvicorn src.api.app:app --reload

test:
	pytest tests/

docker-build:
	docker build -t ml-pipeline-prod .

docker-run:
	docker run -p 8000:8000 ml-pipeline-prod
