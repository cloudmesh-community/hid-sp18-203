PROJ_NAME=e222-jackclar

run:
	python main.py
docker-build:
	docker build -t ${PROJ_NAME} .
docker-start-background:
	docker run -d -p 8080:8080 ${PROJ_NAME}
docker-start-int:
	docker run -it -p 8080:8080 ${PROJ_NAME}
docker-stop:
	docker stop $$(docker ps -a -q --filter ancestor=$(PROJ_NAME))
