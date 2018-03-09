.PHONY: help

help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help

build:
	docker build -t crepe_cnn -f Dockerfile.cpu .

run:
	docker container rm --force crepe_cnn_container && docker run -itd -p 8081:8081 --name crepe_cnn_container crepe_cnn:latest

monitor:
	docker exec -it crepe_cnn_container bash -c "tail -f /models/model.log"