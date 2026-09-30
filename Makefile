include .env-example
-include .env

COMPOSE := docker compose

.DEFAULT_GOAL := help
.PHONY: help build up down logs setup dev eval check-devcontainer check-not-devcontainer





check-devcontainer:
ifndef IN_DEVCONTAINER
	@echo "Error: this target must run inside the devcontainer"
	@exit 1
endif

check-not-devcontainer:
ifdef IN_DEVCONTAINER
	@echo "Error: run this on the host, not inside the devcontainer"
	@exit 1
endif





help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n%s\n", substr($$0, 5)} /^[a-z]+:.*## / {printf "  %-8s %s\n", $$1, $$2}' $(firstword $(MAKEFILE_LIST))

setup: check-not-devcontainer ## Pull the Ollama models and create .env if missing
	@test -f .env || cp .env-example .env
	@ollama show $(OLLAMA_CHAT_MODEL) >/dev/null 2>&1 || ollama pull $(OLLAMA_CHAT_MODEL)
	@ollama show $(OLLAMA_EMBEDDINGS_MODEL) >/dev/null 2>&1 || ollama pull $(OLLAMA_EMBEDDINGS_MODEL)
	@echo "everything is ready"



build: check-not-devcontainer ## Build the app image
	$(COMPOSE) build

up:	check-not-devcontainer ## Start the app container
	$(COMPOSE) up -d

down: check-not-devcontainer ## Stop and remove the app container
	$(COMPOSE) down

logs: check-not-devcontainer ## Tail the app container logs
	$(COMPOSE) logs -f





dev: check-devcontainer ## Run the Streamlit app for development
	@uv run streamlit run app.py

eval: check-devcontainer ## Run the retrieval evaluation
	@uv run eval/evaluate.py