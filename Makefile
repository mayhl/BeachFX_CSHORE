# Docs tasks (MkDocs Material + mkdocstrings), run via uv's `docs` dependency group.
UV        ?= uv
DOCS_GROUP := --group docs

.PHONY: help docs docs-serve docs-build docs-deploy docs-clean

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

docs: docs-serve  ## Alias for docs-serve

docs-serve:  ## Live-reloading local docs at http://127.0.0.1:8000
	$(UV) run $(DOCS_GROUP) mkdocs serve

docs-build:  ## Build the static site into ./site
	$(UV) run $(DOCS_GROUP) mkdocs build

docs-deploy:  ## Manual deploy to the gh-pages branch (CI normally handles this)
	$(UV) run $(DOCS_GROUP) mkdocs gh-deploy --force

docs-clean:  ## Remove the built site
	rm -rf site
