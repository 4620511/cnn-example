.PHONY: build
build:
	poetry build

.PHONY: clean
clean:
	rm -fr *.egg-info
	rm -fr outputs

.PHONY: format
format:
	poetry run black -v cnn_example
	poetry run black -v tests

.PHONY: install
install:
	poetry install

.PHONY: lint
lint:
	poetry run flake8 cnn_example
	poetry run mypy cnn_example

.PHONY: test
test:
	poetry run pytest -v
