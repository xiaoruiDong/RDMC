test-all:
	pytest

install:
	python -m pip install --no-deps -vv ./

install-dev:
	python -m pip install --no-deps -vv -e ./

documentation:
	$(MAKE) -C docs clean html

style-check:
	pycodestyle \
		--statistics \
		--count \
		--max-line-length 150 \
		--max-doc-length 200 \
		--ignore=E203,E266,E501,W503,W505 \
		--show-source \
		./
