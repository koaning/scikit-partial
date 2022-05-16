black:
	black skpartial tests setup.py --check

flake:
	flake8 skpartial tests setup.py

test:
	pytest

interrogate:
	interrogate -vv --ignore-nested-functions --ignore-semiprivate --ignore-private --ignore-magic --ignore-module --ignore-init-method --fail-under 100 tests
	interrogate -vv --ignore-nested-functions --ignore-semiprivate --ignore-private --ignore-magic --ignore-module --ignore-init-method --fail-under 100 skpartial

check: black flake interrogate test

clean:
	rm -rf *.h5
	rm -rf *.joblib
install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"
	pre-commit install
	python -m pip install wheel twine

pypi:
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*