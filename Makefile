
run_train:
	python -c 'from braintumorclassification.interface.main import train; train()'

run_pred:
	python -c 'from braintumorclassification.interface.main import pred; pred()'

run_evaluate:
	python -c 'from braintumorclassification.interface.main import evaluate; evaluate()'

run_all: run_train run_pred run_evaluate

default: pytest

# default: pylint pytest

# pylint:
# 	find . -iname "*.py" -not -path "./tests/test_*" | xargs -n1 -I {}  pylint --output-format=colorized {}; true

pytest:
	echo "no tests"

# ----------------------------------
#         LOCAL SET UP
# ----------------------------------

install_requirements:
	@pip install -r requirements.txt

# ----------------------------------
#         HEROKU COMMANDS
# ----------------------------------

streamlit:
	-@streamlit run braintumorclassification/frontend/app.py


# ----------------------------------
#    LOCAL INSTALL COMMANDS
# ----------------------------------
install:
	@pip install . -U

clean:
	@rm -fr */__pycache__
	@rm -fr __init__.py
	@rm -fr build
	@rm -fr dist
	@rm -fr *.dist-info
	@rm -fr *.egg-info
	-@rm model.joblib

download_data:
	@gsutil -m cp -r gs://braintumorclassification/raw_data/Testing ./raw_data
	@gsutil -m cp -r gs://braintumorclassification/raw_data/Training ./raw_data
