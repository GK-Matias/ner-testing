FROM python:3.8.10
# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

RUN pip install "poetry==1.2.0"
RUN poetry config virtualenvs.create false

COPY poetry.lock .
COPY pyproject.toml .
RUN poetry install
COPY test_ner_model.py .
COPY train_test_ner_model.py .
# copy models to container if present
# COPY model1/ ./model1
# COPY model2/ ./model2
CMD tail -f /dev/null