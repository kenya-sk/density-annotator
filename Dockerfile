FROM python:3.8-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY pyproject.toml .
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl git build-essential \
    && pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install \
    && rm pyproject.toml

#CMD poetry run \
CMD ["echo", "container running..."]