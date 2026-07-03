# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/cora-venv \
    PATH="/opt/cora-venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
    && install -d -m 0755 /etc/apt/keyrings \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV" \
    && pip install --upgrade pip

WORKDIR /workspace

FROM base AS runtime

# hatch-vcs derives the version from git metadata, which the build context
# excludes (.dockerignore drops .git/). Local builds fall back to the
# pyproject `fallback-version`; the release workflow passes the real tag
# version here, which hatch-vcs/setuptools-scm honours over git.
ARG CORA_VERSION=
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${CORA_VERSION}

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

RUN useradd --create-home --shell /bin/bash cora
USER cora

ENTRYPOINT ["cora"]

FROM base AS dev

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests
RUN pip install -e ".[dev,otel]"

CMD ["bash"]
