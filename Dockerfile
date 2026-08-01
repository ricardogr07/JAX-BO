# jaxbo quickstart image (SCOPE.md decision 8, issue #39):
#   docker run -p 8888:8888 ghcr.io/ricardogr07/jaxbo
# launches JupyterLab with jaxbo[all] and the examples/ notebooks.
#
# Everything installs frozen from the committed uv.lock (reposage s4
# environment isolation): jaxbo plus its extras from [project.optional-
# dependencies], JupyterLab from the "docker" dependency group.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN useradd --create-home app
USER app
WORKDIR /home/app/jaxbo

# Dependency layer first so source edits do not re-resolve the environment.
COPY --chown=app pyproject.toml uv.lock ./
# --extra examples on top of [all]: the tutorial notebooks import matplotlib.
RUN uv sync --frozen --no-dev --no-install-project \
    --extra all --extra examples --group docker

# The project itself; README.md and LICENSE are package metadata inputs.
COPY --chown=app README.md LICENSE ./
COPY --chown=app jaxbo ./jaxbo
RUN uv sync --frozen --no-dev --extra all --extra examples --group docker

COPY --chown=app examples ./examples

EXPOSE 8888
CMD ["uv", "run", "--no-sync", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
