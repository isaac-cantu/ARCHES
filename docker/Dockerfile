# ARCHES training image: Python + torch + corsario, ready to run
# pipeline/run_pipeline.py against mounted data/experiments directories.
#
# Build:
#   docker build -t arches -f docker/Dockerfile .
# Run:
#   docker run --rm \
#       -v /path/to/corsika/processed:/data \
#       -v $(pwd)/experiments:/experiments \
#       arches \
#       python pipeline/run_pipeline.py --configs pipeline/config_example.yaml

FROM python:3.11-slim

# build-essential + cmake: required to build the corsario C++ extension.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[corsario]"

WORKDIR /app/src
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

CMD ["bash"]
