# CAN IDS dashboard image with a bundled demo.
#   docker build -t can-ids .
#   docker run --rm -p 8501:8501 can-ids        -> open http://localhost:8501
#
# Contents: package code, configs, trained ROAD models (about 2.6 MB), saved results/figures
# and small CC BY 4.0 ROAD demo logs (data/demo, about 3 MB). In the Live Monitor tab,
# pick a data/demo file and press Start. Replay uses python-can's in-process
# 'virtual' bus; the container has no access to any CAN hardware.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CANIDS_ROOT=/app

# libgomp1 = OpenMP runtime required by xgboost / scikit-learn on slim images
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so this layer is cached when only code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-deps .

COPY configs/ configs/
COPY app/ app/
COPY scripts/ scripts/
COPY models/ models/
COPY reports/ reports/
COPY data/demo/ data/demo/

# Run as an unprivileged user (needs write access for logs/ and the alert database).
RUN useradd --create-home appuser \
    && mkdir -p /app/logs \
    && chown -R appuser /app/logs /app/reports
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
