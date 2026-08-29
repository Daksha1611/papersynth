# Runtime image for PaperSynth.
#
# poppler-utils is not optional despite the LaTeX path being preferred:
# pdftotext backs the no-GROBID fallback, and pdftoppm rasterizes pages for
# math recovery. Both are small; leaving them out turns a degraded path into a
# missing one.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing source does not reinstall them.
COPY pyproject.toml README.md ./
COPY papersynth ./papersynth
RUN pip install --upgrade pip && pip install -e ".[api]"

COPY config ./config
COPY scripts ./scripts

# Runs are outputs and belong on a volume, not in the image.
RUN mkdir -p /app/runs && useradd --create-home --uid 1000 papersynth \
    && chown -R papersynth:papersynth /app
USER papersynth

ENV PAPERSYNTH_WORKSPACE=/app/runs \
    PAPERSYNTH_GROBID_URL=http://grobid:8070

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD papersynth --version || exit 1

ENTRYPOINT ["papersynth"]
CMD ["--help"]
