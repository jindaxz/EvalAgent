# RAG-LLM-Metric

A retrieval-augmented generation LLM evaluation framework with optional GPU acceleration.

## Installation

### Prerequisites
- Python 3.9+
- [Poetry](https://python-poetry.org/) (recommended)
- NVIDIA drivers (for GPU support only)

### Using Poetry (Recommended)

#### CPU Installation (Default)
For CPU-only environments:
```bash
poetry install -E cpu
```
GPU/CUDA Installation
```bash
poetry source add pytorch https://download.pytorch.org/whl/cu121

poetry install -E gpu
```