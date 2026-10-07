"""
seaf_agent.config — Central Configuration for Sovereign-Edge Agentic Framework (SEAF).
Contract parameters:
- Target Hardware: Edge AI PC / Laptop (RTX 3050 6GB / RTX 2050 4GB / Intel Core i5 / 16GB RAM)
- VRAM Budget: 8,192 MB (Unified memory / edge budget) with INT4 AutoRound (4x scaling)
- Context Windows: 4096 tokens base -> 2048 tokens throttled (>80% RAM pressure)
- Local Models: Ollama (qwen2.5-coder:7b for coding, phi4-mini:latest for edge fallback/audit, nomic-embed-text:latest for RAG)
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "frames.db"
MEMORY_MD_DIR = DATA_DIR / "memory_md"
MEMORY_MD_DIR.mkdir(parents=True, exist_ok=True)
MASTER_MD_PATH = MEMORY_MD_DIR / "procedural_memory_master.md"
CHROMA_DIR = DATA_DIR / "chroma_db"
WORKSPACE_DIR = DATA_DIR / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# Hardware & Telemetry Contract
VRAM_BUDGET_MB = int(os.getenv("SEAF_VRAM_BUDGET_MB", "8192"))
QUANTIZATION_PROFILE = os.getenv("SEAF_QUANT_PROFILE", "INT4_AUTOROUND")
RAM_PRESSURE_THRESHOLD_PCT = float(os.getenv("SEAF_RAM_PRESSURE_PCT", "80.0"))
GPU_CACHE_TTL_SEC = float(os.getenv("SEAF_GPU_CACHE_TTL_SEC", "60.0"))

# Context Windows (Adaptive Throttling)
BASE_CONTEXT_WINDOW = int(os.getenv("SEAF_BASE_CONTEXT_WINDOW", "4096"))
THROTTLED_CONTEXT_WINDOW = int(os.getenv("SEAF_THROTTLED_CONTEXT_WINDOW", "2048"))
MAX_OUTPUT_TOKENS = int(os.getenv("SEAF_MAX_OUTPUT_TOKENS", "1536"))

# Local Edge Endpoints & Models (Ollama on localhost)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LOCAL_CODING_SLM = os.getenv("LOCAL_CODING_SLM", "qwen2.5-coder:7b")
LOCAL_FALLBACK_SLM = os.getenv("LOCAL_FALLBACK_SLM", "phi4-mini:latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

# Air-Gapped Sovereign Cloud VDC Endpoint
SOVEREIGN_VDC_ENDPOINT = os.getenv("SEAF_SOVEREIGN_VDC_ENDPOINT", "https://sovereign-vdc.enterprise.internal/v1/chat")
SOVEREIGN_CLOUD_MODEL = os.getenv("SEAF_SOVEREIGN_CLOUD_MODEL", "sovereign-70b-instruct")

# Reasoner-Critic Settings
MAX_RETRIES = int(os.getenv("CRITIC_MAX_RETRIES", "3"))
EXEC_TIMEOUT_SEC = int(os.getenv("EXEC_TIMEOUT_SEC", "120"))
PONYTAIL_STRICT = os.getenv("PONYTAIL_STRICT", "false").lower() == "true"
