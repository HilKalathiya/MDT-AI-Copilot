#!/usr/bin/env bash
# MDT AI Copilot — Setup Script (Linux/macOS)
# Run from the project root: bash setup.sh

set -e
echo "============================================================"
echo "  MDT AI Copilot — Setup"
echo "============================================================"

# Install Python dependencies
echo "[1/5] Installing Python dependencies..."
pip install -r requirements.txt
echo "      Done."

# Create data directory
echo "[2/5] Creating data directory..."
mkdir -p data
echo "      Done."

# Init DB and generate synthetic data
echo "[3/5] Phase 1 — Initialise DB + generate synthetic data..."
python scripts/demo_phase1.py

# Run ML evaluation
echo "[4/5] Phase 2 — ML evaluation..."
python scripts/demo_phase2.py

# Run tests
echo "[5/5] Unit tests..."
python -m pytest tests/ -v

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. cp .env.example .env  → add your API keys"
echo "  2. python -m rag.build_index     # build RAG index"
echo "  3. python -m copilot.cli          # CLI chat"
echo "  4. streamlit run dashboard/app.py # Streamlit dashboard"
echo "  5. python scripts/demo_full.py   # full demo"
