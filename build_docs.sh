#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m sphinx -W --keep-going -b html docs docs/_build/html
