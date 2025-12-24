#!/bin/bash
set -euo pipefail
exec > >(tee -a /databricks/driver/install_vw.log) 2>&1

PYTHON_BIN=$(command -v python3 || command -v python)

"$PYTHON_BIN" -m pip install --upgrade pip || true
"$PYTHON_BIN" -m pip install vowpalwabbit==9.9.0

VW_BIN=$("$PYTHON_BIN" - <<'PY'
import pathlib, vowpalwabbit
root = pathlib.Path(vowpalwabbit.__file__).resolve().parent
candidates = list(root.rglob('vw'))
for c in candidates:
    if c.is_file():
        print(c)
        break
else:
    print("")
PY
)

if [[ -z "$VW_BIN" ]]; then
  echo "[init] Package did not include vw binary, downloading release..."
  VW_URL="https://github.com/VowpalWabbit/vowpal_wabbit/releases/download/9.9.0/vowpalwabbit-9.9.0-linux-x86_64.tar.gz"
  curl -L "$VW_URL" -o /tmp/vw.tar.gz
  tar -xzf /tmp/vw.tar.gz -C /tmp
  VW_BIN=$(find /tmp -maxdepth 5 -type f -name vw | head -n1 || true)
fi

if [[ -z "$VW_BIN" ]]; then
  echo "[init] ERROR: 'vw' binary not found" >&2
  exit 1
fi

chmod +x "$VW_BIN" || true
ln -sf "$VW_BIN" /usr/local/bin/vw
vw --version
