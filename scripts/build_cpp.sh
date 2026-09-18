#!/usr/bin/env bash
# Build the C++ engine, its tests and the Python extension into ./build and ./engine/.
# Usage: scripts/build_cpp.sh [Release|Debug]      (run from the repository root, venv active or .venv present)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || command -v python3 )}
CMAKE=${CMAKE:-$( [ -x .venv/bin/cmake ] && echo .venv/bin/cmake || command -v cmake )}
"$CMAKE" -S . -B build -G "${CMAKE_GENERATOR:-Unix Makefiles}" -DCMAKE_BUILD_TYPE="${1:-Release}" \
  -DPython_EXECUTABLE="$("$PY" -c 'import sys;print(sys.executable)')" \
  -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)"
"$CMAKE" --build build -j
echo "built: $(ls engine/_lob_cpp*.so 2>/dev/null)"
