#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
native_python="${SWIFTCACHE_PYTHON:-.venv/bin/python}"
"$native_python" -m pip install -r requirements-native.txt
SWIFTCACHE_BUILD_CPP=1 "$native_python" setup.py build_ext --inplace
"$native_python" -c 'from swiftcache.engine import _native_scheduler; print(_native_scheduler.__file__)'
