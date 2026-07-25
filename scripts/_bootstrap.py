"""Make the entry points in ``scripts/`` safe to run directly.

Running ``python3 scripts/foo.py`` puts ``scripts/`` on ``sys.path[0]``. This
directory contains ``tokenize.py``, which then shadows the standard library's
``tokenize`` module and breaks ordinary imports (``dataclasses`` ->
``inspect`` -> ``linecache`` -> ``tokenize``). Importing this module before any
project import drops the script directory and puts ``src/`` in its place.
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [path for path in sys.path if os.path.abspath(path or os.getcwd()) != _script_dir]

_src = os.path.join(os.path.dirname(_script_dir), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
