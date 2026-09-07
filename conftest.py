"""Make the project importable from wherever pytest is started.

``klh`` and ``emgcasting`` both import :mod:`shared`, so the project root has to
be on ``sys.path`` before either package is imported. The test files insert it
themselves as well; this covers a run started from another directory, and one
that collects a package module before any test module has run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
