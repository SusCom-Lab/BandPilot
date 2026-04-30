"""Isolated baseline-suite package for reviewer-facing baseline experiments.

This package keeps all new baseline code, configs, artifacts, and report logic
under ``evaluation/baselines`` so the existing compare/search pipeline remains
read-only. The suite reuses the current repository's runtime helpers by import,
but it never edits or registers new algorithms outside this directory.
"""

