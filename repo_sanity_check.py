"""Lightweight repo sanity checker.

Usage:
    python repo_sanity_check.py

Checks:
- Python syntax validity (ast.parse)
- Warns about tracked generated artifacts
"""

import ast
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

IGNORE_DIRS = {".git", "__pycache__", ".backtest_cache", ".venv", "venv"}


def iter_py_files():
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)


def check_syntax():
    errors = []
    for path in iter_py_files():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                ast.parse(fh.read(), filename=path)
        except Exception as e:
            errors.append((path, str(e)))
    return errors


def check_artifacts():
    problems = []
    for root, dirs, files in os.walk(ROOT):
        if ".backtest_cache" in dirs:
            problems.append("Found tracked .backtest_cache directory")
        if ".strategy_store.json" in files:
            problems.append("Found tracked .strategy_store.json file")
    return problems


def main():
    syntax_errors = check_syntax()
    artifact_issues = check_artifacts()

    print("=== SYNTAX CHECK ===")
    if not syntax_errors:
        print("OK: no syntax errors detected")
    else:
        for p, e in syntax_errors:
            print(f"ERROR: {p}: {e}")

    print("\n=== ARTIFACT CHECK ===")
    if not artifact_issues:
        print("OK: no tracked runtime artifacts found")
    else:
        for issue in artifact_issues:
            print(f"WARNING: {issue}")


if __name__ == "__main__":
    main()
