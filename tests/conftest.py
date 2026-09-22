"""
Hands pytest the fixtures that live in core, since only a top level conftest may

The fixtures themselves are in core/conftest.py next to the harness they drive
"""

pytest_plugins = ["core.conftest"]
