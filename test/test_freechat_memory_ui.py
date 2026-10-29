"""Smoke test for FreeChat. Run from the repo root:
PYTHONPATH=. .venv/bin/python test/test_freechat_memory_ui.py
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = Path(__file__).resolve().parent.parent / "pages" / "10_🎙️_FreeChat.py"

at = AppTest.from_file(str(PAGE), default_timeout=60).run()
assert not at.exception, at.exception
labels = [getattr(expander, "label", "") for expander in at.sidebar.expander]
assert any("小南瓜记得的事" in label for label in labels), labels
print("FreeChat memory UI smoke OK")
