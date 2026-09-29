"""Read project-local settings without modifying the process environment."""
import os
from pathlib import Path
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_settings():
    values = dotenv_values(PROJECT_ROOT / ".env", encoding="utf-8-sig")
    return {**{key: value for key, value in values.items() if value is not None}, **os.environ}
