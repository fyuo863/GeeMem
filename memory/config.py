"""Read settings exclusively from the project root .env."""
from pathlib import Path
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_settings():
    values = dotenv_values(PROJECT_ROOT / ".env", encoding="utf-8-sig", interpolate=False)
    return {key: value for key, value in values.items() if value is not None}
