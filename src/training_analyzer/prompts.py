"""Load editable model prompts from the project prompts directory."""

from __future__ import annotations

from string import Template

from training_analyzer.gemini import PROJECT_ROOT


PROMPTS_DIR = PROJECT_ROOT / "prompts"


def load_prompt(name: str) -> str:
    path = (PROMPTS_DIR / f"{name}.md").resolve()
    if not path.is_relative_to(PROMPTS_DIR.resolve()):
        raise ValueError(f"Invalid prompt name: {name}")
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file was not found: {path}")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Prompt file is empty: {path}")
    return content


def render_prompt(name: str, **values: object) -> str:
    return Template(load_prompt(name)).substitute(
        {key: str(value) for key, value in values.items()}
    )
