from pathlib import Path


def test_environment_template_contains_model_setting() -> None:
    template = Path(__file__).parents[1] / ".env.example"
    content = template.read_text(encoding="utf-8")
    assert "FORECAST_MODELS" in content
