from memory import config
from memory.llm import LLM


def test_project_env_from_another_directory(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text(
        'LLM_BASE_URL="https://example.test/v1/"\n'
        'LLM_API_KEY="file-key"\nLLM_MODEL="file-model"\n',
        encoding="utf-8-sig",
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", project)
    monkeypatch.chdir(tmp_path)
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.setenv(key, "ignored-environment-value")
    llm = LLM()
    assert (llm.base_url, llm.key, llm.model) == (
        "https://example.test/v1", "file-key", "file-model")
    monkeypatch.setenv("LLM_MODEL", "environment-model")
    assert LLM().model == "file-model"
    # Recreating the adapter reads updated settings, without stale os.environ values.
    (project / ".env").write_text("LLM_API_KEY=updated-key\n", encoding="utf-8")
    assert LLM().key == "updated-key"


def test_missing_env_uses_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.setenv(key, "ignored-environment-value")
    llm = LLM()
    assert llm.key == ""
    assert llm.base_url == "https://api.openai.com/v1"
    assert llm.model == "gpt-4.1-mini"


def test_env_interpolation_is_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("SECRET_FROM_ENV", "must-not-load")
    (tmp_path / ".env").write_text('LLM_API_KEY=${SECRET_FROM_ENV}\n', encoding="utf-8")
    assert LLM().key == "${SECRET_FROM_ENV}"
