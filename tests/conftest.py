import pytest


@pytest.fixture(autouse=True)
def _no_diarize(monkeypatch):
    """Force DIARIZE=false for all tests.

    Tests run against the stub backend; pyannote diarization can't run without
    an HF_TOKEN, so the no-diarize path (backend speaker labels kept verbatim)
    is the default. Tests that want diarize behaviour re-override the env.
    """
    monkeypatch.setenv("DIARIZE", "false")
    from meetingtool import config

    config._settings = None
    yield
    config._settings = None
