import pytest

from meetingtool import db as db_mod


@pytest.fixture(autouse=True)
def _no_diarize(monkeypatch):
    """Tests use the stub backend; pyannote diarization can't run without HF_TOKEN.

    Forces DIARIZE=false for all tests so submit_transcribe picks the
    no-diarize path (backend's speaker labels are preserved verbatim).
    Tests that explicitly want diarize behaviour can re-override the env.
    """
    monkeypatch.setenv("DIARIZE", "false")
    from meetingtool import config
    config._settings = None
    yield
    config._settings = None


@pytest.fixture
def conn(tmp_path):
    """Fresh SQLite DB per test, reset module-level singleton afterwards."""
    c = db_mod.init(tmp_path / "test.db")
    db_mod.reset_conn_for_tests(c)
    yield c
    c.close()
    db_mod.reset_conn_for_tests(None)


@pytest.fixture
def stub_runner_factory(tmp_path):
    """Build a JobRunner wired to the stub backend with a user-supplied
    window plan. Handles the new protocol (window kwarg, no diarize kwarg).

    Default plan is a single 0..10s window covering the stub's fake audio.
    Pass `windows=[(0, 5), (5, 10)]` to exercise multi-window flows.
    """
    from meetingtool import jobs as jobs_mod
    from meetingtool.backends.stub import StubBackend

    created: list = []

    def factory(*, delay: float = 0.01, windows=None, db_path=None):
        windows = windows if windows is not None else [(0.0, 10.0)]
        stub = StubBackend(delay=delay)

        def transcribe_fn(audio_path, *, progress, window=None):
            return stub.transcribe(audio_path, progress=progress, window=window)

        def plan_fn(_audio_path):
            return list(windows)

        runner = jobs_mod.JobRunner(
            db_path or (tmp_path / "test.db"),
            transcribe_fn,
            plan_windows_fn=plan_fn,
        )
        jobs_mod.reset_runner_for_tests(runner)
        created.append(runner)
        return runner

    yield factory
    for r in created:
        r.shutdown(wait=False)
    jobs_mod = __import__("meetingtool.jobs", fromlist=["reset_runner_for_tests"])
    jobs_mod.reset_runner_for_tests(None)
