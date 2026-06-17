from . import scribe_tools  # noqa: F401  — new thin surface (transcribe, read_transcript)
from . import tools  # noqa: F401  — legacy ontology tools (removed in Phase 2)
from .server import mcp


def main() -> None:
    try:
        mcp.run()
    finally:
        # Stop accepting new jobs on the way out. We don't `wait=True` because
        # a running ASR pass can take hours; the worker thread is non-daemon
        # so Python will wait for it to finish naturally. If the process is
        # killed hard, db.init()'s crash reconciliation handles it on next
        # boot. Touching `_runner` directly (instead of get_runner()) avoids
        # spinning up a runner just to shut it down.
        from . import jobs as _jobs
        if _jobs._runner is not None:
            _jobs._runner.shutdown(wait=False)


if __name__ == "__main__":
    main()
