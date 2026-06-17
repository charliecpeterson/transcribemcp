from . import scribe_tools  # noqa: F401  — registers transcribe + read_transcript
from .server import mcp


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
