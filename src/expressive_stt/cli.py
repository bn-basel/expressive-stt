"""Command line: `estt transcribe file.wav --format text|json|llm` and `estt serve`."""

from __future__ import annotations

import argparse
import sys

from .formats import FORMATS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="estt", description="Expressive speech-to-text")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="transcribe an audio file")
    t.add_argument("file")
    t.add_argument("--format", "-f", choices=list(FORMATS), default="text")
    t.add_argument("--output", "-o", help="write to this file instead of stdout")
    t.add_argument("--model", default="large-v3-turbo", help="Whisper model name")
    t.add_argument("--language", "-l", help="language code, e.g. en or ar (default: detect)")
    t.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    t.add_argument("--no-emotion", action="store_true", help="skip emotion, energy and expression")
    t.add_argument("--no-sounds", action="store_true", help="skip laughter/whisper detection")

    s = sub.add_parser("serve", help="run the local HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    a = p.parse_args(argv)
    if a.cmd == "serve":
        import uvicorn

        uvicorn.run("expressive_stt.api:app", host=a.host, port=a.port)
        return 0

    from .formats import render
    from .pipeline import Options, Pipeline

    opts = Options(a.model, a.language, a.device, not a.no_emotion, not a.no_sounds)
    out = render(Pipeline(opts).run(a.file), a.format)
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(out + "\n")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
