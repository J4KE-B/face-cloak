"""Write server/openapi.json (OpenAPI doc + JSON schemas of the contract). Loads no models."""

from __future__ import annotations

import json
from pathlib import Path

from server.app import create_app
from server.engine import StubEngine
from server.settings import Settings


def main() -> None:
    app = create_app(Settings(demo_key="schema-export-key"), engine_factory=lambda: StubEngine())
    out = Path(__file__).with_name("openapi.json")
    out.write_text(json.dumps(app.openapi(), indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
