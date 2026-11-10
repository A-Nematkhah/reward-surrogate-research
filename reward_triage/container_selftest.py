"""Self-checks expected to run inside the triage container (network none)."""

from __future__ import annotations

import os
import socket
import sys


def main() -> int:
    # Env secrets must be absent / empty inside the worker.
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN"):
        val = os.environ.get(key, "")
        if val:
            print(f"FAIL: secret env {key} is set inside container", file=sys.stderr)
            return 1

    # Outbound sockets must fail under network_mode: none.
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2.0)
        print("FAIL: socket connect succeeded (network not isolated)", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"ok: socket blocked ({type(exc).__name__})")

    print("container_selftest: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
