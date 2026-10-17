"""Create the demo user and a realistic task list through the core API.

    python scripts/seed_demo.py [--base-url http://localhost:8000]     (or: make seed)

Demo login: demo@j26se360.lk / demo-pass-2026 — for local development only.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

from _api import DEMO_EMAIL, DEMO_PASSWORD, client, ensure_demo_tasks, login_or_register


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--timezone", default="Asia/Colombo")
    a = ap.parse_args()

    c = client(a.base_url)
    login_or_register(c, tz=a.timezone)
    tasks = ensure_demo_tasks(c, datetime.now(ZoneInfo(a.timezone)))
    print(f"Demo user ready: {DEMO_EMAIL} / {DEMO_PASSWORD}")
    for t in tasks:
        print(f"  - [{t['status']}] {t['title']}")


if __name__ == "__main__":
    main()
