#!/usr/bin/env python3
"""Guided live tester for spacenav-ws.

Works alongside a running Onshape tab — does NOT replace it.

Two connections only:
  /cursor   WebSocket — injects NDC cursor position programmatically
  /events   SSE       — reads the server's motion+pivot debug stream

Each test:
  1. Sets the cursor NDC position (automatically, no user action)
  2. Prints an instruction ("rotate the puck")
  3. Waits for a motion burst on the SSE stream
  4. Evaluates the result and prints PASS / FAIL with detail
  5. Advances automatically

Usage
-----
    uv run spacenav-ws serve                # already running as service
    uv run python tools/live_test.py

Make sure an Onshape document is open in the browser (provides real
view.affine / view.extents so the server has something to work with).
Press Ctrl-C to abort.
"""

import asyncio
import json
import re
import ssl
import time
from dataclasses import dataclass, field

import numpy as np
import websockets
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# ── Server ────────────────────────────────────────────────────────────────────

HOST = "127.51.68.120"
PORT = 8181

def _make_ssl() -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

# ── Shared live state ─────────────────────────────────────────────────────────

@dataclass
class LiveState:
    sse_connected: bool = False
    cursor_connected: bool = False

    # Cursor we're injecting
    cursor_x: float = 0.0
    cursor_y: float = 0.0
    cursor_enabled: bool = True

    # Latest parsed SSE event
    pivot_src: str = "—"
    pivot_pt: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    pivot_dist: float = 0.0
    pivot_vh: float = 0.0
    sse_cursor_active: bool = False
    sse_motion_count: int = 0  # incremented on every SSE motion event

# ── SSE reader ────────────────────────────────────────────────────────────────

_RE_SRC    = re.compile(r"\bsrc=(\S+)")
_RE_PIVOT  = re.compile(r"\bpivot=\[([^\]]+)\]")
_RE_DIST   = re.compile(r"\bdist=([\d.eE+\-]+)")
_RE_VH     = re.compile(r"\bvh=([\d.eE+\-]+)")
_RE_ACTIVE = re.compile(r"\bactive=(\w+)")

async def sse_task(s: LiveState, quit_event: asyncio.Event) -> None:
    ssl_ctx = _make_ssl()
    request = (
        f"GET /events HTTP/1.1\r\n"
        f"Host: {HOST}:{PORT}\r\n"
        f"Accept: text/event-stream\r\n"
        f"Connection: keep-alive\r\n\r\n"
    ).encode()
    while not quit_event.is_set():
        try:
            reader, writer = await asyncio.open_connection(HOST, PORT, ssl=ssl_ctx)
            writer.write(request)
            await writer.drain()
            # consume HTTP headers
            while True:
                line = await asyncio.wait_for(reader.readline(), 5.0)
                if line in (b"\r\n", b"\n", b""):
                    break
            s.sse_connected = True
            while not quit_event.is_set():
                line = await asyncio.wait_for(reader.readline(), 5.0)
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].decode(errors="replace").strip()
                if "| ndc=" not in data:
                    continue
                m = _RE_SRC.search(data)
                if m:
                    s.pivot_src = m.group(1)
                m = _RE_PIVOT.search(data)
                if m:
                    try:
                        s.pivot_pt = [float(x) for x in m.group(1).split(",")]
                    except ValueError:
                        pass
                m = _RE_DIST.search(data)
                if m:
                    s.pivot_dist = float(m.group(1))
                m = _RE_VH.search(data)
                if m:
                    s.pivot_vh = float(m.group(1))
                m = _RE_ACTIVE.search(data)
                if m:
                    s.sse_cursor_active = m.group(1) == "True"
                s.sse_motion_count += 1
            writer.close()
        except Exception:
            s.sse_connected = False
            if not quit_event.is_set():
                await asyncio.sleep(2)
    s.sse_connected = False

# ── /cursor WebSocket ─────────────────────────────────────────────────────────

async def cursor_task(s: LiveState, quit_event: asyncio.Event) -> None:
    ssl_ctx = _make_ssl()
    while not quit_event.is_set():
        if not s.cursor_enabled:
            s.cursor_connected = False
            await asyncio.sleep(0.2)
            continue
        try:
            async with websockets.connect(
                f"wss://{HOST}:{PORT}/cursor", ssl=ssl_ctx
            ) as ws:
                s.cursor_connected = True
                while not quit_event.is_set() and s.cursor_enabled:
                    await ws.send(json.dumps({"x": s.cursor_x, "y": s.cursor_y}))
                    await asyncio.sleep(0.05)
        except Exception:
            s.cursor_connected = False
            if not quit_event.is_set():
                await asyncio.sleep(1)
    s.cursor_connected = False

# ── Snapshot & evaluation ─────────────────────────────────────────────────────

@dataclass
class Snapshot:
    sse_motion_count: int
    pivot_src: str
    pivot_pt: list
    pivot_dist: float
    pivot_vh: float

def snap(s: LiveState) -> Snapshot:
    return Snapshot(
        sse_motion_count=s.sse_motion_count,
        pivot_src=s.pivot_src,
        pivot_pt=list(s.pivot_pt),
        pivot_dist=s.pivot_dist,
        pivot_vh=s.pivot_vh,
    )

def _check_pivot_src(expected: str, after: Snapshot) -> tuple[bool, str]:
    ok = after.pivot_src == expected
    return ok, f"pivot_src={after.pivot_src!r} (expected {expected!r})"

def _check_pivot_x_positive(after: Snapshot, min_x: float = 0.1) -> tuple[bool, str]:
    ok = after.pivot_pt[0] > min_x
    return ok, f"pivot_x={after.pivot_pt[0]:.3f} (expected >{min_x:.2f})"

def _check_pivot_x_negative(after: Snapshot, max_x: float = -0.1) -> tuple[bool, str]:
    ok = after.pivot_pt[0] < max_x
    return ok, f"pivot_x={after.pivot_pt[0]:.3f} (expected <{max_x:.2f})"

def _check_pivot_near_center(after: Snapshot, max_dist: float = 0.5) -> tuple[bool, str]:
    d = float(np.linalg.norm(after.pivot_pt))
    return d < max_dist, f"|pivot|={d:.3f} (expected <{max_dist:.2f})"

def _and(*checks: tuple[bool, str]) -> tuple[bool, str]:
    return all(ok for ok, _ in checks), "  ".join(d for _, d in checks)

# ── Test suite ────────────────────────────────────────────────────────────────
#
# Each entry: (name, setup_dict, instruction, evaluate_fn, timeout_s)
#
# setup_dict keys:
#   cursor_x, cursor_y   — NDC sent via /cursor (set before waiting for motion)
#   cursor_enabled       — whether /cursor WS is connected at all

TESTS = [
    (
        "SSE connection",
        {},
        "No action needed — checking server connection.",
        lambda before, after, s: (s.sse_connected, f"sse_connected={s.sse_connected}"),
        4.0,
    ),
    (
        "Motion detected",
        {"cursor_enabled": False},
        "Move the puck in any direction.",
        lambda before, after, s: (
            after.sse_motion_count > before.sse_motion_count,
            f"SSE events received: {after.sse_motion_count - before.sse_motion_count}",
        ),
        12.0,
    ),
    (
        "Pivot — cursor at centre (0, 0)",
        {"cursor_x": 0.0, "cursor_y": 0.0, "cursor_enabled": True},
        "Rotate the puck (yaw / pitch / roll) and hold briefly.\n"
        "Cursor is at NDC (0.00, 0.00) — pivot_src must be 'cursor'.",
        lambda before, after, s: _check_pivot_src("cursor", after),
        12.0,
    ),
    (
        "Pivot — cursor right (+0.50, 0)",
        {"cursor_x": 0.5, "cursor_y": 0.0, "cursor_enabled": True},
        "Rotate the puck (yaw / pitch / roll) and hold briefly.\n"
        "Cursor is at NDC (+0.50, 0.00) — pivot_x should be clearly positive.",
        lambda before, after, s: _and(
            _check_pivot_src("cursor", after),
            _check_pivot_x_positive(after, min_x=0.1),
        ),
        12.0,
    ),
    (
        "Pivot — cursor left (−0.50, 0)",
        {"cursor_x": -0.5, "cursor_y": 0.0, "cursor_enabled": True},
        "Rotate the puck (yaw / pitch / roll) and hold briefly.\n"
        "Cursor is at NDC (−0.50, 0.00) — pivot_x should be clearly negative.",
        lambda before, after, s: _and(
            _check_pivot_src("cursor", after),
            _check_pivot_x_negative(after, max_x=-0.1),
        ),
        12.0,
    ),
    (
        "Pivot — cursor out-of-bounds (5.00, 0)",
        {"cursor_x": 5.0, "cursor_y": 0.0, "cursor_enabled": True},
        "Rotate the puck and hold briefly.\n"
        "Cursor NDC (5.00, 0) is far outside the viewport — should fall back to model centre.",
        lambda before, after, s: _check_pivot_src("model_center_oob", after),
        12.0,
    ),
    (
        "Pivot — cursor WS disconnected",
        {"cursor_enabled": False},
        "Rotate the puck and hold briefly.\n"
        "Cursor WebSocket is disabled — should fall back to model_center or native pivot.",
        lambda before, after, s: (
            after.pivot_src in ("model_center", "native"),
            f"pivot_src={after.pivot_src!r} (expected 'model_center' or 'native')",
        ),
        12.0,
    ),
]

# ── Display ───────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool | None = None
    detail: str = ""

_PASS = "[bold green]PASS[/bold green]"
_FAIL = "[bold red]FAIL[/bold red]"
_WAIT = "[yellow]WAITING…[/yellow]"
_SKIP = "[dim]—[/dim]"

def _DOT(ok: bool) -> str:
    return "[green]●[/green]" if ok else "[red]○[/red]"

def _tag(r: TestResult, is_current: bool, current_status: str) -> str:
    if is_current and current_status == "waiting":
        return _WAIT
    if r.passed is None:
        return _SKIP
    return _PASS if r.passed else _FAIL

def _render(
    s: LiveState,
    results: list[TestResult],
    idx: int,
    status: str,
    instruction: str,
    countdown: float,
    live: Live,
) -> None:
    lines = []
    conns = f"SSE {_DOT(s.sse_connected)}  CURSOR {_DOT(s.cursor_connected)}"
    lines.append(f"[bold]spacenav-ws live tester[/bold]   {conns}")
    lines.append("")

    name = results[idx].name
    lines.append(f"[bold cyan]Test {idx+1}/{len(results)}:[/bold cyan] [bold]{name}[/bold]")
    lines.append("")
    for ln in instruction.splitlines():
        lines.append(f"  [italic]{ln}[/italic]")
    lines.append("")

    if status == "stopping":
        lines.append("  [bold yellow]STOP — let the puck settle…[/bold yellow]")
        lines.append(f"  cursor NDC ({s.cursor_x:+.2f}, {s.cursor_y:+.2f})  active={s.sse_cursor_active}")
    elif status == "waiting":
        lines.append(f"  {_WAIT}  timeout {countdown:.0f}s")
        lines.append(f"  cursor NDC ({s.cursor_x:+.2f}, {s.cursor_y:+.2f})  connected={s.cursor_connected}")
    elif status in ("pass", "fail"):
        tag = _PASS if status == "pass" else _FAIL
        lines.append(f"  {tag}  {results[idx].detail}")
    elif status == "done":
        passed = sum(1 for r in results if r.passed)
        lines.append(f"  [bold green]All done — {passed}/{len(results)} passed.[/bold green]")

    if s.sse_connected and status == "waiting":
        p = s.pivot_pt
        lines.append("")
        lines.append(
            f"  [dim]Live: src=[yellow]{s.pivot_src}[/yellow]"
            f"  pt=[{p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}]"
            f"  cursor_active={s.sse_cursor_active}[/dim]"
        )

    lines.append("")
    lines.append("[dim]" + "─" * 56 + "[/dim]")
    lines.append("")

    for i, r in enumerate(results):
        marker = "[cyan]▶[/cyan] " if i == idx else "  "
        tag = _tag(r, i == idx, status)
        detail = f"  [dim]{r.detail}[/dim]" if r.detail else ""
        lines.append(f"{marker}[dim]{i+1:2d}.[/dim] {r.name:<42} {tag}{detail}")

    live.update(Panel(Text.from_markup("\n".join(lines)), subtitle="[dim]Ctrl-C to abort[/dim]"))

# ── Test runner ───────────────────────────────────────────────────────────────

async def run_tests(s: LiveState, quit_event: asyncio.Event) -> None:
    console = Console()
    results = [TestResult(name=t[0]) for t in TESTS]

    with Live(console=console, refresh_per_second=4, screen=False, vertical_overflow="visible") as live:
        for idx, (name, setup, instruction, evaluate, timeout) in enumerate(TESTS):
            if quit_event.is_set():
                break

            # Apply setup
            if "cursor_x" in setup:
                s.cursor_x = setup["cursor_x"]
            if "cursor_y" in setup:
                s.cursor_y = setup["cursor_y"]
            if "cursor_enabled" in setup:
                s.cursor_enabled = setup["cursor_enabled"]

            # Apply cursor WS state change and let it propagate
            await asyncio.sleep(0.5)

            # For tests that need the cursor WS disconnected, confirm via SSE
            if setup.get("cursor_enabled") is False:
                confirm_deadline = time.monotonic() + 3.0
                while s.sse_cursor_active and time.monotonic() < confirm_deadline and not quit_event.is_set():
                    _render(s, results, idx, "stopping", instruction, 99.0, live)
                    await asyncio.sleep(0.1)

            # Wait for any ongoing gesture to end (no SSE events for 0.4s).
            # This prevents a locked pivot from a previous test carrying over.
            if name != "SSE connection":
                last_count = s.sse_motion_count
                quiet_until = time.monotonic() + 0.4
                while time.monotonic() < quiet_until and not quit_event.is_set():
                    if s.sse_motion_count != last_count:
                        last_count = s.sse_motion_count
                        quiet_until = time.monotonic() + 0.4  # reset on new event
                    _render(s, results, idx, "stopping", instruction, 99.0, live)
                    await asyncio.sleep(0.05)

            before = snap(s)
            deadline = time.monotonic() + timeout

            if name == "SSE connection":
                while not s.sse_connected and time.monotonic() < deadline and not quit_event.is_set():
                    _render(s, results, idx, "waiting", instruction, deadline - time.monotonic(), live)
                    await asyncio.sleep(0.25)
                after = snap(s)
            else:
                # Pivot-source tests need 3+ events so the SSE/controller race
                # resolves before we evaluate.  "Motion detected" only needs 1.
                min_events = 1 if name == "Motion detected" else 3
                target = before.sse_motion_count + min_events
                while s.sse_motion_count < target and time.monotonic() < deadline and not quit_event.is_set():
                    _render(s, results, idx, "waiting", instruction, deadline - time.monotonic(), live)
                    await asyncio.sleep(0.1)
                await asyncio.sleep(0.3)  # let SSE parse settle
                after = snap(s)

            if quit_event.is_set():
                break

            # Evaluate
            timed_out = time.monotonic() >= deadline and name != "SSE connection" and after.sse_motion_count == before.sse_motion_count
            if timed_out:
                results[idx].passed = False
                results[idx].detail = "timed out — no motion detected"
            else:
                try:
                    passed, detail = evaluate(before, after, s)
                    results[idx].passed = passed
                    results[idx].detail = detail
                except Exception as exc:
                    results[idx].passed = False
                    results[idx].detail = f"error: {exc}"

            _render(s, results, idx, "pass" if results[idx].passed else "fail", instruction, 0.0, live)
            await asyncio.sleep(2.0)

        _render(s, results, len(results) - 1, "done", "", 0.0, live)
        await asyncio.sleep(4.0)

    quit_event.set()

# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    s = LiveState()
    quit_event = asyncio.Event()
    async with asyncio.TaskGroup() as tg:
        tg.create_task(sse_task(s, quit_event))
        tg.create_task(cursor_task(s, quit_event))
        tg.create_task(run_tests(s, quit_event))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
