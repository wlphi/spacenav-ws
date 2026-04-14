import asyncio
import logging
import struct
from pathlib import Path

import typer
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from rich.logging import RichHandler

from spacenav_ws.buttons import load_config
from spacenav_ws.controller import Controller, create_mouse_controller
from spacenav_ws.spacenav import from_message, get_async_spacenav_socket_reader
from spacenav_ws.wamp import WampSession

# Reference to the currently active controller (set when a client connects to /).
# Used by the /cursor endpoint and /events SSE to access per-connection cursor state.
_active_controller: Controller | None = None

# TODO: This handler isn't used for the uvicorn logs and I can't be bothered finding the magic logging incantations to make it so.
logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])

_DEFAULT_ORIGINS = [
    "https://127.51.68.120",
    "https://127.51.68.120:8181",
    "https://3dconnexion.com",
    "https://cad.onshape.com",
]
# Override via ~/.config/spacenav-ws/config.json → "cors_origins": ["https://..."]
ORIGINS = load_config().get("cors_origins", _DEFAULT_ORIGINS)

CERT_FILE = Path(__file__).parent / "certs" / "ip.crt"
KEY_FILE = Path(__file__).parent / "certs" / "ip.key"

cli = typer.Typer()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "OPTIONS"], allow_headers=["*"])


@app.get("/3dconnexion/nlproxy")
async def get_info():
    """HTTP info endpoint for the 3Dconnexion client. Returns which port the WAMP bridge will use and its version."""
    return {"port": 8181, "version": "1.4.8.21486"}


@app.get("/")
def homepage():
    """Tiny bit of HTML that displays mouse movement data"""
    html = """
    <html>
        <body>
            <h1>Mouse Stream</h1>
            <p>Move your spacemouse and motion data should appear here!</p>
            <pre id="output"></pre>
            <script>
                const evtSource = new EventSource("/events");
                const maxLines = 30;
                const lines = [];

                evtSource.onmessage = function(event) {
                    lines.push(event.data);
                    if (lines.length > maxLines) {lines.shift()}
                    document.getElementById("output").textContent = lines.join("\\n");
                };
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)


async def get_mouse_event_generator():
    reader, _ = await get_async_spacenav_socket_reader()
    while True:
        mouse_event = await reader.readexactly(32)
        nums = struct.unpack("iiiiiiii", mouse_event)
        event_data = from_message(list(nums))
        c = _active_controller
        if c is not None:
            debug = (
                f" | ndc=({c._cursor_ndc[0]:.3f},{c._cursor_ndc[1]:.3f})"
                f" active={c._cursor_active}"
                f" pivot=[{c._cursor_debug_pivot[0]:.3f},{c._cursor_debug_pivot[1]:.3f},{c._cursor_debug_pivot[2]:.3f}]"
                f" dist={c._cursor_debug_dist:.3f}"
                f" vh={c._cursor_debug_viewport_half:.3f}"
                f" cursor={c._cursor_debug_used_cursor}"
            )
        else:
            debug = " | no active controller"
        yield f"data: {event_data}{debug}\n\n"


@app.get("/events")
async def event_stream():
    """Stream mouse motion data"""
    return StreamingResponse(get_mouse_event_generator(), media_type="text/event-stream")


@app.websocket("/cursor")
async def cursor_endpoint(ws: WebSocket):
    """Receives mouse cursor NDC coords from the userscript for pivot computation."""
    await ws.accept()
    if _active_controller is not None:
        _active_controller._cursor_active = True
    try:
        while True:
            data = await ws.receive_json()
            if _active_controller is not None:
                _active_controller._cursor_ndc[0] = float(data.get("x", 0.0))
                _active_controller._cursor_ndc[1] = float(data.get("y", 0.0))
    except Exception:
        pass
    finally:
        if _active_controller is not None:
            _active_controller._cursor_active = False


@app.websocket("/")
async def nlproxy(ws: WebSocket):
    """This is the websocket that webapplications should connect to for mouse data"""
    global _active_controller
    wamp_session = WampSession(ws)
    spacenav_reader, _ = await get_async_spacenav_socket_reader()
    ctrl = await create_mouse_controller(wamp_session, spacenav_reader)
    _active_controller = ctrl
    try:
        # TODO, better error handling then just dropping the websocket disconnect on the floor?
        async with asyncio.TaskGroup() as tg:
            tg.create_task(ctrl.start_mouse_event_stream(), name="mouse")
            tg.create_task(ctrl.wamp_state_handler.start_wamp_message_stream(), name="wamp")
    finally:
        if _active_controller is ctrl:
            _active_controller = None


@cli.command()
def serve(host: str = "127.51.68.120", port: int = 8181, hot_reload: bool = False):
    """Start the server that sends spacenav to browser based applications like onshape"""
    logging.warning(f"Navigate to: https://{host}:{port} You should be prompted to add the cert as an exception to your browser!!")
    uvicorn.run(
        "spacenav_ws.main:app", host=host, port=port, ws="auto", ssl_certfile=CERT_FILE, ssl_keyfile=KEY_FILE, log_level="info", reload=hot_reload
    )


@cli.command()
def read_mouse():
    """This echos the output from the spacenav socket, usefull for checking if things are working under the hood"""

    async def read_mouse_stream():
        logging.info("Start moving your mouse!")
        async for event in get_mouse_event_generator():
            logging.info(event.strip())

    asyncio.run(read_mouse_stream())


if __name__ == "__main__":
    cli()
