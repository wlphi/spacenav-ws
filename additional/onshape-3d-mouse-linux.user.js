// ==UserScript==
// @name         Onshape 3D‑Mouse on Linux (in‑page patch)
// @description  Fake the platform property on 'navigator' to convince Onshape it's running under Windows. This causes it to ask for information on https://127.51.68.120:8181/3dconnexion/nlproxy so that a 3d mouse can be connected.
// @match        https://cad.onshape.com/documents/*
// @run-at       document-start
// @grant        none
// @version 0.0.2
// @license MIT
// @namespace https://greasyfork.org/users/1460506
// ==/UserScript==

Object.defineProperty(Navigator.prototype, 'platform', { get: () => 'Win32' });
console.log('[Onshape patch] navigator.platform →', navigator.platform);

// ── Cursor position relay ────────────────────────────────────────────────────
// Track mouse position in the 3-D viewport and send it to the spacenav-ws
// server as normalised device coordinates (NDC, range [-1, 1]).  The server
// uses these to compute a cursor-based rotation pivot instead of always
// rotating around the model bounding-box centre.
(function () {
    const SERVER = 'wss://127.51.68.120:8181/cursor';
    const SEND_INTERVAL_MS = 50;   // max ~20 updates/s — plenty for pivot

    let ws = null;
    let lastNx = 0, lastNy = 0;
    let pending = false;

    function connect() {
        try {
            ws = new WebSocket(SERVER);
            ws.onopen  = () => console.log('[Onshape patch] cursor WS connected');
            ws.onclose = () => { ws = null; setTimeout(connect, 4000); };
            ws.onerror = () => { ws = null; };
        } catch (e) {
            setTimeout(connect, 4000);
        }
    }

    // Cache the detected viewport canvas to avoid repeated DOM queries.
    let _vpCanvas = null;
    let _vpCanvasChecked = 0;

    function getViewportCanvas() {
        const now = Date.now();
        if (_vpCanvas && now - _vpCanvasChecked < 2000) return _vpCanvas;
        _vpCanvasChecked = now;

        // Onshape places a full-window canvas for the 3-D scene but overlays
        // toolbar / panel divs on top.  document.elementsFromPoint lets us
        // look through the stacking order and find any canvas at the centre of
        // the window, which is reliably inside the 3-D viewport.
        const cx = Math.round(window.innerWidth  * 0.6);  // right of centre → clear of left panel
        const cy = Math.round(window.innerHeight * 0.5);
        const hits = document.elementsFromPoint(cx, cy);
        _vpCanvas = hits.find(el => el.tagName === 'CANVAS' && el.width > 400) || null;
        return _vpCanvas;
    }

    function onMouseMove(e) {
        const canvas = getViewportCanvas();
        if (!canvas) return;

        const r = canvas.getBoundingClientRect();
        if (r.width < 100 || r.height < 100) return;

        // Map pixel position to NDC [-1, 1]; Y is flipped (screen Y grows down).
        lastNx = ((e.clientX - r.left) / r.width)  * 2 - 1;
        lastNy = 1 - ((e.clientY - r.top)  / r.height) * 2;

        if (!pending) {
            pending = true;
            setTimeout(flush, SEND_INTERVAL_MS);
        }
    }

    function flush() {
        pending = false;
        if (ws && ws.readyState === WebSocket.OPEN) {
            try { ws.send(JSON.stringify({ x: lastNx, y: lastNy })); } catch (_) {}
        }
    }

    document.addEventListener('mousemove', onMouseMove, { passive: true });

    // Connect once the page has loaded (the WAMP WebSocket cert will already
    // be accepted by then).
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => setTimeout(connect, 1500));
    } else {
        setTimeout(connect, 1500);
    }
})();
