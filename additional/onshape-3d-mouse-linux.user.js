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

    function getViewportRect() {
        // Prefer the main WebGL canvas so toolbar area is excluded.
        // Onshape renders into a <canvas> that usually covers the right-hand
        // portion of the window; fall back to full window if not found.
        const canvas = document.querySelector('canvas');
        if (canvas) {
            const r = canvas.getBoundingClientRect();
            if (r.width > 200 && r.height > 200) return r;
        }
        return { left: 0, top: 0, width: window.innerWidth, height: window.innerHeight };
    }

    function onMouseMove(e) {
        const r = getViewportRect();
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
