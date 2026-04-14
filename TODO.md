User observations

FIXED
* ~~Perspective warping when using the ISO1, F, R, T Buttons~~ → force orthographic before fitting in _action_set_view
* ~~The rotate button works only once and in the direction inverse to the icon~~ → button 17 now mapped to rotate_view_cw (Shift = rotate_view_ccw), proper 90° roll
* ~~Some icons (perspectives) are not displayed~~ → added icons.py with inline SVGs for all 9 default view hotkeys
* ~~Display shows garbage~~ → removed Z_FIXED from deflate compression
* ~~V1-V3 save/recall: crashes mouse loop on WAMP error~~ → added try/except in event loop and try/finally in recall to guarantee motion=False
* ~~The rotation lock status LED does not light up~~ → EV_LED/LED_MISC via evdev event node; needs re-login for input group to take effect
* ~~Drop Cairo dependency~~ → replaced cairosvg with resvg CLI subprocess; install with `sudo pacman -S resvg`
* ~~Scipy dependency~~ → removed; rotation now uses Rodrigues axis-angle formula
* ~~Shift modifier drops when keyboard key is pressed~~ → workaround for spacenavd bug #78: release suppressed within 150 ms of key inject

OPEN
* Save and recall views (V1-V3) — debug logging in place, needs live validation
* Rotate around cursor / pivot — needs live validation (pivot.position → cursor NDC → bbox fallback chain in place)
* Analyze and document the functional gap to the officially supported windows driver: https://3dconnexion.com/us/applications/ptc-onshape/learn-more/?tab=navigation#navigation-spacemouse → docs/windows-driver-gap.md
* ~~Test if 6-axis knob LED is RGB capable~~ → NOT RGB. HID Report 0x04 (UsagePage 0x08) exposes only 2 binary LED bits (usage 0x4B + 0x4C). Vendor feature reports 0xFF00 exist but would need USB capture from Windows driver to reverse-engineer. Colour feedback not feasible without that work.

Source: https://github.com/jordens/spacenav-ws
 * Math model adopted: frustum-proportional pan, center-preserving ortho zoom, Rodrigues rotation (from PR #5 analysis)
 * PR #5 (jordens rewrite) reviewed; key improvements cherry-picked — full merge deferred (architectural overhaul)
 * Notes on Onshape protocol: https://github.com/jordens/spacenav-ws/blob/rewrite-transform/docs/onshape-observations.md

Architecture decisions:
 * DECISION: do not split logic into spacenavd — spacenavd handles static/global/linear device transforms only (dead zones, sensitivity, axis inversion via /etc/spnavrc); all Onshape-specific logic stays in Python
 * spnavcfg NOT recommended (Qt5→Qt6 migration incomplete, AUR builds unreliable); edit /etc/spnavrc directly instead
 * Config stays in ~/.config/spacenav-ws/config.json (cors_origins, button_map, shift_map, hotkeys, motion scales)
 * ~~Look at jordens Display implementation~~ → reviewed PR #134; pipeline identical to ours (RGB565 → zlib → USB bulk). Nothing to adopt. Cairo dependency was why maintainer rejected it.
 * ~~Reconsider logic split (spacenavd vs spacenav-ws)~~ → resolved: spacenavd = dumb event source + device config; spacenav-ws = all Onshape application logic. cursor_state.py global removed; state now on Controller instance.