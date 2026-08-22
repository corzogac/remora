# Poseidon ops toolkit (ai4water box, 100.118.223.14)

Operational tooling for running the remora/Ornith endpoint on the Poseidon
Windows host (HP Zbook 15 G3, i7-6700HQ, 32 GB, Win10 Education).

## Files
- `ornith_tray.ps1` — system-tray monitor (pure .NET, no installs): green dot
  = server healthy on :11435, red = stopped; Start/Stop/Status/Exit menu;
  polls /health every 10 s. Install: put in the Startup folder (a .lnk with
  `powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File ...`).
- `start_ornith_p.bat` — server launcher: Ornith-1.5-35B-A3B Q4_K_M,
  64K context, q8 KV cache, CPU-only (-ngl 0), port 11435, API key auth.
  Run as a schtasks SYSTEM task with /sc onstart for boot persistence.
- `dl_gguf_p.bat` — resumable model download (curl -C -).
- `build_remora_mt.bat` — builds the remora-llama fork for Windows with the
  static CRT (/MT, CPU-only): single portable exe, no VC runtime dependency
  (fixes 0xC0000135 loader errors on clean Win10 boxes). Requires VS2022
  Build Tools + ninja + git safe.directory exception for the repo.

## Agent wiring
- On Poseidon:    OPENAI_BASE_URL=http://127.0.0.1:11435/v1
- Tailnet agents: OPENAI_BASE_URL=http://100.118.223.14:11435/v1
- Key: see start bat. Model is a reasoning model (<think> first).

## Status 2026-08-21
Endpoint live on the STOCK b10509 build (~3.2 tok/s); remora static-CRT build
in progress to make remora the core on this host too.
