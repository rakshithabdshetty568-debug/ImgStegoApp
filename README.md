# LSB & PVD Image Steganography (Team-10)

A collection of Python scripts demonstrating image steganography techniques (LSB and PVD), with utilities and PyInstaller build specs for Windows executables.

## Features
- Encode and decode text and image payloads into cover images using LSB/PVD techniques.
- Utilities for capacity estimation and quality metrics (MSE/PSNR, altered pixels).
- GUI examples implemented with Tkinter (multiple `main*.py` entrypoints).
- PyInstaller `*.spec` files included to produce standalone Windows executables.

## Repository layout
- `main1.py` … `main8.py` — example entry scripts (variants of GUI/CLI or algorithm demos).
- `LSBImageStegApp.spec`, `PVDImageStegApp.spec` — PyInstaller specs used to build the EXEs.
- `build/` — output directory created by PyInstaller when building the apps.

## Requirements
- Python 3.8 or newer
- Pillow, numpy (used by the scripts)

Install dependencies (recommended in a venv):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If `requirements.txt` is not present, install manually:

```bash
pip install pillow numpy
```

## Running the scripts
- Quick run (example):

```bash
python main8.py
```

Replace `main8.py` with the desired entry script (e.g. `main1.py`). Many of the `main*.py` files launch a Tkinter GUI for encoding/decoding.

## Building Windows executables
PyInstaller specs are included. To build the LSB GUI executable on Windows:

```bash
pyinstaller LSBImageStegApp.spec
```

For the PVD app:

```bash
pyinstaller PVDImageStegApp.spec
```

After building, check the `build/` and `dist/` folders for outputs and bundled executables.

## Notes
- Inspect each `main*.py` to see specific command-line options or GUI features — scripts may include algorithm variants and helper utilities.
- The build outputs in `build/` indicate prior PyInstaller runs and can be removed safely before a fresh build.

## License & Credits
This project was prepared by Team-10 for an academic project. Check with the original authors for licensing and attribution.

---
If you want, I can: add a `requirements.txt`, extract concise usage examples for each `main*.py`, or produce a small demo script — tell me which you'd like next.
