# Kepler — Planetary Image Processing Suite
Kepler is a free, open-source, actively developed planetary image processing application for Linux, macOS and Windows.

## Reporting Bugs and Requesting Features
To report a bug, request a feature, or ask a question, please use the **Issues** tab at the top of this repository.
1. Click **Issues**
2. Click **New Issue**
3. Choose the Bug Report or Feature Request template
4. Fill in the details and submit

This helps keep all feedback organized and easy to track.

## Features
- Wavelet sharpening (Gaussian, Z-Gaussian, Bilateral, B3-spline) with per-layer Sharpen, Denoise, and SharpenFilter controls
- FFT denoising driven by an interactive frequency-spectrum graph — POST (after all wavelet layers) and PRE-Layer 1/2/3 stages can be enabled in **any combination**, each keeping its own Start, End and Curve band
- RGB color balance, saturation, vibrance, hue rotation
- Levels & Curves with draggable tone curve canvas
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- De-rind — limb dark-undershoot correction with Saturn mode, Arc Start/End gap control, and Show Mask overlay
- Channel View — inspect L, R, G, B, and Saturation channels independently in the preview
- Projects — save and reload all processing settings to a named .kproj file; separate image and project folders remembered between sessions
- Preview rotation with batch sync and export
- Batch processing with animation export (WebP/GIF)
- **Planetary de-rotation** — co-add multiple sharpened stacks captured over 20–30 minutes using true oblate-ellipsoid geometry, IAU-2018 pole orientation and JPL Horizons ephemeris, supporting Jupiter (CM II), Saturn (CM III), and Mars (CM I). A **De-rotation LD** control softens the disc limb to suppress bright fringes and dark halos on the stacked result
- **Image Measurement** — a wireframe (limb, equator, grid, and Saturn's rings) fitted to every frame with F11 or by hand. It measures the disc center, size and the pole's orientation on your sensor, so images need no pre-rotation and drift between frames is corrected. A **Measurement LD** compensation brightens the limb in the preview so the disc edge is easy to see and fit

## Download
See the [Releases page](https://github.com/tbgh011/kepler/releases) to download the latest version.

## Requirements
- Windows 10/11 or Linux (tested on Linux Mint): Python 3.8+
- macOS Catalina (10.15) and later: Python 3.11+ (installed automatically by the installer)
- The included installer handles all dependencies automatically

## Installation
### Windows
Download `kepler_v131.zip`, then right-click it and choose **Extract All**.

Windows extracts into a new folder named after the zip, so you may end up with `kepler_v131\kepler_v131\` — that is normal. To avoid it, delete the trailing `kepler_v131` from the destination path in the Extract dialog.

Open the extracted `kepler_v131` folder, go to `installer\windows\`, then right-click `install.bat` and choose **Run as administrator**.

### Linux
Download `kepler_v131.zip` to your Downloads folder, then open Terminal and run:

```bash
cd ~/Downloads
unzip kepler_v131.zip
bash kepler_v131/installer/linux/install.sh
```

### macOS
Download `kepler_v131.zip` — macOS extracts it to `~/Downloads/kepler_v131` automatically. Then open Terminal and run:

```bash
bash ~/Downloads/kepler_v131/installer/macos/install.sh
```

A new Terminal window starts in your home folder, so these commands work as typed. If you put the zip somewhere other than Downloads, use that folder instead.

The installer GUI will open and guide you through the rest of the process.

## Documentation
Full documentation ships inside the download in the `docs/` folder, and is also available as individual PDF and DOCX files on the Releases page:

| Document | Description |
|----------|-------------|
| Installation Guide | Step-by-step installation for Windows, macOS, and Linux |
| User Guide | Interface reference, tab descriptions, practical tips, and a step-by-step guide to FFT denoising |
| Technical Reference | Algorithm details, architecture, API reference |
| Tutorials | Worked examples for Jupiter, Saturn, and Mars |

## Copyright
Copyright © 2026 Tony Bailey. (tbgh011)

This project — **Kepler: Planetary Image Processing Suite** — is released under the MIT License.  
You are free to use, modify, distribute, and build upon this software, provided that the original copyright notice and license text are included in all copies or substantial portions of the software.

See the [LICENSE](LICENSE) file for full terms.
