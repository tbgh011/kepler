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
- FFT denoising with interactive frequency curve and pre/post-wavelet placement
- RGB color balance, saturation, vibrance, hue rotation
- Levels & Curves with draggable tone curve canvas
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- De-rind — limb dark-undershoot correction with Saturn mode, Arc Start/End gap control, and Show Mask overlay
- Channel View — inspect L, R, G, B, and Saturation channels independently in the preview
- Projects — save and reload all processing settings to a named .kproj file; separate image and project folders remembered     between sessions
- Preview rotation with batch sync and export
- Batch processing with animation export (WebP/GIF)

## Download

See the [Releases page](https://github.com/tbgh011/kepler/releases) to download the latest version.

## Requirements

- Python 3.8+
- Windows 10/11 or Linux (tested on Linux Mint)
- macOS Catalina (10.15) and later
- The included installer handles all dependencies automatically

## Installation

### Windows
Extract the zip, open `installer\windows\`, right-click `install.bat` and choose **Run as administrator**.

### macOS
Extract the zip, open Terminal, `cd` to `installer/macos/`, and run:
```bash
bash install.sh
```

### Linux
Extract the zip, open Terminal, `cd` to `installer/linux/`, and run:
```bash
bash install.sh
```

The installer GUI will open and guide you through the rest of the process.

## Documentation

Full documentation is included as downloads on the Releases page:

| Document | Description |
|----------|-------------|
| Installation Guide | Step-by-step installation for Windows, macOS, and Linux |
| User Guide | Interface reference, tab descriptions, practical tips |
| Technical Reference | Algorithm details, architecture, API reference |
| Tutorials | Worked examples for Jupiter, Saturn, and other targets |

## Copyright

Copyright © 2026 Tony Bailey. (tbgh011)

This project — **Kepler: Planetary Image Processing Suite** — is released under the MIT License.  
You are free to use, modify, distribute, and build upon this software, provided that the original copyright notice and license text are included in all copies or substantial portions of the software.

See the [LICENSE](LICENSE) file for full terms.
