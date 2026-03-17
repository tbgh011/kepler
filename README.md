# Kepler — Planetary Image Processing Suite

Kepler is a professional planetary image processing application for Windows and Linux, designed for planetary astrophotographers.

## Features
- Wavelet sharpening (Gaussian, Z-Gaussian, Bilateral, B3-spline)
- FFT denoising with interactive frequency curve
- RGB color balance, saturation, vibrance, hue rotation
- Color Denoise — chroma noise suppression after LRGB sharpening
- Deconvolution — Richardson-Lucy luminance sharpening via Moffat PSF
- Dehaze — chromatic fringing removal at planetary limbs
- Levels & Curves with draggable tone curve canvas
- CLAHE (Contrast Limited Adaptive Histogram Equalisation)
- De-rind (limb halo correction)
- Preview rotation with batch sync
- Batch processing with animation export

## Download
See the [Releases page](https://github.com/tbgh011/kepler/releases) to download the latest version.

**Latest: [Kepler v1.1.6](https://github.com/tbgh011/kepler/releases/tag/v1.1.6)**

## Requirements
- Python 3.8+
- Windows 10/11 or Linux (tested on Linux Mint)
- The included installer handles all dependencies automatically

## Documentation
| Document | View |
|----------|------|
| User Guide | [View PDF](docs/Kepler_User_Guide_v116.pdf) |
| Technical Reference | [View PDF](docs/Kepler_Technical_Reference_v116.pdf) |
| Planet Tutorials | [View PDF](docs/Kepler_Planet_Tutorials_v116.pdf) |
| Installation Guide | [View PDF](docs/Kepler_Installation_Guide_v116.pdf) |

## Installation
1. Download `kepler_v1.1.6.zip` from the [Releases page](https://github.com/tbgh011/kepler/releases)
2. Extract the zip
3. **Windows:** run `installer\windows\install.bat`
4. **Linux:** run `bash installer/linux/install.sh`
