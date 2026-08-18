"""
processing.py - Core image processing algorithms for Kepler
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter, sobel
from scipy.signal import fftconvolve
import threading


# ─────────────────────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────────────────────

def load_png(path: str) -> np.ndarray:
    """
    Load a PNG and return float32 (H,W,3) in [0,1].
    PIL silently truncates 16-bit PNGs to uint8 — this reader works at the
    raw byte level to preserve full 16-bit depth.
    """
    import struct, zlib

    with open(path, 'rb') as f:
        magic = f.read(8)

    if magic != b'\x89PNG\r\n\x1a\n':
        # Not a PNG (e.g. JPEG with .png extension) — fall back to PIL
        from PIL import Image as _PIL
        pil = _PIL.open(path).convert("RGB")
        return np.array(pil).astype(np.float32) / 255.0

    with open(path, 'rb') as f:
        f.read(8)  # skip signature
        ihdr = None
        idat_chunks = []
        while True:
            length = struct.unpack('>I', f.read(4))[0]
            chunk_type = f.read(4)
            data = f.read(length)
            f.read(4)  # CRC
            if chunk_type == b'IHDR':
                w          = struct.unpack('>I', data[0:4])[0]
                h          = struct.unpack('>I', data[4:8])[0]
                bit_depth  = data[8]
                color_type = data[9]
                ihdr = (w, h, bit_depth, color_type)
            elif chunk_type == b'IDAT':
                idat_chunks.append(data)
            elif chunk_type == b'IEND':
                break

    w, h, bit_depth, color_type = ihdr
    # color_type: 0=gray, 2=RGB, 4=gray+alpha, 6=RGBA
    if color_type == 2:    channels = 3
    elif color_type == 6:  channels = 4
    elif color_type == 4:  channels = 2
    else:                  channels = 1

    bpp = (bit_depth // 8) * channels
    raw = zlib.decompress(b''.join(idat_chunks))

    rows = []; prev = bytes(w * bpp); pos = 0
    for _ in range(h):
        filt = raw[pos]; pos += 1
        row = bytearray(raw[pos:pos + w * bpp]); pos += w * bpp
        if filt == 1:
            for i in range(bpp, len(row)):
                row[i] = (row[i] + row[i-bpp]) & 0xFF
        elif filt == 2:
            for i in range(len(row)):
                row[i] = (row[i] + prev[i]) & 0xFF
        elif filt == 3:
            for i in range(len(row)):
                a = row[i-bpp] if i >= bpp else 0
                row[i] = (row[i] + (a + prev[i]) // 2) & 0xFF
        elif filt == 4:
            for i in range(len(row)):
                a = row[i-bpp] if i >= bpp else 0
                b = prev[i]; c = prev[i-bpp] if i >= bpp else 0
                p = a+b-c; pa=abs(p-a); pb=abs(p-b); pc=abs(p-c)
                pr = a if pa<=pb and pa<=pc else (b if pb<=pc else c)
                row[i] = (row[i] + pr) & 0xFF
        rows.append(bytes(row)); prev = bytes(row)

    if bit_depth == 16:
        arr = np.frombuffer(b''.join(rows), dtype=np.dtype('>u2')).reshape(h, w, channels)
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = np.frombuffer(b''.join(rows), dtype=np.uint8).reshape(h, w, channels)
        arr = arr.astype(np.float32) / 255.0

    if channels == 1:   arr = np.concatenate([arr]*3, axis=-1)
    elif channels == 4: arr = arr[..., :3]
    elif channels == 2: arr = np.concatenate([arr[..., :1]]*3, axis=-1)
    return np.clip(arr, 0.0, 1.0)



def pil_to_array(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to float32 array [0,1], preserving 16-bit depth."""
    raw = np.array(img)
    if raw.dtype == np.uint16:
        return (raw.astype(np.float32) / 65535.0)
    elif raw.dtype == np.uint8:
        return (raw.astype(np.float32) / 255.0)
    else:
        # float or other — normalize to 0-1
        r = raw.astype(np.float32)
        mn, mx = r.min(), r.max()
        return (r - mn) / (mx - mn + 1e-9) if mx > mn else r


def array_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert float32 array [0,1] back to PIL Image."""
    clipped = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(clipped, mode="RGB")


def luminance(arr: np.ndarray) -> np.ndarray:
    """Return luminance channel from RGB float array."""
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


# ─────────────────────────────────────────────────────────────
#  Oklab color space
#  Perceptually uniform — sharpening on L only avoids color
#  fringing and gives more natural contrast enhancement.
# ─────────────────────────────────────────────────────────────

def _linear(c: np.ndarray) -> np.ndarray:
    """sRGB → linear RGB (vectorised)."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(c: np.ndarray) -> np.ndarray:
    """Linear RGB → sRGB (vectorised)."""
    return np.where(c <= 0.0031308, c * 12.92,
                    1.055 * np.power(np.clip(c, 0, None), 1.0 / 2.4) - 0.055)


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert float32/64 RGB [0,1] → Oklab [L∈0-1, a,b∈≈-0.5..0.5].
    Returns array of same shape with channels (L, a, b).
    """
    r, g, b = _linear(rgb[..., 0]), _linear(rgb[..., 1]), _linear(rgb[..., 2])
    # RGB → LMS (Oklab M1 matrix)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    # cube root
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    # LMS → Lab (Oklab M2 matrix)
    L =  0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a =  1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bch= 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack([L, a, bch], axis=-1)


def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert Oklab → float RGB [0,1]."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bch= -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return np.clip(np.stack([_srgb(r), _srgb(g), _srgb(bch)], axis=-1), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────
#  À trous Wavelet Sharpening
#  Shift-invariant, no ringing.  Filter kernel selectable per
#  run: Gaussian (B3-spline), Z-Gaussian (sharper), Bilateral
#  (edge-preserving).
# ─────────────────────────────────────────────────────────────

# B3-spline 1D kernel
_B3 = np.array([1/16, 1/4, 3/8, 1/4, 1/16], dtype=np.float64)


def _atrous_smooth_b3(plane: np.ndarray, scale: int) -> np.ndarray:
    """B3-spline à trous smooth at dyadic scale."""
    step = 2 ** scale
    tmp = np.zeros_like(plane)
    out = np.zeros_like(plane)
    for i, k in enumerate(_B3):
        offset = (i - 2) * step
        tmp += k * np.roll(plane, -offset, axis=1)
    for i, k in enumerate(_B3):
        offset = (i - 2) * step
        out += k * np.roll(tmp, -offset, axis=0)
    return out


def _atrous_smooth_gaussian(plane: np.ndarray, scale: int) -> np.ndarray:
    """Gaussian smooth — sigma grows with scale."""
    sigma = 0.85 * (2 ** scale)
    return gaussian_filter(plane, sigma=sigma)


def _atrous_smooth_zgaussian(plane: np.ndarray, scale: int) -> np.ndarray:
    """Z-Gaussian — uses a wider smooth per scale, so each detail plane captures
    a broader frequency band. Results in stronger, crispier sharpening than
    standard Gaussian — most aggressive mode."""
    sigma = 1.4 * (2 ** scale)   # wider than gaussian's 0.85, more energy in detail
    return gaussian_filter(plane, sigma=sigma)


def _atrous_smooth_bilateral(plane: np.ndarray, scale: int) -> np.ndarray:
    """Edge-preserving bilateral-style smooth using iterative Gaussian."""
    sigma_s = 0.85 * (2 ** scale)
    sigma_r = 0.08   # range sigma (edge threshold)
    smooth = gaussian_filter(plane, sigma=sigma_s)
    edge_weight = np.exp(-((plane - smooth) ** 2) / (2 * sigma_r ** 2))
    # Blend: near edges keep original, smooth areas get blurred
    return plane * (1 - edge_weight) + smooth * edge_weight


_SMOOTH_FN = {
    "gaussian":  _atrous_smooth_gaussian,
    "zgaussian": _atrous_smooth_zgaussian,
    "bilateral": _atrous_smooth_bilateral,
    "b3":        _atrous_smooth_b3,
}


def _atrous_decompose(channel: np.ndarray, n_levels: int,
                      filter_method: str = "b3"):
    """Decompose channel into n_levels detail planes + residual."""
    smooth_fn = _SMOOTH_FN.get(filter_method, _atrous_smooth_b3)
    planes = []
    current = channel.astype(np.float64)
    for scale in range(n_levels):
        smoothed = smooth_fn(current, scale)
        planes.append(current - smoothed)
        current = smoothed
    return planes, current


# Dyadic à trous decomposition base sigmas.
# Each level doubles the scale: L1=1.5px, L2=3.0px, L3=6.0px.
# The UI Decomp σ sliders let the user override these per layer.
_WV_SIGMAS = [1.5, 3.0, 6.0]


def estimate_filter_sigma(sharpen: float, level: int,
                          img_arr: np.ndarray = None) -> float:
    """Returns the Decomp sigma for each layer used by Autosize Filter.

    Sub-pixel operating point: base = 0.5px.
    Each layer doubles the sigma (dyadic):
      L1: 0.5px, L2: 1.0px, L3: 2.0px

    sharpen and img_arr are accepted for API compatibility but not used.
    """
    base = 0.5
    return round(base * float(2 ** level), 2)   # 0.5 / 1.0 / 2.0 px


def _fine_scale_snr(img: np.ndarray) -> float:
    """Fine-scale signal-to-noise of the disc — a proxy for how noisy an image is.

    Detail at the finest scale (luma minus a 0.5px Gaussian) has its robust
    spread measured over the whole disc (signal + noise) and, separately, its
    MAD-based noise in a small central patch. Their ratio is high for clean,
    detailed images and ~1 for noisy ones. Returns a large value if no disc.
    """
    if img is None or np.ndim(img) != 3 or img.shape[-1] < 3:
        return 99.0
    lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    peak = float(np.percentile(lum, 99.5))
    disc = lum > max(peak * 0.2, 1e-4)
    if int(disc.sum()) < 400:
        return 99.0
    ys, xs = np.where(disc)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    d1 = lum - gaussian_filter(lum, 0.5)
    patch = d1[max(0, cy - 15):cy + 15, max(0, cx - 15):cx + 15].ravel()
    noise = 1.4826 * np.median(np.abs(patch - np.median(patch)))
    signal = float(np.std(d1[disc]))
    return signal / max(noise, 1e-9)


def estimate_sharpen_filter(sharpen: float, sigma: float, level: int = 0,
                            img: np.ndarray = None) -> float:
    """
    Return the per-layer SharpenFilter value for Autosize / Estimate Filter,
    matching WaveSharp 3.

    Layers 2 and 3 use a fixed 0.100 — Autosize/Estimate leave them at that
    default regardless of the Sharpen amount (WaveSharp 3 only auto-fits Layer 1).

    Layer 1 follows WaveSharp 3's curve — nearly flat with a slight rise, NOT a
    line through the origin: SF = 0.118 + 0.0308 * frac (frac = sharpen/200),
    fitted to WaveSharp's 0.149→0.122 across the sharpen range.

    WaveSharp's Estimate Filter is image-adaptive — it analyzes contrast, edges
    and noise, so a noisier image gets a higher filter. Kepler mirrors that with
    an image-dependent LIFT keyed on the disc's fine-scale SNR (see
    _fine_scale_snr). Calibrated against measured WaveSharp values: a clean image
    (SNR ≳ 1.6) gets no lift (base curve), and the lift ramps up as SNR falls but
    PLATEAUS at +0.021 — WaveSharp's estimate is nearly the same (~0.153–0.155)
    across noisy images of quite different SNR (1.1 vs 0.86), so the lift must
    saturate rather than keep climbing. Without an image (img=None) only the base
    curve is used.

    sigma is accepted for API compatibility but not used for the calculation.
    """
    lvl = min(int(level), 2)
    if lvl >= 1:                      # Layers 2 & 3: fixed default
        return 0.100
    if sharpen <= 0:                  # Layer 1: off when the layer is off
        return 0.0
    frac = min(float(sharpen) / 200.0, 1.0)           # 0..1
    base = 0.118 + 0.0308 * frac
    lift = 0.0
    if img is not None:
        snr = _fine_scale_snr(img)
        lift = float(np.clip(0.042 * (1.6 - snr), 0.0, 0.021))
    return round(min(base + lift, 0.20), 3)


def _smooth_at_scale(plane: np.ndarray, scale: int, fm: str,
                     sigma: float = None,
                     bilateral_radius: float = 2.0,
                     zgauss_factor: float = 1.0) -> np.ndarray:
    """Apply one smoothing step.

    "gaussian"  : true Gaussian at the provided sigma value.
    "zgaussian" : Gaussian at sigma = 1.4 * 2^scale * zgauss_factor.
                  zgauss_factor=0 collapses to Gaussian mode; =1 is full Z-Gaussian.
    "bilateral" : edge-preserving bilateral approximation.
                  bilateral_radius controls the spatial sigma (0=thin, 10=wide).
    "b3"        : dilated B3-spline à trous kernel.
    """
    if fm == "gaussian" and sigma is not None:
        return gaussian_filter(plane.astype(np.float64), sigma=sigma)
    elif fm == "zgaussian":
        # Factor blends between full Z-Gaussian (factor=0) and Gaussian (factor=1).
        # factor=0 → sigma = 1.4 * 2^scale  (full Z-Gaussian, wider/stronger)
        # factor=1 → sigma = Decomp σ value  (identical to Gaussian mode)
        base_s = sigma if sigma is not None else 0.85 * (2 ** scale)
        zg_s   = 1.4 * (2 ** scale)
        s = zg_s + zgauss_factor * (base_s - zg_s)   # interpolate ZG→Gauss
        s = max(s, 0.1)
        return gaussian_filter(plane.astype(np.float64), sigma=s)
    elif fm == "bilateral":
        # Bilateral: Gaussian smoothing in flat areas, edge/limb preservation at edges.
        # Uses Sobel edge magnitude for the range weighting — this correctly detects
        # the planetary limb and belt edges regardless of the decomp sigma scale.
        #
        # bilateral_radius (1-10) → logarithmic threshold scale:
        #   radius=10 → threshold very large → ew≈1 everywhere → identical to Gaussian
        #   radius=1  → threshold small     → ew≈0 at strong edges → edges fully preserved
        #                                     ew≈1 in flat areas → full Gaussian in interior
        sigma_s = sigma if sigma is not None else 0.85 * (2 ** scale)
        plane64 = plane.astype(np.float64)
        smooth  = gaussian_filter(plane64, sigma=sigma_s)
        # Compute Sobel edge magnitude for edge detection
        sx = sobel(plane64, axis=1)
        sy = sobel(plane64, axis=0)
        edge_mag = np.hypot(sx, sy)
        # Robust edge scale: 99th percentile of edge magnitude
        edge_scale = np.percentile(edge_mag, 99.0)
        if edge_scale < 1e-9:
            return smooth  # flat image — just return Gaussian
        # Logarithmic threshold: radius=1→0.3×scale, radius=10→100×scale
        log_threshold = edge_scale * (0.3 * (100.0 / 0.3) ** ((bilateral_radius - 1.0) / 9.0))
        ew = np.exp(-(edge_mag ** 2) / (2.0 * log_threshold ** 2))
        return plane64 * (1.0 - ew) + smooth * ew
    else:
        return _atrous_smooth_b3(plane, scale)


def _background_mask(plane: np.ndarray) -> np.ndarray:
    """Compute a soft foreground mask to suppress sky/background sharpening.

    Planetary images have a bright disk on a near-black sky background.
    Without this mask, high-gain sharpening amplifies sky noise into visible
    specks around the planet.

    Strategy: ramp from 0 (sky) to 1 (planet) using thresholds relative to
    the image peak luma. This is robust regardless of what fraction of the
    image is sky — unlike percentile-based thresholds which fail when sky
    dominates (as in typical drizzled planetary images).

      lo = peak * 1.5%  — ramp start (true sky noise)
      hi = peak * 3.0%  — ramp end   (safely inside even the darkest belts)

    The mask is then smoothed so the limb transition is gradual.

    Returns float64 mask in [0, 1], same shape as plane.
    """
    p = plane.astype(np.float64)
    lmax = p.max()
    if lmax < 0.01:
        return np.ones_like(p)   # near-blank image — no masking

    lo = lmax * 0.015
    hi = lmax * 0.030
    mask = np.clip((p - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    # Smooth so limb transition is gradual
    sigma = max(plane.shape[0] // 120, 3)
    mask = gaussian_filter(mask, sigma=sigma)

    mask_max = mask.max()
    if mask_max > 1e-6:
        mask = np.clip(mask / mask_max, 0.0, 1.0)

    return mask


def _sharpen_plane(plane: np.ndarray, levels: list, fm: str,
                   pre_fft: list = None,
                   sharp_filter = 0.0,
                   bg_mask: np.ndarray = None,
                   pre_smooth: float = 0.0,
                   power_fn: float = 1.0,
                   bilateral_radius: float = 2.0,
                   zgauss_factor: float = 1.0,
                   bilateral_layers: list = None) -> np.ndarray:
    # bilateral_layers: list of bool per layer; if None all layers use fm
    # when fm=="bilateral", layers where bilateral_layers[i]==False use gaussian
    """
    Multi-scale wavelet / USM sharpening.

    fm=="gaussian": Multi-scale USM — each layer: detail = base - gaussian(base, σ).
                    Gain = sharpen/100 (slider 0-200, unity at 100).
                    Sigma comes from Decomp σ sliders (default dyadic: 1/2/4 px).
                    Standard Gaussian mode.
    fm=="b3" etc:   Iterative B3 à trous cascade. Gain = sharpen/75.
    """
    has_pre = pre_fft and any(p is not None for p in pre_fft)
    working = gaussian_filter(plane.astype(np.float64), sigma=pre_smooth) \
              if pre_smooth > 0 else plane.astype(np.float64)
    sigmas = [s for _, _, s in levels]
    # Gaussian mode: gain_div calibrated for Kepler's operating point.
    # sigma≈0.5px (sub-pixel kernel) with gain≈350x at slider=200.
    # gain_div_L1 = 200/350 = 0.5714 so slider 200 -> gain 350x on sigma=0.5 USM.
    # Coarser layers scale by 4x per layer — gives clearly visible effect at L2 and L3:
    #   L1 (i=0): gain_div=0.5714   (sigma=0.5px, gain=350x at slider 200)
    #   L2 (i=1): gain_div=2.2857   (sigma=1.0px, gain=87.5x at slider 200)
    #   L3 (i=2): gain_div=9.1429   (sigma=2.0px, gain=21.9x at slider 200)
    # Non-gaussian modes keep a flat gain_div=75 (unchanged from prior behavior).
    # Gaussian mode: gain calibrated so slider 200 = 350x gain on L1.
    # Z-Gaussian: factor=0 is ~3.3x stronger than Gaussian (sliders need ~30% of
    #             Gaussian values). factor=1 is identical to Gaussian.
    #             gain_div scales as gaussian_gain_div * (0.30 + 0.70*factor).
    # Other modes: flat gain_div=75.
    if fm == "gaussian":
        _base_gain_div = 200.0 / 350.0           # 0.5714
    elif fm == "zgaussian":
        # factor=0 → full Z-Gaussian: sliders need ~33% of Gaussian values
        # factor=1 → identical to Gaussian: gain_div = gaussian value
        _zg_mult = 2.173 * (1.0 - zgauss_factor) + 1.0 * zgauss_factor
        _base_gain_div = (200.0 / 350.0) * _zg_mult
    elif fm == "bilateral":
        # Bilateral uses the same gain_div as Gaussian so that at radius=10
        # (pure Gaussian behavior) the slider values produce identical output.
        _base_gain_div = 200.0 / 350.0           # 0.5714
    else:
        _base_gain_div = 75.0
    # Scale factor per layer: controls how much L2/L3 contribute relative to L1.
    # scale=4 gives meaningful visible effect at all three layers.
    _gaussian_scale = 4.0
    # Resolve sharp_filter: list = per-layer; scalar = broadcast to all layers
    _sf_list = sharp_filter if isinstance(sharp_filter, (list, tuple))                else [float(sharp_filter)] * len(levels)

    if not has_pre:
        # Build cascade
        if fm == "gaussian":
            # Gaussian bandpass decomposition:
            #   L1 (i=0): detail = base - gauss(base, s0)            <- USM
            #   L2 (i=1): detail = gauss(base, s0) - gauss(base, s1) <- bandpass
            #   L3 (i=2): detail = gauss(base, s1) - gauss(base, s2) <- bandpass
            # Cumulative USM for L2+ re-sharpened fine content already boosted
            # by L1, causing ~4x over-contribution. Bandpass prevents this.
            base = working
            smoothed_layers = [gaussian_filter(base, s) for s in sigmas]
        else:
            # Iterative a trous: B3/zgaussian/bilateral at dyadic scale indices
            c = [working]
            for i in range(len(levels)):
                _lfm = fm if (bilateral_layers is None or bilateral_layers[i]) else "gaussian"
                c.append(_smooth_at_scale(c[-1], i, _lfm, sigmas[i],
                                          bilateral_radius=bilateral_radius,
                                          zgauss_factor=zgauss_factor))

        out = plane.astype(np.float64).copy()
        for i, (sharpen, denoise, sigma) in enumerate(levels):
            if sharpen == 0:
                continue
            _sf = _sf_list[i]
            if fm == "gaussian":
                if i == 0:
                    detail = base - smoothed_layers[0]          # L1: USM
                else:
                    detail = smoothed_layers[i-1] - smoothed_layers[i]  # L2+: bandpass
                # SharpenFilter: DoG boost on the extraction kernel.
                if _sf > 0.0:
                    dog_base = base if i == 0 else smoothed_layers[i-1]
                    dog = smoothed_layers[i] - gaussian_filter(dog_base, sigma * 1.6)
                    detail = detail + _sf * dog
            else:
                detail = c[i] - c[i + 1]             # wavelet bandpass
                _lfm = fm if (bilateral_layers is None or bilateral_layers[i]) else "gaussian"
                if _sf > 0.0:
                    prev_sigma = sigmas[max(i - 1, 0)]
                    sf_smooth = _smooth_at_scale(detail, max(i - 1, 0), _lfm, prev_sigma,
                                                 bilateral_radius=bilateral_radius,
                                                 zgauss_factor=zgauss_factor)
                    detail = detail + _sf * (detail - sf_smooth)

            thresh = (denoise / 100.0) * 0.004
            if thresh > 0:
                abs_d = np.abs(detail)
                detail = np.where(abs_d > thresh, np.sign(detail) * (abs_d - thresh), 0.0)

            # Power function: sign(d) * |d|^exp — applied after threshold, before gain
            if power_fn != 1.0:
                abs_d = np.abs(detail)
                detail = np.sign(detail) * np.power(abs_d, power_fn)

            gain_div = _base_gain_div * (_gaussian_scale ** i) if fm in ("gaussian", "zgaussian", "bilateral") else _base_gain_div
            amp = (sharpen / gain_div) * detail
            if bg_mask is not None:
                amp *= bg_mask
            out += amp

        return out

    # ── PRE-layer FFT path (always iterative) ────────────────────────────────
    out = plane.astype(np.float64).copy()
    c_i = working.copy()

    for i, (sharpen, denoise, sigma) in enumerate(levels):
        if pre_fft and i < len(pre_fft) and pre_fft[i] is not None:
            fs, fw, fc = pre_fft[i]
            if fs < 100.0:
                tmp = np.stack([c_i] * 3, axis=-1).astype(np.float32)
                tmp = fft_denoise(tmp, fft_start=fs, fft_width=fw, fft_curve=fc)
                c_i = tmp[..., 0].astype(np.float64)

        _lfm = fm if (bilateral_layers is None or (
            bilateral_layers is not None and i < len(bilateral_layers) and bilateral_layers[i])) else "gaussian"
        c_next = gaussian_filter(c_i, sigma) if _lfm == "gaussian" \
                 else _smooth_at_scale(c_i, i, _lfm, sigma,
                                       bilateral_radius=bilateral_radius,
                                       zgauss_factor=zgauss_factor)
        detail = c_i - c_next

        if sharpen > 0:
            _sf = _sf_list[i]
            if _sf > 0.0:
                if fm == "gaussian":
                    dog = c_next - gaussian_filter(c_i, sigma * 1.6)
                    detail = detail + _sf * dog
                else:
                    prev_sigma = sigmas[max(i - 1, 0)]
                    sf_smooth = _smooth_at_scale(detail, max(i - 1, 0), fm, prev_sigma)
                    detail = detail + _sf * (detail - sf_smooth)

            thresh = (denoise / 100.0) * 0.004
            if thresh > 0:
                abs_d = np.abs(detail)
                detail = np.where(abs_d > thresh, np.sign(detail) * (abs_d - thresh), 0.0)

            # Power function: sign(d) * |d|^exp — applied after threshold, before gain
            if power_fn != 1.0:
                abs_d = np.abs(detail)
                detail = np.sign(detail) * np.power(abs_d, power_fn)

            gain_div = _base_gain_div * (_gaussian_scale ** i) if fm in ("gaussian", "zgaussian", "bilateral") else _base_gain_div
            amp = (sharpen / gain_div) * detail
            if bg_mask is not None:
                amp *= bg_mask
            out += amp

        c_i = c_next

    return out


def _extract_luma(img: np.ndarray, color_model: str):
    """Extract the luma/lightness channel for sharpening, return (L, aux).
    aux holds whatever is needed to reconstruct the full image afterwards."""
    if color_model == "hsl":
        import colorsys
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        cmax = np.max(img, axis=-1); cmin = np.min(img, axis=-1)
        L = (cmax + cmin) / 2.0
        return L, ("hsl", img)
    elif color_model == "hsv":
        V = np.max(img, axis=-1)
        return V, ("hsv", img)
    else:  # oklab (default)
        lab = rgb_to_oklab(img)
        return lab[..., 0].copy(), ("oklab", lab)


def _reconstruct(L_sharp: np.ndarray, aux, color_model: str) -> np.ndarray:
    """Reconstruct RGB from sharpened luma + original aux data."""
    tag, data = aux
    if tag == "oklab":
        out = data.copy(); out[..., 0] = np.clip(L_sharp, 0.0, 1.0)
        return oklab_to_rgb(out).astype(np.float32)
    elif tag == "hsl":
        # Shift L by the delta, keep H and S
        orig_img = data
        r, g, b  = orig_img[..., 0], orig_img[..., 1], orig_img[..., 2]
        cmax = np.max(orig_img, axis=-1); cmin = np.min(orig_img, axis=-1)
        L_orig = (cmax + cmin) / 2.0
        delta  = np.clip(L_sharp, 0.0, 1.0) - L_orig
        # Apply delta as brightness shift to RGB (preserves H/S)
        return np.clip(orig_img + delta[..., np.newaxis], 0.0, 1.0).astype(np.float32)
    else:  # hsv
        orig_img = data
        V_orig   = np.max(orig_img, axis=-1)
        scale    = np.clip(L_sharp, 0.0, 1.0) / np.maximum(V_orig, 1e-6)
        return np.clip(orig_img * scale[..., np.newaxis], 0.0, 1.0).astype(np.float32)


def _mean_saturation(rgb: np.ndarray) -> float:
    """Mean HSV saturation (max-min)/max over the planetary disc only.

    Measured over the disc (luminance above 15% of the 99.5th-percentile peak —
    the same threshold Kepler uses to detect the disc) and not the whole frame:
    Chroma Denoise raises disc saturation while lowering the noisy sky's, so a
    whole-frame mean would look flat and hide the drift we need to correct. A flat
    over-the-disc average also tracks perceived saturation better than a
    brightness-weighted one, which over-favors the low-saturation bright center."""
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    peak = float(np.percentile(lum, 99.5))
    disc = lum > max(peak * 0.15, 1e-4)
    if int(disc.sum()) < 16:
        return float(s.mean())
    return float(s[disc].mean())


def _apply_chroma_denoise(img: np.ndarray, radius: float) -> np.ndarray:
    """
    Chroma Denoise — suppress the color speckle RGB/LRGB wavelet sharpening
    amplifies, without softening luminance detail or shifting overall saturation.

    A straightforward chrominance blur in oklab: leave the L (luminance) channel
    untouched and smooth the a*/b* chroma channels. No separate hue/magnitude
    manipulation — just the two chroma components — which keeps the routine simple
    and free of the edge artifacts a polar (hue-angle) treatment introduces.

    Two refinements keep it clean on planetary discs:

    • Sky-excluding (normalized) convolution. A plain a*/b* blur averages the
      disc's chroma with the sky's ZERO chroma across the limb, draining
      saturation in a band that reads as a faint colored fringe ~20% into the
      disc. The blur is therefore weighted by a soft disc mask (luminance above
      ~30% of peak): sky pixels carry ~zero weight and cannot dilute the limb,
      while the disc — limb included — is smoothed normally.

    • Saturation restore. Averaging noisy chroma vectors shrinks their mean length
      (the random-hue noise cancels toward gray), so a bare a*/b* blur desaturates
      the disc. A single global scale on the chroma returns the result's mean disc
      saturation to the input's — one scalar, applied uniformly, not a per-pixel
      hue/saturation edit. A few fixed-point steps converge (HSV saturation is not
      quite linear in the chroma scale).

    radius: Gaussian sigma in pixels applied to the chroma channels.
    """
    src = np.clip(img, 0.0, 1.0).astype(np.float64)
    lab = rgb_to_oklab(src)
    a, b = lab[..., 1], lab[..., 2]

    # Soft disc weight — excludes the dark sky so it can't dilute the limb.
    lum  = 0.299 * src[..., 0] + 0.587 * src[..., 1] + 0.114 * src[..., 2]
    peak = float(np.percentile(lum, 99.5))
    w    = np.clip(lum / max(0.30 * peak, 1e-6), 0.0, 1.0)
    gw   = gaussian_filter(w, sigma=radius) + 1e-6

    # Normalized (sky-excluded) blur of the two chroma channels.
    a_new = gaussian_filter(a * w, sigma=radius) / gw
    b_new = gaussian_filter(b * w, sigma=radius) / gw
    # In the deep sky (negligible weight) keep the original chroma — avoids 0/0
    # amplification and leaves the near-black background untouched.
    keep  = gaussian_filter(w, sigma=radius) < 0.05
    a_new = np.where(keep, a, a_new)
    b_new = np.where(keep, b, b_new)

    # Restore the input's mean saturation (WaveSharp 3 is saturation-neutral).
    target = _mean_saturation(src)
    if target > 1e-5:
        k = 1.0
        for _ in range(4):
            lab[..., 1] = a_new * k
            lab[..., 2] = b_new * k
            out = np.clip(oklab_to_rgb(lab), 0.0, 1.0)
            cur = _mean_saturation(out)
            if cur < 1e-6:
                break
            k *= target / cur
        return out.astype(np.float32)

    lab[..., 1] = a_new
    lab[..., 2] = b_new
    return np.clip(oklab_to_rgb(lab), 0.0, 1.0).astype(np.float32)


def wavelet_sharpen(
    img_array: np.ndarray,
    levels: list,
    filter_method: str = "gaussian",
    filter_enabled: bool = True,
    color_model: str = "oklab",
    convolve: str = "L",
    pre_fft: list = None,
    sharp_filter: float = 0.0,
    pre_smooth: float = 0.0,
    power_fn: float = 1.0,
    bilateral_radius: float = 2.0,
    zgauss_factor: float = 1.0,
    bilateral_layers: list = None,
    color_denoise: float = 0.0,
) -> np.ndarray:
    """
    À trous wavelet sharpening — color-safe.

    pre_smooth : Gaussian sigma applied to the luma channel BEFORE decomposition.
                 Eliminates sub-pixel moire by removing noise below the finest
                 decomposition scale before the wavelet cascade begins.
                 Default 0 (off).
    """
    fm  = filter_method if filter_enabled else "gaussian"
    img = img_array.astype(np.float64)

    # Compute sky-suppression mask from luma. Prevents amplification of sky noise
    # at high gain. Uses peak-relative thresholds so dark belt pixels are not masked.
    lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    bg_mask = _background_mask(lum)

    if convolve == "RGB":
        result = np.empty_like(img)
        for c in range(3):
            result[..., c] = np.clip(
                _sharpen_plane(img[..., c], levels, fm, pre_fft, sharp_filter, bg_mask, pre_smooth, power_fn,
                                bilateral_radius, zgauss_factor, bilateral_layers), 0.0, 1.0)
        if color_denoise > 0.0:
            result = _apply_chroma_denoise(result, color_denoise)
        return result.astype(np.float32)

    if convolve == "LRGB":
        # LRGB mode sharpens each RGB channel independently at full gain.
        # The previous two-pass (L delta → pass1 → RGB sharpen on pass1) diluted
        # the gain because pass2 was sharpening an already-boosted signal.
        result = np.empty_like(img)
        for c in range(3):
            result[..., c] = np.clip(
                _sharpen_plane(img[..., c], levels, fm, pre_fft, sharp_filter, bg_mask, pre_smooth, power_fn,
                                bilateral_radius, zgauss_factor, bilateral_layers), 0.0, 1.0)
        if color_denoise > 0.0:
            result = _apply_chroma_denoise(result, color_denoise)
        return result.astype(np.float32)

    # convolve == "L"
    L, aux  = _extract_luma(img, color_model)
    L_sharp = _sharpen_plane(L, levels, fm, pre_fft, sharp_filter, bg_mask, pre_smooth, power_fn,
                            bilateral_radius, zgauss_factor, bilateral_layers)
    return _reconstruct(L_sharp, aux, color_model)


def auto_wavelet_settings(img_array: np.ndarray, auto_strength: float, snr_target: float):
    """
    Analyze image and return wavelet settings scaled to user's controls.
    auto_strength 0-200: maps directly to slider values (100=normal, 200=max)
    snr_target 10-50:    maps to sharpen amount — higher = more aggressive

    Returns: [(slider1,sharpen1,denoise1), (slider2,...), (slider3,...)]
    """
    lum = luminance(img_array)

    # Noise estimate from finest wavelet plane
    planes, _ = _atrous_decompose(lum, 3, "b3")
    noise_est = np.std(planes[0]) * 1.4826
    noise_est = max(noise_est, 1e-6)

    # Denoise proportional to noise floor
    dn1 = int(np.clip(noise_est * 4000,  2, 40))
    dn2 = int(np.clip(noise_est * 5000,  2, 50))
    dn3 = int(np.clip(noise_est * 6000,  3, 60))

    # Slider: directly driven by auto_strength (0-200 range)
    sl  = int(np.clip(auto_strength, 0, 200))
    sl1 = sl
    sl2 = max(10, int(sl * 0.9))
    sl3 = max(10, int(sl * 0.7))

    # Sharpen: driven by snr_target (10-50) — maps to 0-500 sharpen range
    # snr_target=10 → very gentle,  snr_target=50 → extreme
    sh_scale = (snr_target - 10) / 40.0   # 0.0 to 1.0
    sh1 = int(sh_scale * 500)
    sh2 = int(sh_scale * 400)
    sh3 = int(sh_scale * 300)

    return [(sl1, sh1, dn1), (sl2, sh2, dn2), (sl3, sh3, dn3)]



# ─────────────────────────────────────────────────────────────
#  FFT Denoise
# ─────────────────────────────────────────────────────────────

def _window_func(h: int, w: int, kind: str) -> np.ndarray:
    """Build a 2D window function."""
    if kind == "hann":
        wy = np.hanning(h)
        wx = np.hanning(w)
    elif kind == "hamming":
        wy = np.hamming(h)
        wx = np.hamming(w)
    elif kind == "blackman":
        wy = np.blackman(h)
        wx = np.blackman(w)
    else:  # none
        return np.ones((h, w), dtype=np.float32)
    return np.outer(wy, wx).astype(np.float32)


def _soft_threshold(spectrum: np.ndarray, threshold: float) -> np.ndarray:
    """Soft thresholding in frequency domain."""
    mag = np.abs(spectrum)
    phase = np.angle(spectrum)
    mag_thresh = np.maximum(mag - threshold, 0)
    return mag_thresh * np.exp(1j * phase)


def _fft_curve_exponent(curve_amount: float) -> float:
    """Map curve slider 0-100 to power exponent k.
    k < 1 = concave-down / gentle (left);  k=1 = linear (center);  k > 1 = concave-up / aggressive (right).
    Formula: k = 0.3 * (10/3)^(curve_amount/50)
    slider=0 → k≈0.30, slider=50 → k=1.0, slider=100 → k≈3.33
    """
    return 0.3 * ((10.0 / 3.0) ** (float(curve_amount) / 50.0))


def fft_denoise(
    img_array: np.ndarray,
    fft_start: float,
    fft_width: float,
    fft_curve,   # float 0-100 (continuous slider) or legacy str for back-compat
) -> np.ndarray:
    """
    FFT low-pass filter.

    fft_start  : 0-100, percentage of Nyquist where attenuation begins.
    fft_width  : 0-100, width of the rolloff as percentage of Nyquist.
    fft_curve  : float 0-100 — curve shape slider.
                 0 = concave-down / gentle (less removal)
                 50 = linear
                 100 = concave-up / aggressive (more removal)
                 Also accepts legacy strings for back-compatibility.
    """
    # Legacy string back-compat
    if isinstance(fft_curve, str):
        _legacy = {"cosine": 25, "linear": 50, "gaussian": 75, "hard": 100}
        fft_curve = float(_legacy.get(fft_curve, 50))

    start_frac = np.clip(fft_start / 100.0, 0.0, 1.0)
    width_frac  = np.clip(fft_width  / 100.0, 0.0, 1.0)
    if start_frac >= 1.0:
        return img_array.copy()

    h, w = img_array.shape[:2]
    cy, cx   = h // 2, w // 2
    Y2, X2   = np.ogrid[:h, :w]
    dist     = np.sqrt((X2 - cx)**2 + (Y2 - cy)**2)
    max_dist = float(min(cx, cy)) + 1e-10   # Nyquist = half min dimension, not corner
    nd       = dist / max_dist

    cutoff_frac = start_frac + width_frac
    t = np.clip((nd - start_frac) / max(width_frac, 1e-6), 0.0, 1.0)
    in_ramp = (nd >= start_frac) & (nd < cutoff_frac)
    above   = nd >= cutoff_frac

    k = _fft_curve_exponent(fft_curve)
    filter_2d = np.ones((h, w), dtype=np.float64)
    filter_2d[in_ramp] = (1.0 - t[in_ramp]) ** k
    filter_2d[above]   = 0.0

    result = np.empty_like(img_array)
    for c in range(3):
        channel  = img_array[..., c].astype(np.float64)
        Fshift   = np.fft.fftshift(np.fft.fft2(channel))
        Fshift  *= filter_2d
        denoised = np.real(np.fft.ifft2(np.fft.ifftshift(Fshift)))
        result[..., c] = np.clip(denoised, 0.0, 1.0)

    return result


def build_fft_filter_curve(fft_start: float, fft_width: float, fft_curve,
                            n_bins: int = 64) -> np.ndarray:
    """Return a 1-D array (length n_bins) of filter values 0..1 for display."""
    if isinstance(fft_curve, str):
        _legacy = {"cosine": 25, "linear": 50, "gaussian": 75, "hard": 100}
        fft_curve = float(_legacy.get(fft_curve, 50))
    nd = np.linspace(0.0, 1.0, n_bins)
    start_frac  = np.clip(fft_start / 100.0, 0.0, 1.0)
    width_frac  = np.clip(fft_width  / 100.0, 0.0, 1.0)
    cutoff_frac = start_frac + width_frac
    t = np.clip((nd - start_frac) / max(width_frac, 1e-6), 0.0, 1.0)
    in_ramp = (nd >= start_frac) & (nd < cutoff_frac)
    above   = nd >= cutoff_frac
    k = _fft_curve_exponent(fft_curve)
    filt = np.ones(n_bins, dtype=np.float64)
    filt[in_ramp] = (1.0 - t[in_ramp]) ** k
    filt[above]   = 0.0
    return filt


def compute_fft_power(img_array: np.ndarray) -> np.ndarray:
    """Return 1D radially averaged power spectrum (64 bins) for display."""
    lum = luminance(img_array)
    h, w = lum.shape
    window = _window_func(h, w, "hann")
    F = np.fft.fftshift(np.fft.fft2(lum * window))
    power = np.log1p(np.abs(F) ** 2)
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
    max_dist = min(cx, cy)
    bins = 64
    spectrum = np.zeros(bins)
    counts = np.zeros(bins)
    for i in range(bins):
        r0 = int(i * max_dist / bins)
        r1 = int((i + 1) * max_dist / bins)
        mask = (dist >= r0) & (dist < r1)
        if mask.any():
            spectrum[i] = np.mean(power[mask])
            counts[i] = mask.sum()
    # Normalize
    if spectrum.max() > 0:
        spectrum /= spectrum.max()
    return spectrum


# ─────────────────────────────────────────────────────────────
#  RGB Balance & Saturation
# ─────────────────────────────────────────────────────────────

def _rgb_to_hsl(r, g, b):
    """Vectorized RGB to HSL."""
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    L = (cmax + cmin) / 2.0
    S = np.where(delta == 0, 0, delta / (1 - np.abs(2 * L - 1) + 1e-10))

    H = np.zeros_like(r)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)
    H[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    H[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
    H[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
    H = H / 6.0
    return H, S, L


def _hsl_to_rgb(H, S, L):
    """Vectorized HSL to RGB."""
    C = (1 - np.abs(2 * L - 1)) * S
    X = C * (1 - np.abs((H * 6) % 2 - 1))
    m = L - C / 2

    H6 = (H * 6).astype(int) % 6
    R = np.select([H6==0, H6==1, H6==2, H6==3, H6==4, H6==5], [C,X,0*C,0*C,X,C])
    G = np.select([H6==0, H6==1, H6==2, H6==3, H6==4, H6==5], [X,C,C,X,0*C,0*C])
    B = np.select([H6==0, H6==1, H6==2, H6==3, H6==4, H6==5], [0*C,0*C,X,C,C,X])
    return R + m, G + m, B + m


def align_rgb_channels(img_array: np.ndarray, reference: str = "G",
                       upsample: int = 10, max_shift: float = 40.0):
    """
    Register the R and B channels to a reference channel and shift them back into
    line — corrects chromatic misregistration (atmospheric dispersion, or an
    RGB-align step skipped in the stacker) that otherwise shows as colored fringes
    on one side of the disc and its opposite on the other.

    The shift for each channel is found by phase cross-correlation against the
    reference (green by default — the middle wavelength, usually the sharpest with
    the best SNR), measured to sub-pixel precision (upsample=10 → 0.1 px), then
    applied with cubic resampling. Runs BEFORE sharpening so the fringes are gone
    before sharpening would amplify them.

    max_shift clamps the correction so a low-signal channel that phase-correlates
    to garbage cannot throw the disc across the frame. A mono image (all channels
    equal) yields zero shifts and is returned untouched.

    Returns (aligned_img, shifts) where shifts maps 'R'/'G'/'B' → (dy, dx) in px.
    """
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import shift as _nd_shift

    a = img_array.astype(np.float64)
    if a.ndim != 3 or a.shape[-1] < 3:
        return img_array, {"R": (0.0, 0.0), "G": (0.0, 0.0), "B": (0.0, 0.0)}

    ref_idx = {"R": 0, "G": 1, "B": 2}.get(reference.upper(), 1)
    ref = a[..., ref_idx]
    out = a.copy()
    shifts = {"R": (0.0, 0.0), "G": (0.0, 0.0), "B": (0.0, 0.0)}

    for c, name in ((0, "R"), (1, "G"), (2, "B")):
        if c == ref_idx:
            continue
        # Identical channel → nothing to do (mono, or already-merged luminance).
        if np.array_equal(a[..., c], ref):
            continue
        try:
            sh, _err, _phase = phase_cross_correlation(
                ref, a[..., c], upsample_factor=upsample)
        except Exception:
            continue
        dy = float(np.clip(sh[0], -max_shift, max_shift))
        dx = float(np.clip(sh[1], -max_shift, max_shift))
        if dy == 0.0 and dx == 0.0:
            continue
        out[..., c] = _nd_shift(a[..., c], (dy, dx), order=3,
                                mode="constant", cval=0.0)
        shifts[name] = (dy, dx)

    return np.clip(out, 0.0, 1.0).astype(np.float32), shifts


def apply_color_adjustments(
    img_array: np.ndarray,
    r_gain: float, g_gain: float, b_gain: float,
    r_gamma: float, g_gamma: float, b_gamma: float,
    saturation: float,
    vibrance: float,
    hue_rotation: float,
    brightness: float,
    contrast: float,
    r_black: float = 0.0,
    g_black: float = 0.0,
    b_black: float = 0.0,
) -> np.ndarray:
    """Apply per-channel black point, gain, gamma, saturation, vibrance, hue, brightness, contrast.

    Black point is applied first: out = (in - black) / (1 - black), lifting
    the shadow floor per channel. This is the standard levels stretch and is
    what brings diverging histogram shadow edges together.
    """
    result = img_array.copy().astype(np.float64)

    # Per-channel black point (levels stretch: lift shadow floor)
    for c, bp in enumerate([r_black, g_black, b_black]):
        if bp > 0.0:
            span = max(1.0 - bp, 1e-6)
            result[..., c] = np.clip((result[..., c] - bp) / span, 0.0, 1.0)

    # Per-channel gamma
    eps = 1e-8
    result[..., 0] = np.power(np.clip(result[..., 0], eps, 1), 1.0 / max(r_gamma, eps))
    result[..., 1] = np.power(np.clip(result[..., 1], eps, 1), 1.0 / max(g_gamma, eps))
    result[..., 2] = np.power(np.clip(result[..., 2], eps, 1), 1.0 / max(b_gamma, eps))

    # Per-channel gain
    result[..., 0] *= r_gain
    result[..., 1] *= g_gain
    result[..., 2] *= b_gain

    # Brightness
    result += brightness

    # Contrast (pivot at 0.5)
    result = (result - 0.5) * (1.0 + contrast) + 0.5

    result = np.clip(result, 0, 1)

    # Saturation
    lum = luminance(result)
    lum3 = lum[..., np.newaxis]
    result = lum3 + (result - lum3) * saturation

    # Vibrance (selective saturation — boost less-saturated colors more)
    H, S, L = _rgb_to_hsl(result[..., 0], result[..., 1], result[..., 2])
    vib_boost = (vibrance - 1.0) * (1.0 - S)
    S_new = np.clip(S + vib_boost * S, 0, 1)
    rr, gg, bb = _hsl_to_rgb(H, S_new, L)
    result[..., 0] = rr; result[..., 1] = gg; result[..., 2] = bb

    # Hue rotation
    if abs(hue_rotation) > 0.001:
        H, S, L = _rgb_to_hsl(result[..., 0], result[..., 1], result[..., 2])
        H = (H + hue_rotation / 360.0) % 1.0
        rr, gg, bb = _hsl_to_rgb(H, S, L)
        result[..., 0] = rr; result[..., 1] = gg; result[..., 2] = bb

    return np.clip(result, 0, 1).astype(np.float32)


def auto_balance(img_array: np.ndarray):
    """Full per-channel levels balance — aligns both shadow and highlight
    edges of R, G, B in the histogram simultaneously.

    Uses green as the reference channel. Computes black point (0.5th percentile)
    and white point (99.5th percentile) per channel, then returns the gain and
    black point needed so each channel maps [black..white] → [0..1] with the
    same effective range as green.

    For heavily filtered images (e.g. solar H-alpha or white-light filter) where
    one or more channels carry negligible signal, channels whose highlight is
    below 10% of the brightest channel are left at gain=1.0 to avoid amplifying
    noise.

    Returns:
        r_gain, g_gain, b_gain   — multipliers  (e.g. 1.23 → slider 123)
        r_black, g_black, b_black — black points 0-1 (e.g. 0.05 → slider 5)
    """
    blacks = np.array([np.percentile(img_array[..., c], 0.5) for c in range(3)])
    whites = np.array([np.percentile(img_array[..., c], 99.5) for c in range(3)])
    spans  = np.maximum(whites - blacks, 1e-6)

    # Reference is the brightest channel (not always green —
    # for solar/narrowband images the dominant channel may be R or B).
    ref_c   = int(np.argmax(whites))
    ref_span = spans[ref_c]
    max_white = float(whites.max())

    gains = []
    for c in range(3):
        if whites[c] < max_white * 0.10:
            gains.append(1.0)   # near-empty channel — leave alone
        elif c == ref_c:
            gains.append(1.0)   # reference channel — no scaling needed
        else:
            gains.append(float(np.clip(ref_span / spans[c], 0.1, 8.0)))

    r_gain, g_gain, b_gain = gains

    r_black = float(np.clip(blacks[0], 0.0, 0.5))
    g_black = float(np.clip(blacks[1], 0.0, 0.5))
    b_black = float(np.clip(blacks[2], 0.0, 0.5))

    return r_gain, g_gain, b_gain, r_black, g_black, b_black


def auto_white_balance(img_array: np.ndarray):
    """Per-channel gain stretch so each channel's bright highlights match,
    normalizing color cast while preserving overall brightness.
    Operates only on planet disk pixels (ignores black sky background).
    Returns (r_gain, g_gain, b_gain, r_gamma, g_gamma, b_gamma)
    as slider units (gains 0-200, gammas 50-200)."""

    # Build a disk mask using luminance threshold (same 15%-of-peak as De-rind)
    lum = 0.299 * img_array[...,0] + 0.587 * img_array[...,1] + 0.114 * img_array[...,2]
    lum_max = float(np.percentile(lum, 99.5))
    disk_thresh = max(lum_max * 0.15, 0.01)
    disk_mask = lum > disk_thresh

    # If mask is empty (no planet found) fall back to full image
    if disk_mask.sum() < 100:
        disk_mask = np.ones(lum.shape, dtype=bool)

    # Compute 99th-percentile highlight for each channel on disk pixels only
    whites = []
    for c in range(3):
        whites.append(float(np.percentile(img_array[..., c][disk_mask], 99)))

    # Normalize relative to the brightest channel (not always green —
    # for solar/narrowband images the dominant channel may be R or B).
    # Channels below 10% of the brightest are left at gain=1.0.
    max_white = max(whites)
    ref_white = max(max_white, 1e-6)
    raw_gains = []
    for w in whites:
        if w < max_white * 0.10:
            raw_gains.append(1.0)
        else:
            raw_gains.append(ref_white / max(w, 1e-6))

    # Clamp so no channel is pushed more than 2× brighter or 0.5× darker
    gains = [float(np.clip(g, 0.5, 2.0)) for g in raw_gains]

    # No gamma adjustment — just color balance
    gammas = [1.0, 1.0, 1.0]

    return tuple(gains), tuple(gammas)


# ─────────────────────────────────────────────────────────────
#  Deringing / Onion-Rind Removal
# ─────────────────────────────────────────────────────────────

def _derind_mask(
    lum: np.ndarray,
    edge: float,
    smooth: float,
    gap_width: float,
    gap_angle: float,
    saturn_mode: bool,
    feather: float = 0.0,
    inset: float = 0.0,
    ref_lum: np.ndarray = None,
) -> np.ndarray:
    """
    Build the De-rind active mask — a soft band around the planetary limb.

    edge      : outer extent in pixels (how far outside the disk the mask reaches)
    feather   : inward falloff in pixels (mask tapers to zero this far inside limb)
    smooth    : Gaussian sigma to broaden mask both ways (removes discontinuities)
    gap_width : arc start angle in degrees, 0=top, +CW, -CCW (0 = full circle)
    gap_angle : arc end angle in degrees, 0=top, +CW, -CCW (0 = full circle)
    saturn_mode : detect and exclude ring structures outside the disk from the mask,
                  ensuring only the planet disk limb is corrected even when rings
                  extend beyond the limb on one or both sides
    inset     : pull mask band inward from limb by this many px (0 = starts at limb)
    """
    from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, gaussian_filter
    import numpy as np

    h, w = lum.shape

    # ── Detect planet disk via luminance threshold ──────────────────
    # Use ref_lum (original unsharpened image) when available — sharpening halos
    # push the detected boundary inward. The raw image gives the true visual limb.
    detect_lum  = ref_lum if ref_lum is not None else lum
    lum_max     = float(np.percentile(detect_lum, 99.5))
    disk_thresh = max(lum_max * 0.15, 0.01)
    disk_mask   = (detect_lum > disk_thresh).astype(bool)

    # ── Saturn mode: replace disk_mask with PCA-circular mask ───────
    # When rings are open (eigenvalue ratio ≥ 1.5) the threshold disk_mask
    # includes the rings, causing distance ramps to fire at the ring boundary
    # instead of the planet limb. We replace disk_mask with a circle whose
    # radius = PCA minor-axis spread × 2.2, which reliably matches the planet
    # disk diameter independent of ring tilt angle or ring brightness.
    # set by the PCA fit below when Saturn mode succeeds
    _ring_perp = _ring_along = _ring_r_eq = _ring_r_pol = None
    if saturn_mode:
        # Saturn mode: replace disk_mask with a fitted oriented ellipse that
        # follows the planet's oblate disk shape while ignoring ring contamination.
        # Rings are excluded — fit the disk and extrapolate
        # the mask continuously over the ring area as if rings were not there.
        #
        # Method: PCA of bright pixels gives major/minor axes. If rings contaminate
        # the major axis (aspect > 1.3), cap it at semi_minor * 1.1 (Saturn oblateness).
        # The ellipse is oriented along PCA eigenvectors, following any image rotation.
        _bys, _bxs = np.where(disk_mask)
        if len(_bys) > 10:
            _ccy = float(_bys.mean()); _ccx = float(_bxs.mean())
            _py = _bys.astype(np.float64) - _ccy
            _px = _bxs.astype(np.float64) - _ccx
            try:
                _coords = np.stack([_px, _py], axis=1)
                _cov = np.cov(_coords.T)
                _evals, _evecs = np.linalg.eigh(_cov)
                _minor_vec = _evecs[:, np.argmin(_evals)]
                _major_vec = _evecs[:, np.argmax(_evals)]
                # p99.5 of absolute projection gives the actual limb radius
                # (std*multiplier overshoots; percentile of bright pixels matches limb)
                _proj_minor_abs = np.abs(_coords @ _minor_vec)
                _proj_major_abs = np.abs(_coords @ _major_vec)
                _semi_minor = float(np.percentile(_proj_minor_abs, 99.5))
                _semi_major_raw = float(np.percentile(_proj_major_abs, 99.5))
                # Cap major axis if rings contaminate it (aspect > 1.3)
                _aspect = _semi_major_raw / max(_semi_minor, 1.0)
                _semi_major = _semi_minor * 1.10 if _aspect > 1.3 else _semi_major_raw
                # Build oriented ellipse mask
                _Y, _X = np.mgrid[:h, :w]
                _px_all = _X.astype(np.float64) - _ccx
                _py_all = _Y.astype(np.float64) - _ccy
                _proj_minor = _px_all*_minor_vec[0] + _py_all*_minor_vec[1]
                _proj_major = _px_all*_major_vec[0] + _py_all*_major_vec[1]
                disk_mask = ((_proj_major / _semi_major)**2 +
                             (_proj_minor / _semi_minor)**2) <= 1.0
                # Keep the ring-plane coordinate maps and the globe's equatorial
                # radius. The exclusion below uses them to reach the rings that
                # cross IN FRONT of the globe, where brightness cannot find them.
                _ring_perp  = _proj_minor    # perpendicular to ring plane
                _ring_along = _proj_major    # along the ring plane
                _ring_r_eq  = _semi_major    # globe equatorial radius
                _ring_r_pol = _semi_minor    # globe polar radius
            except Exception:
                pass

    # ── Build outer-only mask: fires outside disk boundary only ─────────
    # Uses the fitted ellipse (Saturn mode) or threshold mask for the distance
    # transform — this ignores rings and follows the oblate planet shape.
    # The ref_lum threshold provides the hard inner clip so the mask starts
    # exactly at the visual planet limb with no gap.
    # The Smooth slider feathers the ellipse boundary before distance computation
    # so it affects disk-edge smoothness, not the mask-to-limb gap.
    edge_px   = max(float(edge), 1.0)
    smooth_px = max(float(smooth), 0.0)
    inset_px  = float(inset)   # negative = push mask inward over limb

    # Optionally smooth the disk_mask boundary (feathers edge detection)
    if smooth_px > 0:
        disk_mask_soft = gaussian_filter(disk_mask.astype(np.float64), sigma=smooth_px) > 0.5
    else:
        disk_mask_soft = disk_mask

    # Distance from disk boundary (ellipse or threshold)
    dist_out_inv = distance_transform_edt(~disk_mask_soft)   # 0 at limb, +ve outward
    dist_in      = distance_transform_edt(disk_mask_soft)    # 0 at limb, +ve inward

    # Signed distance from limb: positive = outside (sky), negative = inside (planet)
    signed_dist = np.where(~disk_mask_soft, dist_out_inv, -dist_in)

    # Mask band model:
    #   Edge  = total band width in pixels
    #   Inset = shift band center toward planet interior (+) or sky (-)
    #           inset=0  → band centered on limb (half inward, half outward)
    #           inset=+N → band shifts N px inward (covers rind more)
    #           inset=-N → band shifts N px outward (gap between mask and limb)
    half = edge_px / 2.0
    # inset_px>0 means push inward → center moves to negative signed_dist
    center = -inset_px
    # Linear ramp peaking at band center, falling to 0 at band edges
    mask = np.clip(1.0 - np.abs(signed_dist - center) / half, 0.0, 1.0)

    mask = np.clip(mask, 0.0, 1.0)

    # Hard clip at visual limb: zero well inside the planet but allow the mask
    # to overlap the limb transition zone where the rind (dark undershoot) lives.
    # We erode the planet interior by edge_px pixels so the clip boundary sits
    # clearly inside the disk — the outer ramp naturally dies to zero before
    # reaching deep interior pixels.
    clip_radius = max(int(round(edge_px)), 1)
    from scipy.ndimage import generate_binary_structure
    struct = generate_binary_structure(2, 1)
    if ref_lum is not None:
        planet_interior = binary_erosion(
            ref_lum >= disk_thresh,
            structure=struct, iterations=clip_radius, border_value=0
        )
        mask[planet_interior] = 0.0
    else:
        planet_interior = binary_erosion(
            disk_mask,
            structure=struct, iterations=clip_radius, border_value=0
        )
        mask[planet_interior] = 0.0

    # ── Saturn mode: protect bright ring structure near the limb ─────
    # The mask band wraps the entire globe limb, so wherever the rings pass
    # close to (or across) the limb it would blend them back toward the
    # unsharpened original and soften real ring detail. We carve the rings out
    # of the band. Key distinction: the rind we WANT to correct is a sharpening
    # artifact — it is absent from the unsharpened original (ref_lum) — while
    # the rings are genuinely bright there. So any bright pixel that lies
    # OUTSIDE the fitted disk in ref_lum is ring/moon, not rind, and is removed
    # from the mask. Dark undershoot stays below threshold and is untouched.
    # Detecting rings pixel-by-pixel near the limb does not work: the fitted
    # ellipse tracks the 99.5th-percentile limb while the brightness silhouette
    # runs wider (soft limb + seeing), so the globe's own rim reads as "ring"
    # all the way round and suppresses the mask everywhere. Growing the ellipse
    # to dodge that then stops detecting the ring material closest to the globe,
    # and the band runs over the rings again.
    #
    # Instead, decide per AZIMUTH. Sample a probe annulus well clear of the
    # globe's soft rim (t >= 1.25 of the fitted ellipse) — the limb can never
    # reach there, but rings extend far past it. Any azimuth carrying bright
    # material out there is a ring bearing, and the whole wedge is dropped. That
    # also covers rings crossing IN FRONT of the globe: those lie on the same
    # bearings as the ansae, so their wedge is already excluded.
    if saturn_mode and ref_lum is not None and _ring_r_eq and _ring_r_pol:
        _a = float(_ring_r_eq); _b = float(_ring_r_pol)
        t_ell = np.sqrt((_ring_along / _a)**2 + (_ring_perp / _b)**2)
        probe = (ref_lum > disk_thresh) & (t_ell >= 1.25) & (t_ell <= 4.0)
        if int(probe.sum()) > 30:
            NB   = 180                                   # 2-degree bearings
            th   = np.degrees(np.arctan2(_ring_perp, _ring_along))
            idx  = np.clip(((th + 180.0) / 360.0 * NB).astype(np.int64), 0, NB - 1)
            cnt  = np.bincount(idx[probe], minlength=NB).astype(np.float64)
            # circular smoothing so wedge edges are not ragged
            k    = np.array([1.0, 2.0, 3.0, 2.0, 1.0]); k /= k.sum()
            cnt  = np.convolve(np.concatenate([cnt[-4:], cnt, cnt[:4]]),
                               k, mode="same")[4:-4]
            ring_az = cnt > max(cnt.max() * 0.15, 3.0)
            if ring_az.any() and not ring_az.all():
                wedge = ring_az[idx]
                wsoft = gaussian_filter(wedge.astype(np.float64), sigma=3.0)
                mask *= (1.0 - np.clip(wsoft, 0.0, 1.0))

    # ── Arc restriction (manual) ─────────────────────────────────────
    # gap_width = Arc Start, gap_angle = Arc End (both 0 = full circle).
    # Convention: 0=top, +CW, -CCW.
    # The EXCLUDED zone goes CW from Arc Start to Arc End.
    # The mask is KEPT everywhere outside the excluded zone.
    # Example: Start=90, End=-90 excludes the right arc, keeps left+top arc.
    if gap_width != 0.0 or gap_angle != 0.0:
        ys, xs = np.where(disk_mask)
        cy, cx = (ys.mean(), xs.mean()) if len(ys) > 0 else (h/2, w/2)
        Y, X = np.mgrid[:h, :w]
        # Angle map: 0=top, clockwise, range -180..180
        angles = np.degrees(np.arctan2(X - cx, -(Y - cy)))  # -180..180

        arc_start = float(gap_width)
        arc_end   = float(gap_angle)

        # Build exclusion zone: CW from arc_start to arc_end
        if arc_start <= arc_end:
            excl_mask = (angles >= arc_start) & (angles <= arc_end)
        else:
            # Wraps over ±180 boundary
            excl_mask = (angles >= arc_start) | (angles <= arc_end)

        # Keep zone = NOT excluded. Feather the boundary softly.
        from scipy.ndimage import gaussian_filter as gf
        excl_soft = gf(excl_mask.astype(np.float32), sigma=4.0)
        mask *= (1.0 - np.clip(excl_soft, 0.0, 1.0))

    return mask.astype(np.float32)


def apply_derind(
    img_array: np.ndarray,
    edge: float        = 8.0,
    smooth: float      = 2.0,
    inset: float       = 0.0,
    gap_width: float   = 0.0,
    gap_angle: float   = 0.0,
    saturn_mode: bool  = False,
    dark_edge: bool    = True,
    pre_blur: float    = 0.0,
    show_ring_map: bool = False,
    ref_lum: np.ndarray = None,
    ref_arr: np.ndarray = None,
) -> np.ndarray:
    """
    De-rind: suppress the halo/rind artifact around bright planetary limbs.

    Parameters:
      edge        : outer extent of correction (px beyond limb)
      feather     : inward feather depth (px inside limb, tapering to zero)
      smooth      : broaden mask both ways — removes discontinuities (Gaussian sigma)
      inset       : pull mask band inward from limb by this many px (0 = starts at limb)
      gap_width   : angular gap in the mask in degrees (0=none); use for Saturn rings
      gap_angle   : rotation of the gap center in degrees (0=top, clockwise)
      saturn_mode : auto-apply symmetric 180° gaps on both sides of disk
      dark_edge   : suppress bright diffraction rings outside the main halo band
      pre_blur    : optional pre-blur sigma applied before correction
      show_ring_map : overlay the active mask as a red tint for inspection
    """
    from scipy.ndimage import gaussian_filter
    result = img_array.astype(np.float64)
    _ref_arr = ref_arr  # original image for correction target

    # Optional pre-blur
    if pre_blur > 0:
        result = np.stack([
            gaussian_filter(result[..., c], sigma=pre_blur) for c in range(3)
        ], axis=-1)

    lum = luminance(result.astype(np.float32))

    # Build the active mask
    mask = _derind_mask(lum, edge=edge, smooth=smooth,
                        gap_width=gap_width, gap_angle=gap_angle,
                        saturn_mode=saturn_mode, inset=inset, ref_lum=ref_lum)

    # Dark-edge extension: if enabled, extend correction outward to catch
    # diffraction rings beyond the main halo band (common on Mars).
    # The extended mask is merged with the main mask using max() so the result
    # is a single smooth gradient, not two separate rings.
    if dark_edge:
        outer_ext = _derind_mask(lum, edge=edge * 2.5,
                                  smooth=smooth * 1.5, gap_width=gap_width,
                                  gap_angle=gap_angle, saturn_mode=saturn_mode,
                                  inset=inset, ref_lum=ref_lum)
        # Merge: take the maximum of main mask and half-weight outer extension
        # so there are no gaps and the result is a single continuous band
        mask = np.maximum(mask, outer_ext * 0.5)

    # ── Rind correction ──────────────────────────────────────────────────────
    # The rind is a dark undershoot introduced by wavelet sharpening.
    # It does not exist in ref_arr (the pre-sharpening original).
    # Correction: blend the processed image back toward ref_arr under the mask.
    # Blur (sigma) smooths ref_arr first to suppress any HF noise in the
    # original that we don't want to reintroduce at the limb.
    # Smooth controls mask edge feathering independently.

    blur_sigma   = max(float(smooth), 1.0)   # mask feather sigma
    target_sigma = max(float(smooth), 1.0)   # ref_arr smoothing sigma

    if _ref_arr is not None:
        target = np.stack([
            gaussian_filter(_ref_arr[..., c].astype(np.float64), sigma=target_sigma)
            for c in range(3)
        ], axis=-1)
    else:
        # Fallback: use a heavily blurred version of the processed image
        target = np.stack([
            gaussian_filter(result[..., c].astype(np.float64), sigma=target_sigma * 3)
            for c in range(3)
        ], axis=-1)

    # Blur the mask for smooth falloff at band edges
    mask_blurred = gaussian_filter(mask.astype(np.float64), sigma=blur_sigma)
    mask_blurred = np.clip(mask_blurred, 0.0, 1.0)

    # Blend toward the rind-free reference in LUMINANCE only, keeping the current
    # (already color-corrected) chroma. ref_arr is the pre-sharpen original,
    # which is also pre color-correction — blending full RGB toward it would
    # drag the limb back to the uncorrected hue and leave a colored seam wherever
    # the mask acts. The rind is a brightness halo, so we take the corrected
    # luminance from the blend, then rescale each pixel's RGB to that luminance.
    # This removes the rind while the RGB color grade downstream stays intact,
    # and it is a no-op outside the mask (blended == result there → scale ≈ 1).
    mask3    = mask_blurred[..., np.newaxis]
    blended  = result * (1.0 - mask3) + target * mask3
    cur_lum  = luminance(result.astype(np.float32)).astype(np.float64)
    new_lum  = luminance(blended.astype(np.float32)).astype(np.float64)
    eps      = 1e-4
    scale    = np.where(cur_lum > eps, new_lum / (cur_lum + eps), 1.0)
    scale    = np.clip(scale, 0.0, 4.0)
    result   = np.clip(result * scale[..., np.newaxis], 0.0, 1.0)

    if show_ring_map:
        # Overlay: blend red tint only where mask > 0. Interior (mask=0) is untouched.
        # Use additive tint so planet detail remains visible inside the mask band.
        tint_strength = np.clip(mask_blurred * 0.90, 0, 1)[..., np.newaxis]
        red_tint = np.zeros_like(result)
        red_tint[..., 0] = 1.0   # full red channel
        red_tint[..., 1] = 0.0
        red_tint[..., 2] = 0.0
        # Blend: masked pixels dimmed + vivid red; unmasked pixels unchanged
        result = result * (1.0 - tint_strength * 0.7) + red_tint * tint_strength * 0.7
        result = np.clip(result, 0, 1)

    return result.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Tools — Levels & Curves  +  CLAHE
# ══════════════════════════════════════════════════════════════════════════════

def _build_curve_lut(control_points):
    """
    Build a 256-entry float32 LUT from a list of (x, y) control points in [0,1].
    Uses PCHIP monotone cubic interpolation so the curve never overshoots.
    When only two endpoints exist, uses linear interpolation to guarantee a
    straight line regardless of where the black/white points sit.
    Returns a float32 array of shape (256,) with values in [0,1].
    """
    pts = sorted(control_points, key=lambda p: p[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    t = np.linspace(0.0, 1.0, 256)

    # With only two points (black and white endpoints) always use linear
    # interpolation — PCHIP can curve between two points with asymmetric x
    if len(xs) == 2:
        x0, y0 = xs[0], ys[0]
        x1, y1 = xs[1], ys[1]
        span = max(x1 - x0, 1e-6)
        # Values below black point → 0, above white point → 1, linear in between
        lut = np.clip((t - x0) / span * (y1 - y0) + y0, 0.0, 1.0).astype(np.float32)
        return lut

    from scipy.interpolate import PchipInterpolator
    # Ensure boundary anchors for multi-point curves
    if xs[0] > 0.001:
        xs.insert(0, 0.0); ys.insert(0, 0.0)
    if xs[-1] < 0.999:
        xs.append(1.0); ys.append(1.0)
    interp = PchipInterpolator(xs, ys)
    lut = np.clip(interp(t).astype(np.float32), 0.0, 1.0)
    return lut



# ── Deconvolution ──────────────────────────────────────────────────────────────

def _moffat_psf(gamma: float = 1.0, beta: float = 2.0, size: int = 5) -> np.ndarray:
    """Moffat PSF kernel — models atmospheric seeing.  Normalized to sum=1."""
    half = size // 2
    yx = np.mgrid[-half:half+1, -half:half+1].astype(np.float64)
    r2 = yx[0]**2 + yx[1]**2
    psf = (1.0 + r2 / gamma**2) ** (-beta)
    return (psf / psf.sum()).astype(np.float32)


def apply_deconvolution(
    img_array: np.ndarray,
    strength: float = 0.0,
    use_contrast: bool = False,
    contrast_strength: float = 0.0,
) -> np.ndarray:
    """
    Adaptive Richardson-Lucy deconvolution on the oklab L channel only.

    strength        : 0–100.  Controls how much correction is applied.
                      Maps to RL gain via  gain = strength / 3.14.
    use_contrast    : If True, weight the correction by local contrast so
                      high-contrast edges get more sharpening than flat areas.
    contrast_strength: 0–10.  Controls the power of the contrast weighting.
                       0 = uniform weight.  10 = strongly contrast-dependent.

    Algorithm:
      1. Extract oklab L channel.
      2. Build a 5×5 Moffat PSF (gamma=1, beta=2).
      3. One Richardson-Lucy iteration:
           conv        = PSF ⊗ L
           relative    = L / (conv + ε)
           correction  = PSF_mirror ⊗ relative
           corr_delta  = correction - 1
      4. If use_contrast: weight by sqrt(local_std_norm)^contrast_power
      5. L_out = clip(L + gain * weight * corr_delta * L, 0, 1)
      6. Reconstruct RGB by scaling chroma to match the new L.
    """
    from scipy.ndimage import gaussian_filter, convolve as nd_convolve

    if strength <= 0.0:
        return img_array.astype(np.float32)

    img = np.clip(img_array, 0.0, 1.0).astype(np.float64)
    gain = strength / 3.14

    # ── Extract oklab L ──
    lab = rgb_to_oklab(img)
    L = lab[..., 0].copy()

    # ── Moffat PSF ──
    psf = _moffat_psf(gamma=1.0, beta=2.0, size=5).astype(np.float64)
    psf_mirror = psf[::-1, ::-1]

    # ── One Richardson-Lucy iteration ──
    conv = nd_convolve(L, psf, mode='reflect')
    relative = L / (conv + 1e-12)
    correction = nd_convolve(relative, psf_mirror, mode='reflect')
    corr_delta = correction - 1.0

    # ── Contrast weighting ──
    if use_contrast and contrast_strength > 0.0:
        # Local std dev via Gaussian approximation (radius ~3.5px window)
        local_mean = gaussian_filter(L, sigma=3.5)
        local_sq_mean = gaussian_filter(L**2, sigma=3.5)
        local_var = np.maximum(local_sq_mean - local_mean**2, 0.0)
        local_std = np.sqrt(local_var)
        # Normalize to [0, 1]
        std_max = local_std.max()
        if std_max > 1e-10:
            contrast_norm = local_std / std_max
        else:
            contrast_norm = np.zeros_like(local_std)
        # Apply power: higher contrast_strength = more contrast-dependent
        power = 0.5 + contrast_strength * 0.25  # 0.5 at 0, 3.0 at 10
        weight = contrast_norm ** power
    else:
        weight = np.ones_like(L)

    # ── Apply correction to L ──
    L_out = np.clip(L + gain * weight * corr_delta * L, 0.0, 1.0)

    # ── Reconstruct: scale a*,b* by L ratio to preserve hue ──
    ratio = np.where(L > 1e-6, L_out / (L + 1e-6), 1.0)
    ratio = np.clip(ratio, 0.0, 4.0)
    lab_out = lab.copy()
    lab_out[..., 0] = L_out
    # Gently scale chroma proportionally to keep color consistent
    lab_out[..., 1] *= ratio
    lab_out[..., 2] *= ratio

    return np.clip(oklab_to_rgb(lab_out), 0.0, 1.0).astype(np.float32)



def apply_dehaze(
    img_array: np.ndarray,
    block_size: int = 5,
    amount: float = 0.5,
) -> np.ndarray:
    """
    Dehaze — reduces chromatic fringing (e.g. blue/cyan halos) at planetary edges.

    Uses edge-based chroma suppression:
    1. Detect high-gradient edge zones on the luminance channel (Sobel).
    2. Broaden the edge mask to cover the fringe halo region.
    3. In fringe zones, blend each RGB channel toward luminance (desaturate toward gray).
       This suppresses the color offset caused by channel misregistration / CA.
    4. Blend result with original using amount as the mix ratio.

    block_size : controls the edge detection scale and fringe zone width (1–100).
                 Smaller = tighter to edges, larger = wider fringe removal.
    amount     : blend ratio — 0.0 = original, 1.0 = full chroma suppression at edges.
    """
    from scipy.ndimage import gaussian_filter, sobel as _sobel

    img = np.clip(img_array, 0.0, 1.0).astype(np.float32)
    if amount <= 0.0:
        return img

    # ── Luminance ──
    lum = (0.299 * img[..., 0] +
           0.587 * img[..., 1] +
           0.114 * img[..., 2]).astype(np.float32)

    # ── Edge detection on luminance ──
    sigma = max(0.5, block_size / 5.0)
    lum_smooth = gaussian_filter(lum, sigma=sigma)
    gx = _sobel(lum_smooth, axis=1)
    gy = _sobel(lum_smooth, axis=0)
    edge_mag = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)
    emax = edge_mag.max()
    if emax < 1e-6:
        return img
    edge_norm = edge_mag / emax

    # ── Spread edge mask to cover fringe zone ──
    fringe_sigma = max(0.5, block_size * 0.4)
    fringe_mask = gaussian_filter(edge_norm, sigma=fringe_sigma)
    fmax = fringe_mask.max()
    if fmax < 1e-6:
        return img
    fringe_mask = np.clip(fringe_mask / fmax, 0.0, 1.0)

    # ── Chroma suppression: blend each channel toward luma at fringe locations ──
    result = img.copy()
    for c in range(3):
        result[..., c] = img[..., c] + fringe_mask * amount * (lum - img[..., c])

    return np.clip(result, 0.0, 1.0).astype(np.float32)

def apply_levels_curves(
    img_array: np.ndarray,
    black_r: float = 0.0, white_r: float = 1.0, gamma_r: float = 1.0,
    black_g: float = 0.0, white_g: float = 1.0, gamma_g: float = 1.0,
    black_b: float = 0.0, white_b: float = 1.0, gamma_b: float = 1.0,
    curve_lut_r = None,
    curve_lut_g = None,
    curve_lut_b = None,
) -> np.ndarray:
    """
    Per-channel levels (black point, white point, gamma) followed by a
    tone curve LUT.  All inputs are float32 [0,1].

    black / white : clamp and stretch the input range
        stretched = (x - black) / (white - black)
    gamma : applied after stretch
        out = stretched ^ (1/gamma)
    curve_lut : 256-entry float32 array mapping [0,255] → [0,1]
        Applied as a final LUT after levels+gamma.
        None = identity (no curve).
    """
    img = img_array.astype(np.float64)
    result = np.empty_like(img)

    def _process_channel(ch, black, white, gamma, lut):
        w_range = max(white - black, 1e-6)
        out = np.clip((ch - black) / w_range, 0.0, 1.0)
        if abs(gamma - 1.0) > 0.001:
            out = np.power(out, 1.0 / gamma)
        if lut is not None:
            # Map float [0,1] through the 256-entry LUT
            idx = np.clip((out * 255.0).astype(np.int32), 0, 255)
            out = lut[idx].astype(np.float64)
        return out

    result[..., 0] = _process_channel(img[..., 0], black_r, white_r, gamma_r, curve_lut_r)
    result[..., 1] = _process_channel(img[..., 1], black_g, white_g, gamma_g, curve_lut_g)
    result[..., 2] = _process_channel(img[..., 2], black_b, white_b, gamma_b, curve_lut_b)

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _clahe_channel(plane: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """
    Apply CLAHE to a single float32 [0,1] plane.
    Uses skimage if available, otherwise falls back to a scipy tile-based implementation.
    clip_limit is in the 0.5-10.0 user range (divided by 10 for skimage).
    """
    try:
        from skimage.exposure import equalize_adapthist
        return equalize_adapthist(
            plane, kernel_size=tile_grid,
            clip_limit=clip_limit / 10.0).astype(np.float32)
    except ImportError:
        pass

    # scipy fallback: proper CLAHE with bilinear tile interpolation
    # Matches skimage's approach: compute per-tile CDFs, then interpolate
    from scipy.ndimage import zoom as _zoom
    h, w = plane.shape
    nr = tile_grid   # number of tile rows
    nc = tile_grid   # number of tile cols

    def _tile_cdf(patch, cl):
        """Compute clipped CDF for a tile, returns 256-entry LUT."""
        flat = (np.clip(patch, 0.0, 1.0) * 255.0).astype(np.int32).flatten()
        hist = np.bincount(flat, minlength=256).astype(np.float32)
        # Clip and redistribute excess
        clip_val = max(1.0, cl * flat.size / 256.0)
        excess = np.sum(np.maximum(hist - clip_val, 0.0))
        hist = np.minimum(hist, clip_val)
        hist += excess / 256.0
        cdf = np.cumsum(hist)
        lo, hi = cdf[0], cdf[-1]
        if hi > lo:
            cdf = (cdf - lo) / (hi - lo)
        else:
            cdf = np.linspace(0.0, 1.0, 256)
        return cdf.astype(np.float32)

    # Build one CDF per tile center
    cdfs = np.zeros((nr, nc, 256), dtype=np.float32)
    # Tile boundaries (edges get half-tile padding like skimage)
    row_borders = np.linspace(0, h, nr + 1).astype(int)
    col_borders = np.linspace(0, w, nc + 1).astype(int)
    for r in range(nr):
        for c in range(nc):
            patch = plane[row_borders[r]:row_borders[r+1],
                          col_borders[c]:col_borders[c+1]]
            cdfs[r, c] = _tile_cdf(patch, clip_limit)

    # Tile center coordinates (for interpolation weights)
    row_centers = ((row_borders[:-1] + row_borders[1:]) / 2.0)
    col_centers = ((col_borders[:-1] + col_borders[1:]) / 2.0)

    # Apply bilinear interpolation between 4 nearest tile CDFs
    out = np.empty_like(plane)
    px_idx = np.clip((plane * 255.0).astype(np.int32), 0, 255)

    for r in range(nr):
        for c in range(nc):
            r0 = row_borders[r]; r1 = row_borders[r+1]
            c0 = col_borders[c]; c1 = col_borders[c+1]
            patch_idx = px_idx[r0:r1, c0:c1]

            # Determine 4 neighboring tile CDFs for bilinear blend
            ry = np.arange(r0, r1)
            cx = np.arange(c0, c1)

            # Find left/right tile neighbors
            rl = r - 1 if r > 0 else r
            rr = r + 1 if r < nr - 1 else r
            cl_n = c - 1 if c > 0 else c
            cr_n = c + 1 if c < nc - 1 else c

            # Vertical weight: distance from row_centers[rl] to row_centers[rr]
            yc_lo = row_centers[rl]; yc_hi = row_centers[rr]
            if yc_hi > yc_lo:
                wy = np.clip((ry - yc_lo) / (yc_hi - yc_lo), 0.0, 1.0)
            else:
                wy = np.zeros(len(ry))

            # Horizontal weight
            xc_lo = col_centers[cl_n]; xc_hi = col_centers[cr_n]
            if xc_hi > xc_lo:
                wx = np.clip((cx - xc_lo) / (xc_hi - xc_lo), 0.0, 1.0)
            else:
                wx = np.zeros(len(cx))

            wy = wy[:, np.newaxis]   # (rows, 1)
            wx = wx[np.newaxis, :]   # (1, cols)

            # Bilinear blend of 4 CDFs
            cdf_tl = cdfs[rl, cl_n][patch_idx]
            cdf_tr = cdfs[rl, cr_n][patch_idx]
            cdf_bl = cdfs[rr, cl_n][patch_idx]
            cdf_br = cdfs[rr, cr_n][patch_idx]

            out[r0:r1, c0:c1] = (
                (1-wy)*(1-wx)*cdf_tl +
                (1-wy)*wx    *cdf_tr +
                wy    *(1-wx)*cdf_bl +
                wy    *wx    *cdf_br
            )

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_clahe(
    img_array: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: int = 8,
    channel_mode: str = "luminance",
    strength: float = 1.0,
) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalisation).

    channel_mode : "luminance" — apply to L only (safe, no color shift)
                   "rgb"       — apply to each channel independently
    clip_limit   : 0.5-10.0.  Higher = more contrast enhancement.
    tile_grid    : NxN grid of tiles (4-64).  Smaller = more local.
    strength     : 0.0-1.0 blend with original (1.0 = full CLAHE).
    """
    img = np.clip(img_array, 0.0, 1.0).astype(np.float32)

    if channel_mode == "luminance":
        lum = (0.299 * img[..., 0] +
               0.587 * img[..., 1] +
               0.114 * img[..., 2]).astype(np.float32)
        lum_eq = _clahe_channel(lum, clip_limit, tile_grid)
        ratio = np.where(lum > 1e-6, lum_eq / (lum + 1e-6), 1.0)
        ratio = np.clip(ratio, 0.0, 4.0)
        enhanced = img * ratio[..., np.newaxis]
    else:  # "rgb"
        enhanced = np.stack([
            _clahe_channel(img[..., c], clip_limit, tile_grid)
            for c in range(3)
        ], axis=-1)

    result = img + strength * (enhanced - img)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ── Moon Recovery ──────────────────────────────────────────────────────────────

def apply_moon_recovery(
    img_array: np.ndarray,
    original_array: np.ndarray,
    circles: list,
    boost: float = 2.0,
    feather: float = 5.0,
    saturation: float = 1.0,
    darken_edge: float = 0.0,
) -> np.ndarray:
    """
    Recover faint moons using user-placed circle masks.

    For each (cx, cy, radius) circle in image pixel coordinates:
      - Build a soft circular mask feathered at the edge
      - Blend a boosted copy of the original into the processed result
      - Optionally desaturate within the circle to remove color cast
      - Optionally darken the region just outside the circle to suppress
        the oblong background shape beneath the moon

    Args:
        img_array:      Processed image  (H, W, 3) float32 0-1
        original_array: Pre-pipeline original (H, W, 3) float32 0-1
        circles:        List of (cx, cy, radius) in full image pixel coords
        boost:          Brightness multiplier applied to the original in each circle
        feather:        Cosine feather width (px) applied to circle mask edge
        saturation:     0.0 = fully desaturated inside circles, 1.0 = full color
        darken_edge:    0.0-1.0 — how much to darken the ring just outside the circle
    """
    from scipy.ndimage import gaussian_filter

    if not circles:
        return img_array

    processed = np.clip(img_array,      0.0, 1.0).astype(np.float32)
    original  = np.clip(original_array, 0.0, 1.0).astype(np.float32)
    H, W      = processed.shape[:2]

    combined_mask  = np.zeros((H, W), dtype=np.float32)
    combined_darken = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    for cx, cy, radius in circles:
        if radius < 0.5:
            continue
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        # ── Inner boost mask ──────────────────────────────────────────
        if feather > 0.0:
            inner = radius - feather * 0.5
            outer = radius + feather * 0.5
            mask = np.where(
                dist <= inner, 1.0,
                np.where(dist >= outer, 0.0,
                    0.5 * (1.0 + np.cos(np.pi * (dist - inner) / feather))
                )
            ).astype(np.float32)
        else:
            mask = (dist <= radius).astype(np.float32)
        combined_mask = np.maximum(combined_mask, mask)

        # ── Outer darken ring ─────────────────────────────────────────
        # Ring is centered ON the circle edge, spanning inward and outward
        # so it overlaps the feather zone. This darkens the background
        # beneath the feather rather than creating a separate outer ring.
        if darken_edge > 0.0:
            ring_half = max(feather * 1.5, max(8.0, radius * 0.3))
            ring_inner = radius - ring_half
            ring_outer = radius + ring_half
            # Cosine bell centered on radius — peaks at circle edge
            darken_mask = np.where(
                (dist >= ring_inner) & (dist <= ring_outer),
                0.5 * (1.0 - np.cos(np.pi * (dist - ring_inner) / (ring_half * 2))),
                0.0
            ).astype(np.float32)
            combined_darken = np.maximum(combined_darken, darken_mask)

    combined_mask = np.clip(combined_mask, 0.0, 1.0)

    # Boosted original
    boosted = np.clip(original * boost, 0.0, 1.0)

    # Blend boost inside circle
    mask3  = combined_mask[..., np.newaxis]
    result = processed * (1.0 - mask3) + boosted * mask3

    # Saturation adjustment inside circles
    if saturation < 0.999:
        lum = (0.299 * result[..., 0] +
               0.587 * result[..., 1] +
               0.114 * result[..., 2])[..., np.newaxis]
        gray = np.concatenate([lum, lum, lum], axis=-1)
        result = result + mask3 * (gray - result) * (1.0 - saturation)

    # Darken outer ring
    if darken_edge > 0.0 and combined_darken.max() > 0.0:
        darken3 = combined_darken[..., np.newaxis]
        result = result * (1.0 - darken3 * darken_edge)

    return np.clip(result, 0.0, 1.0).astype(np.float32)
