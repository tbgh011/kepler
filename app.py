"""
app.py – Kepler  v1.1
· Live processing as sliders move (debounced 400 ms)
· Right-click OR double-click any slider → reset to default
· Per-tab Reset (resets that tab's sliders only)
· Global Reset button (resets image + all sliders)
· Wavelet tab: clear workflow sections (Manual / Filter / Auto)
· Histogram: annotated with shadow / mid-tone / highlight regions
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import sys
import os
import json
from PIL import Image, ImageTk, ImageDraw
import numpy as np

import processing as proc

# ─────────────────────────────────────────────────────────────
#  Colors
# ─────────────────────────────────────────────────────────────
BG_DEEP    = "#f0f4f8"
BG_PANEL   = "#e2e8f0"
BG_CARD    = "#ffffff"
BG_RAISED  = "#cbd5e1"
SLIDER_TRG = "#b0bec5"
SLIDER_THM = "#334155"
FG_BRIGHT  = "#000000"
FG_MID     = "#0a0e18"
FG_DIM     = "#1e2532"
ACCENT_C   = "#014f7a"
ACCENT_O   = "#6a2306"
ACCENT_G   = "#0b4621"
ACCENT_P   = "#5b2bad"
ACCENT_R   = "#dc2626"
BORDER     = "#64748b"
SEL_BG     = "#bfdbfe"
BTN_BG     = "#b8c5d6"   # button face — slightly darker than BG_RAISED
BTN_ACTIVE = "#93a8c0"   # button hover/press — clearly darker
FFT_CHECK  = "#86efac"   # FFT stage checkbox fill — readable on either row bg

# ─────────────────────────────────────────────────────────────
#  Fonts
# ─────────────────────────────────────────────────────────────
_MAC   = sys.platform == "darwin"
_LINUX = sys.platform.startswith("linux")
def _ff(): return "Menlo" if _MAC else "Consolas"
def _fs(win, mac): return mac if _MAC else (win - 2 if _LINUX else win)
_FF = _ff()

F_XS   = (_FF, _fs(10,  9))
F_HINT = (_FF, _fs(12, 11))
F_TINY = (_FF, _fs(14, 12))
F_SM   = (_FF, _fs(14, 12))
F_MD   = (_FF, _fs(15, 13))
F_BOLD = (_FF, _fs(17, 14), "bold")
F_HDR  = (_FF, _fs(21, 17), "bold")

PREV_W, PREV_H = 400, 300
PREFS_FILE = os.path.join(os.path.expanduser("~"), ".kepler_prefs.json")

# ─────────────────────────────────────────────────────────────
#  Prefs
# ─────────────────────────────────────────────────────────────
def load_prefs():
    try:
        with open(PREFS_FILE) as f: return json.load(f)
    except Exception: return {}

def save_prefs(data):
    try:
        d = load_prefs(); d.update(data)
        with open(PREFS_FILE, "w") as f: json.dump(d, f)
    except Exception: pass

# ─────────────────────────────────────────────────────────────
#  LabeledSlider  — right-click or double-click resets to default
# ─────────────────────────────────────────────────────────────
class LabeledSlider(tk.Frame):
    def __init__(self, parent, label, from_, to_, initial,
                 color=ACCENT_C, fmt="{:.0f}", callback=None, resolution=None, **kw):
        super().__init__(parent, bg=BG_CARD, **kw)
        self._default  = initial
        self.fmt       = fmt
        self.callback  = callback
        self.var       = tk.DoubleVar(value=initial)
        self._dragging = False   # True while mouse button held on slider

        # Auto-derive resolution from format string if not supplied
        if resolution is None:
            if callable(fmt):
                resolution = 1
            else:
                import re
                m = re.search(r'\.(\d+)f', fmt)
                decimals = int(m.group(1)) if m else 0
                resolution = 10 ** (-decimals) if decimals > 0 else 1

        self._lbl = tk.Label(self, text=label, bg=BG_CARD, fg=FG_MID,
                             font=F_SM, anchor="w")
        self._lbl.pack(side="left", padx=(0, 4))

        self.scale = tk.Scale(
            self, from_=from_, to=to_, orient="horizontal",
            variable=self.var, showvalue=False, length=110,
            resolution=resolution,
            troughcolor=SLIDER_TRG, bg=BG_CARD, fg=SLIDER_THM,
            activebackground=color, highlightthickness=0, bd=0,
            sliderrelief="raised", sliderlength=14,
            command=self._on_change)
        self.scale.pack(side="left", fill="x", expand=True, padx=2)

        self.val_lbl = tk.Label(self, text=fmt(initial) if callable(fmt) else fmt.format(initial),
                                bg=BG_CARD, fg=color, font=F_MD,
                                width=7, anchor="e")
        self.val_lbl.pack(side="left")

        # Suppress live pipeline during drag; fire once on release
        self.scale.bind("<ButtonPress-1>",   self._drag_start)
        self.scale.bind("<ButtonRelease-1>", self._drag_end)

        # Right-click or double-click → reset to default
        for w in (self.scale, self._lbl, self.val_lbl):
            w.bind("<Button-3>",       lambda e: self.reset())
            w.bind("<Double-Button-1>",lambda e: self.reset())

    def _drag_start(self, event):
        self._dragging = True

    def _drag_end(self, event):
        self._dragging = False
        # Fire callback once with final value after drag ends
        if self.callback:
            self.callback(self.var.get())

    def _on_change(self, v):
        val = float(v)
        self.val_lbl.configure(text=self.fmt(val) if callable(self.fmt) else self.fmt.format(val))
        # Only fire pipeline callback if not mid-drag
        if not self._dragging and self.callback:
            self.callback(val)

    def get(self):   return self.var.get()
    def set(self, v, fire_callback=True):
        self.var.set(v)
        self.val_lbl.configure(text=self.fmt(v) if callable(self.fmt) else self.fmt.format(v))
        if fire_callback and self.callback:
            self.callback(v)

    def reset(self):
        self.set(self._default, fire_callback=True)

# ─────────────────────────────────────────────────────────────
#  ToggleVar
# ─────────────────────────────────────────────────────────────
class ToggleVar(tk.Frame):
    def __init__(self, parent, label, sublabel="", initial=True, command=None, **kw):
        super().__init__(parent, bg=BG_CARD, **kw)
        self._default = initial
        self.var      = tk.BooleanVar(value=initial)
        tk.Checkbutton(self, text=label, variable=self.var,
                       bg=BG_CARD, fg=FG_MID, selectcolor=BG_RAISED,
                       activebackground=BG_CARD, activeforeground=ACCENT_C,
                       font=F_SM, anchor="w", indicatoron=True,
                       command=command).pack(side="left")
        if sublabel:
            tk.Label(self, text=sublabel, bg=BG_CARD,
                     fg=FG_DIM, font=F_TINY).pack(side="left", padx=4)
    def get(self):   return self.var.get()
    def reset(self): self.var.set(self._default)

# ─────────────────────────────────────────────────────────────
#  UI helpers
# ─────────────────────────────────────────────────────────────
def section_header(parent, text, color=ACCENT_C):
    f = tk.Frame(parent, bg=BG_PANEL)
    f.pack(fill="x", pady=(3,2))
    tk.Label(f, text=f"  {text}  ", bg=color, fg="white",
             font=F_SM).pack(side="left")
    tk.Frame(f, bg=BORDER, height=1).pack(side="left", fill="x",
                                          expand=True, padx=4)
    return f

def card_frame(parent, title=None, color=ACCENT_C, subtitle=None):
    outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
    outer.pack(fill="both", expand=True, padx=6, pady=1)
    inner = tk.Frame(outer, bg=BG_CARD, padx=8, pady=3)
    inner.pack(fill="both", expand=True)
    if title:
        hf = tk.Frame(inner, bg=BG_CARD)
        hf.pack(fill="x", pady=(0,2))
        tk.Label(hf, text=title, bg=BG_CARD, fg=color,
                 font=F_BOLD).pack(side="left")
        if subtitle:
            tk.Label(hf, text=f"  {subtitle}", bg=BG_CARD,
                     fg=FG_DIM, font=F_TINY).pack(side="left")
    return inner

def _stretch_cards(parent):
    """Make all card_frame outer border frames fill their parent vertically.
    Call this after building all cards inside a grid cell."""
    for w in parent.winfo_children():
        if isinstance(w, tk.Frame) and w.cget("bg") == BORDER:
            w.pack_configure(fill="both", expand=True)

# ─────────────────────────────────────────────────────────────
#  ZoomWindow
# ─────────────────────────────────────────────────────────────
class ZoomWindow:
    def __init__(self, root, title, get_pil_fn):
        self.get_pil_fn = get_pil_fn
        self._zoom = 1.0; self._pan_x = 0; self._pan_y = 0
        self._drag_x = 0; self._drag_y = 0
        win = tk.Toplevel(root); win.title(title)
        win.geometry("900x720"); win.configure(bg=BG_PANEL)
        self.win = win
        tb = tk.Frame(win, bg=BG_PANEL); tb.pack(fill="x", padx=6, pady=4)
        def zb(txt, cmd, w=3):
            return tk.Button(tb, text=txt, command=cmd, bg=BG_RAISED,
                             fg=FG_MID, font=F_BOLD, relief="flat",
                             cursor="hand2", width=w, padx=4, pady=2,
                             activebackground=BTN_ACTIVE)
        zb("−", self._zoom_out).pack(side="left", padx=(0,1))
        self.zoom_lbl = tk.Label(tb, text="100%", bg=BG_RAISED,
                                 fg=ACCENT_C, font=F_BOLD, width=6,
                                 relief="flat", padx=4)
        self.zoom_lbl.pack(side="left")
        zb("+", self._zoom_in).pack(side="left", padx=(1,6))
        for pct, lbl in [(25,"25%"),(50,"50%"),(100,"100%"),(200,"200%"),(400,"400%")]:
            zb(lbl, lambda v=pct: self._set_zoom(v/100), w=4).pack(side="left",padx=1)
        zb("Fit", self._fit).pack(side="left", padx=(6,0))
        tk.Label(tb, text="  scroll/+− zoom  ·  drag pan",
                 bg=BG_PANEL, fg=FG_DIM, font=F_HINT).pack(side="left", padx=6)
        self.canvas = tk.Canvas(win, bg="#606060", highlightthickness=0, cursor="fleur")
        self.canvas.pack(fill="both", expand=True, padx=4, pady=(0,4))
        for ev in ("<MouseWheel>","<Button-4>","<Button-5>"):
            self.canvas.bind(ev, self._on_wheel)
        self.canvas.bind("<ButtonPress-1>", lambda e: setattr(self,'_drag_x',e.x) or setattr(self,'_drag_y',e.y))
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        win.bind("<plus>",  lambda e: self._zoom_in())
        win.bind("<equal>", lambda e: self._zoom_in())
        win.bind("<minus>", lambda e: self._zoom_out())
        self._redraw()

    def _zoom_in(self):  self._set_zoom(self._zoom * 1.25)
    def _zoom_out(self): self._set_zoom(self._zoom / 1.25)
    def _set_zoom(self, z):
        self._zoom = max(0.05, min(32.0, z))
        self.zoom_lbl.configure(text=f"{self._zoom*100:.0f}%")
        self._redraw()
    def _fit(self):
        pil = self.get_pil_fn()
        if pil is None: return
        cw = self.canvas.winfo_width() or 900
        ch = self.canvas.winfo_height() or 680
        self._zoom = min(cw/pil.width, ch/pil.height)*0.95
        self._pan_x = self._pan_y = 0
        self.zoom_lbl.configure(text=f"{self._zoom*100:.0f}%")
        self._redraw()
    def _on_wheel(self, e):
        f = 1.15 if not (e.num==5 or getattr(e,"delta",0)<0) else 1/1.15
        self._set_zoom(self._zoom * f)
    def _on_drag(self, e):
        self._pan_x += e.x - self._drag_x; self._pan_y += e.y - self._drag_y
        self._drag_x = e.x; self._drag_y = e.y; self._redraw()
    def _redraw(self):
        pil = self.get_pil_fn()
        if pil is None: return
        cw = self.canvas.winfo_width() or 900
        ch = self.canvas.winfo_height() or 680
        nw = max(1, int(pil.width*self._zoom)); nh = max(1, int(pil.height*self._zoom))
        resample = Image.NEAREST if self._zoom >= 1.0 else Image.LANCZOS
        res = pil.resize((nw,nh), resample)
        x=(cw-nw)//2+self._pan_x; y=(ch-nh)//2+self._pan_y
        t = ImageTk.PhotoImage(res)
        self.canvas.delete("all"); self.canvas.create_image(x,y,anchor="nw",image=t)
        self.canvas._img = t

# ─────────────────────────────────────────────────────────────
#  Click-to-lock Magnifier
# ─────────────────────────────────────────────────────────────
class StaticMagnifier:
    MAG_SIZE    = 440
    # patch_src = source pixels cropped; canvas fills to MAG_SIZE
    # x1 = patch equals canvas = 1:1 (same view as preview)
    # x2 = half patch, x4 = quarter patch
    ZOOM_LEVELS = [(440,"x1"),(220,"x2"),(110,"x4")]
    DEFAULT_IDX = 1     # start at x2

    def __init__(self, parent):
        self._zoom_idx  = self.DEFAULT_IDX
        self._last_pil  = None
        self._last_px   = 0
        self._last_py   = 0
        self._last_name = ""

        outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
        outer.pack(fill="x", padx=6, pady=(0,6))
        inner = tk.Frame(outer, bg=BG_CARD, padx=6, pady=6)
        inner.pack(fill="both", expand=True)

        hdr = tk.Frame(inner, bg=BG_CARD)
        hdr.pack(fill="x", pady=(0,4))

        tk.Label(hdr, text="MAGNIFIER", bg=BG_CARD,
                 fg=ACCENT_C, font=F_SM).pack(side="left")

        zf = tk.Frame(hdr, bg=BG_CARD)
        zf.pack(side="left", padx=(8,0))
        tk.Button(zf, text="v", command=self._zoom_out,
                  bg=BG_RAISED, fg=FG_MID, font=F_TINY,
                  relief="flat", cursor="hand2", padx=4, pady=0,
                  activebackground=BTN_ACTIVE).pack(side="left")
        self._zoom_lbl = tk.Label(zf, text=self.ZOOM_LEVELS[self._zoom_idx][1],
                                  bg=BG_RAISED, fg=ACCENT_C, font=F_SM,
                                  width=4, anchor="center")
        self._zoom_lbl.pack(side="left", padx=2)
        tk.Button(zf, text="^", command=self._zoom_in,
                  bg=BG_RAISED, fg=FG_MID, font=F_TINY,
                  relief="flat", cursor="hand2", padx=4, pady=0,
                  activebackground=BTN_ACTIVE).pack(side="left")

        self.info_lbl = tk.Label(hdr, text="click a preview to magnify",
                                 bg=BG_CARD, fg=FG_DIM, font=F_HINT)
        self.info_lbl.pack(side="right")

        self.canvas = tk.Canvas(inner, width=self.MAG_SIZE, height=self.MAG_SIZE,
                                bg="#cccccc", highlightthickness=1,
                                highlightbackground=BORDER)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self._draw_idle()

    def _zoom_in(self):
        if self._zoom_idx < len(self.ZOOM_LEVELS)-1:
            self._zoom_idx += 1
            self._zoom_lbl.configure(text=self.ZOOM_LEVELS[self._zoom_idx][1])
            self._redraw()

    def _zoom_out(self):
        if self._zoom_idx > 0:
            self._zoom_idx -= 1
            self._zoom_lbl.configure(text=self.ZOOM_LEVELS[self._zoom_idx][1])
            self._redraw()

    def _redraw(self):
        if self._last_pil is not None:
            self.click_magnify(self._last_pil, self._last_px,
                               self._last_py, self._last_name)

    def _on_resize(self, event):
        w = event.width; h = event.height
        if w > 10 and h > 10:
            self.MAG_SIZE = min(w, h)
            self._redraw() if self._last_pil is not None else self._draw_idle()

    def _draw_idle(self):
        self.canvas.delete("all")
        sz = self.canvas.winfo_width() or self.MAG_SIZE
        self.canvas.create_text(sz//2, (self.canvas.winfo_height() or self.MAG_SIZE)//2,
                                text="click on\noriginal or processed\nto magnify",
                                fill=FG_DIM, font=F_SM, justify="center")

    def click_magnify(self, pil, px, py, source_name):
        self._last_pil  = pil
        self._last_px   = px
        self._last_py   = py
        self._last_name = source_name
        patch_src = self.ZOOM_LEVELS[self._zoom_idx][0]
        iw, ih = pil.size
        half   = patch_src // 2
        patch  = pil.crop((max(0,px-half), max(0,py-half),
                           min(iw,px+half), min(ih,py+half)))
        cw = max(self.canvas.winfo_width(),  self.MAG_SIZE)
        ch = max(self.canvas.winfo_height(), self.MAG_SIZE)
        # Scale patch uniformly to COVER the canvas (no gray bars, no distortion).
        # Use the larger scale factor so the image fills the pane on both axes,
        # then center-crop the overflow.
        pw, ph2 = patch.size
        if pw < 1: pw = 1
        if ph2 < 1: ph2 = 1
        scale = max(cw / pw, ch / ph2)
        sw = max(1, int(pw * scale)); sh = max(1, int(ph2 * scale))
        zoomed = patch.resize((sw, sh), Image.NEAREST)
        # Center-crop to canvas size
        ox = (sw - cw) // 2; oy = (sh - ch) // 2
        zoomed = zoomed.crop((ox, oy, ox + cw, oy + ch))
        d   = ImageDraw.Draw(zoomed)
        cx  = cw // 2; cy = ch // 2
        arm = max(30, min(cw, ch) // 10)
        d.line([(cx-arm,cy),(cx+arm,cy)], fill="#ff0000", width=2)
        d.line([(cx,cy-arm),(cx,cy+arm)], fill="#ff0000", width=2)
        t = ImageTk.PhotoImage(zoomed)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=t)
        self.canvas._img = t
        self.info_lbl.configure(text=f"{source_name}  ({px}, {py})")


# ─────────────────────────────────────────────────────────────
#  Annotated line histogram
# ─────────────────────────────────────────────────────────────
def draw_line_histogram(canvas, arr: np.ndarray, height: int):
    """
    RGB line histogram, log-scale, 0–65535 x-axis.
    Annotated regions: shadows (left), midtones (center), highlights (right).
    """
    canvas.delete("all")
    w  = canvas.winfo_width()
    if w < 100:
        try: w = canvas.winfo_reqwidth()
        except Exception: w = 800
    if w < 100: w = 800
    w = max(w, 100)
    h  = height
    L  = 8; BOT = 40; TOP = 20; R = 8
    pw = w - L - R
    ph = h - TOP - BOT

    BINS = 256
    rh, _ = np.histogram(arr[...,0], bins=BINS, range=(0,1))
    gh, _ = np.histogram(arr[...,1], bins=BINS, range=(0,1))
    bh, _ = np.histogram(arr[...,2], bins=BINS, range=(0,1))

    # ── Region shading ──
    # shadow 0–10%
    xs = L + int(0.10 * pw)
    canvas.create_rectangle(L, TOP, xs, TOP+ph, fill="#e8f0f8", outline="")
    # midtone 10–90%
    xm = L + int(0.90 * pw)
    canvas.create_rectangle(xs, TOP, xm, TOP+ph, fill="#f8f9fa", outline="")
    # highlight 90–100%
    canvas.create_rectangle(xm, TOP, L+pw, TOP+ph, fill="#fff3e0", outline="")

    # ── Region labels (tiny, inside bands) ──
    for txt, xc, col in [
        ("SHADOWS",    L + int(0.05*pw) + 14, "#6090c0"),
        ("MIDTONES",   L + int(0.50*pw), "#60a060"),
        ("HIGHLIGHTS", L + int(0.91*pw), "#c08040"),
    ]:
        canvas.create_text(xc, TOP+4, text=txt, fill=col,
                           font=F_TINY, anchor="n")

    # ── Clipping warning lines ──
    canvas.create_line(L+1,     TOP, L+1,     TOP+ph, fill="#6090c0", dash=(3,3), width=1)
    canvas.create_line(L+pw-1,  TOP, L+pw-1,  TOP+ph, fill="#c08040", dash=(3,3), width=1)

    # ── Grid — every 10000 units ──
    for tv in range(0, 65536, 10000):
        x = L + int(tv/65535*pw)
        canvas.create_line(x, TOP, x, TOP+ph, fill="#d8dde4", dash=(2,2))

    # ── Plot background border ──
    canvas.create_rectangle(L, TOP, L+pw, TOP+ph, fill="", outline=BORDER)

    # ── Channel lines ──
    all_max = max(np.log1p(rh).max(), np.log1p(gh).max(), np.log1p(bh).max())
    if all_max < 1: all_max = 1

    for hist, color, width in [
        (bh, "#2563eb", 1),   # blue first (back)
        (gh, "#15803d", 1),
        (rh, "#dc2626", 1),
    ]:
        lh = np.log1p(hist.astype(float))
        pts = []
        for i, v in enumerate(lh):
            x = L + int(i/(BINS-1)*pw)
            y = TOP + ph - int(v/all_max*ph)
            pts.append((x, y))
        for i in range(len(pts)-1):
            canvas.create_line(pts[i][0], pts[i][1],
                               pts[i+1][0], pts[i+1][1],
                               fill=color, width=width)

    # ── X axis — ticks every 10000 units ──
    canvas.create_line(L, TOP+ph, L+pw, TOP+ph, fill=FG_DIM)
    for tv in range(0, 65536, 10000):
        x = L + int(tv/65535*pw)
        lbl = str(tv)
        canvas.create_line(x, TOP+ph, x, TOP+ph+3, fill=FG_DIM)
        canvas.create_text(x, TOP+ph+10, text=lbl, fill=FG_DIM,
                           font=F_XS, anchor="center")
    # ── Y axis line ──
    canvas.create_line(L, TOP, L, TOP+ph, fill=FG_DIM)
    canvas.create_text(L+4, TOP+2, text="▲", fill=FG_DIM, font=F_XS, anchor="nw")
    canvas.create_text(L+4, TOP+ph-2, text="0", fill=FG_DIM, font=F_XS, anchor="sw")

    # ── Clip annotation ──
    canvas.create_text(L+2,   TOP+ph+24, text="◀ clipped blacks",
                       fill="#6090c0", font=F_XS, anchor="w")
    canvas.create_text(L+pw-2, TOP+ph+24, text="clipped whites ▶",
                       fill="#c08040", font=F_XS, anchor="e")

# ─────────────────────────────────────────────────────────────
#  Main Application
# ─────────────────────────────────────────────────────────────
class KeplerApp:
    # Debounce delay for live processing (ms)
    LIVE_DELAY = 400

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Kepler — Planetary Image Processing Suite")
        self.root.configure(bg=BG_DEEP)
        self.root.minsize(900, 700)

        prefs = load_prefs()
        if (geom := prefs.get("window_geometry")):
            try: self.root.geometry(geom)
            except Exception: pass
        if sys.platform == "darwin":
            try:
                self.root.update_idletasks()
                sw = self.root.winfo_screenwidth()
                sh = self.root.winfo_screenheight()
                # Leave room for menu bar, dock, and visible resize border
                w = sw - 20
                h = sh - 120
                x = 10
                y = 40
                self.root.geometry(f"{w}x{h}+{x}+{y}")
            except Exception:
                self.root.geometry("1400x900")
        else:
            try: self.root.state(prefs.get("window_state","zoomed"))
            except Exception:
                try: self.root.attributes("-zoomed",True)
                except Exception: self.root.geometry("1050x900")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.original_pil: Image.Image = None
        self.working_pil:  Image.Image = None
        self.original_arr: np.ndarray  = None
        self.working_arr:  np.ndarray  = None
        self._source_path: str         = None
        prefs2 = load_prefs()
        self._project_dir: str = prefs2.get("project_dir", "")
        self._image_dir:   str = prefs2.get("image_dir", "")
        self.processing   = False
        self._fft_spectrum = None
        self._live_after  = None
        self._live_pending = False   # pending after() id
        self._pending_tab  = 0       # which tab triggered last live update

        # Per-tab slider registries for per-tab reset
        self._tab_sliders: dict[int, list] = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
        self._current_tab_index = 0

        self._apply_ttk_theme()
        self._build_ui()

    def _on_close(self):
        try:
            prefs = {
                "window_state":    self.root.state() if sys.platform != "darwin" else "zoomed",
                "window_geometry": self.root.geometry(),
            }
            # Save horizontal main sash (left panel width)
            try:
                prefs["main_sash"] = self._main_pane.sash_coord(0)[0]
            except Exception: pass
            # Save vertical sash positions (preview/magnifier/histogram heights)
            try:
                prefs["vert_sash0"] = self._vert_pane.sash_coord(0)[1]
                prefs["vert_sash1"] = self._vert_pane.sash_coord(1)[1]
            except Exception: pass
            try: prefs["project_dir"] = self._project_dir
            except Exception: pass
            try: prefs["image_dir"] = self._image_dir
            except Exception: pass
            try: prefs["derotate_planet"] = self._dr2_planet.get()
            except Exception: pass
            save_prefs(prefs)
        except Exception: pass
        self.root.destroy()

    def _restore_sash_positions(self):
        """Restore saved sash positions from prefs."""
        if sys.platform == "darwin": return
        prefs = load_prefs()
        try:
            if "main_sash" in prefs:
                total_w = self.root.winfo_width()
                sash = min(prefs["main_sash"], max(total_w - 460, 390), int(total_w * 0.45))
                self._main_pane.sash_place(0, sash, 0)
        except Exception: pass
        try:
            if "vert_sash0" in prefs:
                self._vert_pane.sash_place(0, 0, prefs["vert_sash0"])
            if "vert_sash1" in prefs:
                self._vert_pane.sash_place(1, 0, prefs["vert_sash1"])
        except Exception: pass

    def _reset_sash_positions(self):
        """Reset all interior pane sizes to defaults."""
        self.root.update_idletasks()
        try:
            total_w = self.root.winfo_width()
            left_w  = max(390, min(int(total_w * 0.45), total_w - 460))
            if sys.platform == "darwin":
                panes = self._main_pane.panes()
                if panes:
                    self._main_pane.paneconfigure(panes[0], width=left_w)
                    self.root.update_idletasks()
                    self._main_pane.paneconfigure(panes[0], width=left_w)
            else:
                self._main_pane.sash_place(0, left_w, 0)
        except Exception: pass
        try:
            total_h = self._vert_pane.winfo_height()
            if sys.platform == "darwin":
                panes = self._vert_pane.panes()
                if panes:
                    self._vert_pane.paneconfigure(panes[0], height=int(total_h * 0.60))
                    self.root.update_idletasks()
                    self._vert_pane.paneconfigure(panes[0], height=int(total_h * 0.60))
            else:
                self._vert_pane.sash_place(0, 0, int(total_h * 0.50))
                self._vert_pane.sash_place(1, 0, int(total_h * 0.85))
        except Exception: pass
        # Restore Show Original if it was hidden
        try:
            if not self._show_orig.get():
                self._show_orig.set(True)
                self._toggle_show_original()
        except Exception: pass

    def _apply_ttk_theme(self):
        s = ttk.Style()
        try:
            if sys.platform == "darwin":
                s.theme_use("default")
            else:
                s.theme_use("clam")
        except Exception: pass
        s.configure("TNotebook", background=BG_PANEL, borderwidth=0)
        s.configure("TNotebook.Tab", background=BG_RAISED, foreground=FG_MID,
                    padding=[8,3], font=F_HINT)
        s.map("TNotebook.Tab",
              background=[("selected", BG_CARD)],
              foreground=[("selected", ACCENT_C)])
        s.configure("TProgressbar",
                    troughcolor=SLIDER_TRG, background=ACCENT_C,
                    lightcolor=ACCENT_C, darkcolor=ACCENT_C, bordercolor=BORDER)
        s.configure("Vertical.TScrollbar",
                    background=BG_RAISED, troughcolor=BG_PANEL, arrowcolor=FG_MID)
        s.configure("TCombobox",
                    fieldbackground="#ffffff", background=BG_RAISED,
                    foreground=FG_BRIGHT, selectbackground=SEL_BG, font=F_SM)

    # ── Build UI ────────────────────────────────────────────
    def _build_ui(self):
        self._build_header()
        self._main_pane = tk.PanedWindow(self.root, orient="horizontal",
                              bg=BG_DEEP, sashwidth=5, sashrelief="flat")
        self._main_pane.pack(fill="both", expand=True, padx=6, pady=(0,6))
        left = tk.Frame(self._main_pane, bg=BG_PANEL, width=390)
        self._main_pane.add(left, minsize=400)
        self._build_image_panel(left)
        right = tk.Frame(self._main_pane, bg=BG_PANEL)
        self._main_pane.add(right, minsize=360)
        self._build_controls(right)
        # Restore saved sash positions after layout is complete
        self.root.after(100, self._restore_sash_positions)

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=BG_PANEL, height=46)
        hdr.pack(fill="x", padx=6, pady=(6,4))
        hdr.pack_propagate(False)
        # Load icon from file; fall back to drawn planet if unavailable
        _icon_shown = False
        try:
            import os
            from PIL import Image, ImageTk
            _ico_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
            if os.path.isfile(_ico_path):
                _pil = Image.open(_ico_path)
                _sizes = _pil.info.get("sizes", {_pil.size})
                _best  = max(_sizes, key=lambda s: s[0])
                _pil   = _pil.resize((32, 32), Image.LANCZOS).convert("RGBA")
                self._header_icon = ImageTk.PhotoImage(_pil)
                tk.Label(hdr, image=self._header_icon, bg=BG_PANEL).pack(side="left", padx=(10,6), pady=5)
                _icon_shown = True
        except Exception:
            pass
        if not _icon_shown:
            ic = tk.Canvas(hdr, width=36, height=36, bg=BG_PANEL, highlightthickness=0)
            ic.pack(side="left", padx=(10,6), pady=5)
            ic.create_oval(3,3,33,33, fill="#e8760a", outline="#b45309", width=2)
            ic.create_oval(6,6,30,30, fill="#c05a05", outline="")
            ic.create_oval(0,14,36,22, outline="#92400e", width=2, fill="")
        tk.Label(hdr, text="Kepler", bg=BG_PANEL, fg=ACCENT_C, font=F_HDR).pack(side="left")
        tk.Label(hdr, text="  —  Planetary Image Processing Suite  ·  v1.3.4",
                 bg=BG_PANEL, fg=FG_DIM, font=F_HINT).pack(side="left")
        self.status_var = tk.StringVar(value="READY")
        tk.Label(hdr, textvariable=self.status_var, bg=BG_PANEL,
                 fg=ACCENT_G, font=F_BOLD).pack(side="right", padx=12)
        tk.Label(hdr, text="STATUS:", bg=BG_PANEL,
                 fg=FG_DIM, font=F_HINT).pack(side="right")
        self.img_size_var = tk.StringVar(value="—")
        tk.Label(hdr, textvariable=self.img_size_var, bg=BG_PANEL,
                 fg=ACCENT_C, font=F_HINT).pack(side="right", padx=(0,12))
        tk.Label(hdr, text="SIZE:", bg=BG_PANEL,
                 fg=FG_DIM, font=F_HINT).pack(side="right")
        tk.Button(hdr, text="⚠ Reset Layout",
                  command=self._reset_sash_positions,
                  bg=BTN_BG, fg=FG_MID, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=8, pady=2,
                  activebackground=BTN_ACTIVE).pack(side="right", padx=(0, 16))

    # ── Left panel ──────────────────────────────────────────
    def _build_image_panel(self, parent):
        # Top strip: IMAGE I/O buttons (fixed height, not in paned window)
        top = tk.Frame(parent, bg=BG_PANEL)
        top.pack(fill="x", padx=6, pady=(6,2))
        section_header(top, "IMAGE I/O").pack(fill="x", pady=(0,2))
        bf = tk.Frame(top, bg=BG_PANEL); bf.pack(fill="x", pady=3)
        def _btn(p, text, cmd, fg):
            return tk.Button(p, text=text, command=cmd, bg=BTN_BG,
                             fg=fg, font=F_MD, relief="groove", bd=2, cursor="hand2",
                             padx=10, pady=5, activebackground=BTN_ACTIVE,
                             activeforeground=fg)
        _btn(bf,"📂 Open",   self.open_image,    ACCENT_C).pack(side="left",fill="x",expand=True,padx=(0,3))
        _btn(bf,"💾 Export", self.export_image,  ACCENT_G).pack(side="left",fill="x",expand=True,padx=(0,3))

        # Progress bar (fixed, always visible at bottom of top strip)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(top, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=(4,2))

        # ── PROJECTS section ────────────────────────────────────────────────
        proj_outer = tk.Frame(parent, bg=BG_PANEL)
        proj_outer.pack(fill="x", padx=6, pady=(2,2))
        section_header(proj_outer, "PROJECTS").pack(fill="x", pady=(0,2))

        # Object name + date row
        pn_row = tk.Frame(proj_outer, bg=BG_PANEL); pn_row.pack(fill="x", pady=(0,3))
        tk.Label(pn_row, text="Object:", bg=BG_PANEL, fg=FG_MID, font=F_SM).pack(side="left", padx=(0,4))
        self._proj_object = tk.StringVar(value="")
        tk.Entry(pn_row, textvariable=self._proj_object, bg=BG_RAISED, fg=FG_MID,
                 font=F_SM, relief="flat", width=12).pack(side="left", padx=(0,6))
        tk.Label(pn_row, text="Date:", bg=BG_PANEL, fg=FG_MID, font=F_SM).pack(side="left", padx=(0,4))
        import datetime as _dt
        self._proj_date = tk.StringVar(value=_dt.date.today().strftime("%m-%d-%Y"))
        tk.Entry(pn_row, textvariable=self._proj_date, bg=BG_RAISED, fg=FG_MID,
                 font=F_SM, relief="flat", width=10).pack(side="left")

        # Save / Open / Folder row
        pb_row = tk.Frame(proj_outer, bg=BG_PANEL); pb_row.pack(fill="x", pady=(0,2))
        def _pbtn(p, text, cmd, fg):
            return tk.Button(p, text=text, command=cmd, bg=BTN_BG,
                             fg=fg, font=F_SM, relief="groove", bd=2, cursor="hand2",
                             padx=8, pady=4, activebackground=BTN_ACTIVE, activeforeground=fg)
        _pbtn(pb_row, "💾 Save Project",  self._project_save,   ACCENT_C).pack(side="left", fill="x", expand=True, padx=(0,3))
        _pbtn(pb_row, "📂 Open Project",  self._project_open,   ACCENT_G).pack(side="left", fill="x", expand=True, padx=(0,3))
        _pbtn(pb_row, "📁",               self._project_choose_dir, FG_MID).pack(side="left", padx=(0,0))

        # Folder label
        self._proj_dir_lbl = tk.Label(proj_outer, text=self._fmt_proj_dir(),
                                      bg=BG_PANEL, fg=FG_DIM, font=F_HINT,
                                      anchor="w", wraplength=600)
        self._proj_dir_lbl.pack(fill="x", pady=(0,2))

        # ── Vertical PanedWindow: Preview | Magnifier | Histogram ───────────
        self._vert_pane = tk.PanedWindow(parent, orient="vertical",
                               bg=BG_DEEP, sashwidth=6, sashrelief="flat",
                               sashpad=2)
        self._vert_pane.pack(fill="both", expand=True, padx=6, pady=(2,6))
        vpane = self._vert_pane

        # ── Pane 1: Preview ──────────────────────────────────────────────────
        prev_frm = tk.Frame(vpane, bg=BG_PANEL)
        vpane.add(prev_frm, minsize=180)

        section_header(prev_frm,"PREVIEW").pack(fill="x", pady=(4,2))
        zf = tk.Frame(prev_frm, bg=BG_PANEL); zf.pack(fill="x", pady=(0,3))
        tk.Label(zf, text="Zoom:", bg=BG_PANEL, fg=FG_MID, font=F_SM).pack(side="left")
        self._prev_zoom = 1.0
        self._pan = {}
        def _zbtn(txt, cmd):
            return tk.Button(zf, text=txt, command=cmd, bg=BG_RAISED,
                             fg=FG_MID, font=F_BOLD, relief="flat",
                             cursor="hand2", width=3, padx=2, pady=1,
                             activebackground=BTN_ACTIVE)
        _zbtn("−", self._preview_zoom_out).pack(side="left", padx=(4,1))
        self._zoom_pct_var = tk.StringVar(value="Fit")
        tk.Label(zf, textvariable=self._zoom_pct_var, bg=BG_RAISED,
                 fg=ACCENT_C, font=F_BOLD, width=5, anchor="center",
                 relief="flat").pack(side="left")
        _zbtn("+", self._preview_zoom_in).pack(side="left", padx=(1,4))
        for pct, lbl in [(50,"50%"),(100,"100%"),(200,"200%"),(400,"400%")]:
            tk.Button(zf, text=lbl, command=lambda v=pct: self._set_preview_zoom(v/100),
                      bg=BTN_BG, fg=FG_MID, font=F_TINY, relief="groove", bd=2,
                      cursor="hand2", padx=4, pady=1,
                      activebackground=BTN_ACTIVE).pack(side="left", padx=1)
        tk.Button(zf, text="Fit", command=self._preview_fit,
                  bg=BTN_BG, fg=ACCENT_G, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=6, pady=1,
                  activebackground=BTN_ACTIVE).pack(side="left", padx=(4,0))
        # Show Original checkbox — right side of zoom row
        self._show_orig = tk.BooleanVar(value=True)
        tk.Checkbutton(
            zf, text="Show Original", variable=self._show_orig,
            font=F_TINY, bg=BG_PANEL, fg=FG_MID,
            selectcolor=BG_RAISED, activebackground=BG_PANEL,
            cursor="hand2", command=self._toggle_show_original
        ).pack(side="right", padx=(8,0))

        # ── Preview Rotation row ─────────────────────────────────────────
        rf = tk.Frame(prev_frm, bg=BG_PANEL); rf.pack(fill="x", pady=(0,2))
        tk.Label(rf, text="Rotate:", bg=BG_PANEL, fg=FG_MID, font=F_SM).pack(side="left", padx=(0,4))
        self._preview_rot_deg = tk.DoubleVar(value=0.0)
        self._chan_view = tk.StringVar(value="RGB")
        self._preview_rot_spinbox = tk.Spinbox(
            rf, from_=-360.0, to=360.0, increment=1.0,
            textvariable=self._preview_rot_deg,
            width=6, font=F_MD, format="%.1f", relief="flat",
            bg=BG_RAISED, fg=ACCENT_C, buttonbackground=BG_RAISED,
            activebackground=BTN_ACTIVE,
            command=self._on_preview_rotate)
        self._preview_rot_spinbox.pack(side="left", padx=(0,4))
        self._preview_rot_spinbox.bind("<Return>",   lambda e: self._on_preview_rotate())
        self._preview_rot_spinbox.bind("<FocusOut>", lambda e: self._on_preview_rotate())
        self._bind_spinbox(self._preview_rot_spinbox, self._preview_rot_deg, -360.0, 360.0, 1.0, -1, callback=self._on_preview_rotate)
        tk.Label(rf, text="°", bg=BG_PANEL, fg=FG_MID, font=F_SM).pack(side="left", padx=(0,6))
        for deg, lbl in [(-90,"−90°"), (90,"90°"), (180,"180°")]:
            tk.Button(rf, text=lbl,
                      command=lambda d=deg: self._preview_set_rotation(d),
                      bg=BTN_BG, fg=FG_MID, font=F_TINY, relief="groove", bd=1,
                      cursor="hand2", padx=5, pady=1,
                      activebackground=BTN_ACTIVE).pack(side="left", padx=1)
        tk.Button(rf, text="⊘ Reset",
                  command=lambda: self._preview_set_rotation(0),
                  bg=BTN_BG, fg=FG_MID, font=F_HINT, relief="groove", bd=1,
                  cursor="hand2", padx=5, pady=1,
                  activebackground=BTN_ACTIVE).pack(side="left", padx=(4,0))
        # ── Channel View buttons — right of rotate controls ──
        tk.Frame(rf, bg=BORDER, width=1).pack(side="left", fill="y", padx=(8,4))
        tk.Label(rf, text="    View:", bg=BG_PANEL, fg=FG_DIM, font=F_HINT).pack(side="left")
        self._chan_btns = {}
        for ch in ["RGB", "L", "R", "G", "B", "S"]:
            btn = tk.Button(rf, text=ch, font=F_TINY, width=3,
                            relief="flat", cursor="hand2", padx=2, pady=1,
                            bg=ACCENT_C if ch=="RGB" else BG_RAISED,
                            fg="white" if ch=="RGB" else FG_MID,
                            activebackground=BTN_ACTIVE,
                            command=lambda c=ch: self._set_chan_view(c))
            btn.pack(side="left", padx=1)
            self._chan_btns[ch] = btn

        pf = tk.Frame(prev_frm, bg=BG_PANEL); pf.pack(fill="both", expand=True, pady=3)
        def _preview_canvas(parent, title):
            border = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
            border.pack(side="left", expand=True, fill="both",
                        padx=(0,2) if title=="ORIGINAL" else (2,0))
            inner = tk.Frame(border, bg=BG_CARD); inner.pack(fill="both", expand=True)
            lbl = tk.Label(inner, text=title, bg=BG_CARD,
                           fg=FG_DIM, font=F_SM)
            lbl.pack(pady=(3,0))
            c = tk.Canvas(inner, width=PREV_W, height=PREV_H,
                          bg="#606060", highlightthickness=0, cursor="crosshair")
            c.pack(fill="both", expand=True, padx=3, pady=3)
            return c, lbl, border, inner

        self.cnv_orig, _orig_lbl, self._orig_border, _orig_inner = _preview_canvas(pf, "ORIGINAL")
        self.cnv_proc, self._proc_lbl, self._proc_border, _proc_inner = _preview_canvas(pf, "PROCESSED")
        # (channel buttons moved to above the preview panels — see cv_row below)

        for c in (self.cnv_orig, self.cnv_proc):
            c.bind("<MouseWheel>", lambda e,cc=c: self._preview_wheel(e,cc))
            c.bind("<Button-4>",   lambda e,cc=c: self._preview_wheel(e,cc))
            c.bind("<Button-5>",   lambda e,cc=c: self._preview_wheel(e,cc))
            self._pan[id(c)] = [0, 0, 0, 0]
            c.bind("<ButtonPress-2>",   lambda e,cc=c: self._pan_start(e,cc))
            c.bind("<B2-Motion>",       lambda e,cc=c: self._pan_move(e,cc))
            c.bind("<ButtonRelease-2>", lambda e,cc=c: self._pan_end(e,cc))
            c.bind("<ButtonPress-3>",   lambda e,cc=c: self._pan_start(e,cc))
            c.bind("<B3-Motion>",       lambda e,cc=c: self._pan_move(e,cc))
            c.bind("<ButtonRelease-3>", lambda e,cc=c: self._pan_end(e,cc))
        self.cnv_orig.bind("<Double-Button-1>",
            lambda e: ZoomWindow(self.root,"Zoom — Original",lambda: self.original_pil))
        self.cnv_proc.bind("<Double-Button-1>",
            lambda e: ZoomWindow(self.root,"Zoom — Processed",
                lambda: self._get_processed_pil() or self.original_pil))
        self._set_placeholder_previews()

        # ── Pane 2: Magnifier ────────────────────────────────────────────────
        mag_frm = tk.Frame(vpane, bg=BG_PANEL)
        vpane.add(mag_frm, minsize=120)
        section_header(mag_frm,"MAGNIFIER  (click preview to lock spot)").pack(fill="x", pady=(4,2))
        self.magnifier = StaticMagnifier(mag_frm)
        self.cnv_orig.bind("<Button-1>",
            lambda e: self._magnify_at(e,self.cnv_orig,lambda: self.original_pil,"ORIG"))
        self.cnv_proc.bind("<Button-1>",
            lambda e: self._magnify_at(e,self.cnv_proc,
                lambda: self._get_processed_pil()
                        if self.working_arr is not None else self.original_pil,"PROC"))

        # Histogram moved to RGB tab

    # ── Preview zoom helpers ─────────────────────────────────
    def _preview_zoom_in(self):  self._set_preview_zoom(self._prev_zoom*1.25)
    def _preview_zoom_out(self): self._set_preview_zoom(self._prev_zoom/1.25)
    def _set_preview_zoom(self, z):
        self._prev_zoom = max(0.1, min(16.0, z))
        self._zoom_pct_var.set(f"{self._prev_zoom*100:.0f}%")
        self._redraw_previews()
    def _preview_fit(self):
        self._prev_zoom = 1.0; self._zoom_pct_var.set("Fit")
        for k in self._pan: self._pan[k][:2] = [0, 0]
        self._redraw_previews()
    def _pan_start(self, event, cnv):
        p = self._pan[id(cnv)]; p[2] = event.x; p[3] = event.y
        self._is_panning = True
        cnv.configure(cursor="fleur")
    def _pan_move(self, event, cnv):
        p = self._pan[id(cnv)]
        dx = event.x - p[2]; dy = event.y - p[3]
        p[0] += dx; p[1] += dy
        p[2] = event.x; p[3] = event.y
        for item in cnv.find_all():
            cnv.move(item, dx, dy)
    def _pan_redraw(self, cnv):
        pass  # kept for compatibility
    def _pan_end(self, event, cnv):
        self._is_panning = False
        cnv.configure(cursor="" if not getattr(self, "_mr_mode", False) else "crosshair")
    def _preview_wheel(self, event, canvas):
        f = 1.15 if not (event.num==5 or getattr(event,"delta",0)<0) else 1/1.15
        self._set_preview_zoom(self._prev_zoom * f)
    def _get_processed_pil(self):
        """Return the processed image as PIL, with rotation and channel view applied."""
        if self.working_arr is None:
            return None
        pil = proc.array_to_pil(self.working_arr)
        try:
            deg = float(self._preview_rot_deg.get())
        except (ValueError, AttributeError, tk.TclError):
            deg = 0.0
        if deg != 0.0:
            pil = pil.rotate(-deg, expand=False,
                             resample=Image.BICUBIC, fillcolor=(0, 0, 0))
        # Apply channel view transform (display only — does not affect working_arr)
        try:
            ch = self._chan_view.get()
        except AttributeError:
            ch = "RGB"
        if ch != "RGB":
            import numpy as _np
            arr = _np.array(pil).astype(_np.float32) / 255.0
            if ch == "L":
                gray = (0.299*arr[...,0]+0.587*arr[...,1]+0.114*arr[...,2])
                arr = _np.stack([gray,gray,gray], axis=-1)
            elif ch == "R":
                arr = _np.stack([arr[...,0],arr[...,0],arr[...,0]], axis=-1)
            elif ch == "G":
                arr = _np.stack([arr[...,1],arr[...,1],arr[...,1]], axis=-1)
            elif ch == "B":
                arr = _np.stack([arr[...,2],arr[...,2],arr[...,2]], axis=-1)
            elif ch == "S":
                # Luminance-weighted saturation — dark pixels stay dark,
                # so background noise and limb fringing are shown in context
                mx = arr.max(axis=-1); mn = arr.min(axis=-1)
                diff = mx - mn
                lum = 0.299*arr[...,0]+0.587*arr[...,1]+0.114*arr[...,2]
                sat = _np.where(mx > 1e-6, diff / (mx + 1e-6), 0.0)
                sat = sat * lum  # weight by luminance — suppresses dark noise
                sat = _np.clip(sat / (sat.max() + 1e-6), 0.0, 1.0)  # normalize
                arr = _np.stack([sat,sat,sat], axis=-1)
            arr = _np.clip(arr * 255.0, 0, 255).astype(_np.uint8)
            pil = Image.fromarray(arr)
        # Moon preview boost — display-only brightness amplification
        try:
            pb = float(self.mr_preview_boost.get()) / 10.0
        except (AttributeError, ValueError, tk.TclError):
            pb = 1.0
        if pb > 1.01:
            import numpy as _np
            arr = _np.array(pil).astype(_np.float32)
            arr = _np.clip(arr * pb, 0, 255).astype(_np.uint8)
            pil = Image.fromarray(arr)
        return pil

    def _toggle_show_original(self):
        show = self._show_orig.get()
        if show:
            # Restore both canvases in correct order: ORIGINAL left, PROCESSED right
            # Forget both first so we can re-pack in the right order
            self._orig_border.pack_forget()
            self._proc_border.pack_forget()
            self._orig_border.pack(side="left", expand=True, fill="both", padx=(0,2))
            self._proc_border.pack(side="left", expand=True, fill="both", padx=(2,0))
        else:
            # Hide original — processed fills all available space
            self._orig_border.pack_forget()
            self._proc_border.pack_forget()
            self._proc_border.pack(side="left", expand=True, fill="both", padx=0)

    def _set_chan_view(self, ch):
        """Toggle the channel view for the PROCESSED canvas."""
        # Clicking the active button returns to RGB
        current = self._chan_view.get()
        new_ch = "RGB" if (ch == current and ch != "RGB") else ch
        self._chan_view.set(new_ch)
        # Update button appearances
        for c, btn in self._chan_btns.items():
            active = (c == new_ch)
            btn.configure(
                bg=ACCENT_C if active else BG_RAISED,
                fg="white" if active else FG_MID)
        # Update PROCESSED label
        label_txt = f"PROCESSED  [{new_ch}]" if new_ch != "RGB" else "PROCESSED"
        try: self._proc_lbl.configure(text=label_txt)
        except AttributeError: pass
        # Refresh the processed canvas instantly (no pipeline re-run needed)
        pil = self._get_processed_pil()
        if pil is not None:
            self._draw_on_canvas(self.cnv_proc, pil)
            # Also refresh magnifier if it is showing the processed view
            try:
                if self.magnifier._last_pil is not None and self.magnifier._last_name == "PROC":
                    self.magnifier.click_magnify(
                        pil,
                        self.magnifier._last_px,
                        self.magnifier._last_py,
                        "PROC")
            except Exception:
                pass

    def _redraw_previews(self):
        if self.original_arr is not None:
            self._draw_on_canvas(self.cnv_orig, proc.array_to_pil(self.original_arr))
        pil = self._get_processed_pil()
        if pil is not None:
            self._draw_on_canvas(self.cnv_proc, pil)
            if hasattr(self, "_mr_circles"):
                self._mr_update_overlays()
    def _draw_on_canvas(self, cnv, pil):
        cw = cnv.winfo_width()
        ch = cnv.winfo_height()
        if cw <= 1 or ch <= 1:
            cnv.update_idletasks()
            cw = cnv.winfo_width()
            ch = cnv.winfo_height()
        if cw <= 1: cw = PREV_W
        if ch <= 1: ch = PREV_H
        scale = (min(cw/pil.width, ch/pil.height)
                 if self._zoom_pct_var.get()=="Fit" else self._prev_zoom)
        nw = max(1, int(pil.width*scale)); nh = max(1, int(pil.height*scale))
        resample = Image.NEAREST if scale >= 1.0 else Image.LANCZOS
        res = pil.resize((nw,nh), resample)
        p = self._pan.get(id(cnv), [0,0,0,0])
        x=(cw-nw)//2 + p[0]; y=(ch-nh)//2 + p[1]
        t = ImageTk.PhotoImage(res)
        cnv.delete("all"); cnv.create_image(x,y,anchor="nw",image=t); cnv._img=t
        # Always redraw circle overlays on top after any cnv_proc redraw
        if cnv is self.cnv_proc and hasattr(self, "_mr_circles") and \
                not getattr(self, "_mr_suppress_redraw", False) and \
                not getattr(self, "_is_panning", False):
            self._mr_update_overlays()

    def _magnify_at(self, event, cnv, get_pil_fn, source_name):
        pil = get_pil_fn()
        if pil is None: return
        cw = cnv.winfo_width() or PREV_W; ch = cnv.winfo_height() or PREV_H
        scale = (min(cw/pil.width, ch/pil.height)
                 if self._zoom_pct_var.get()=="Fit" else self._prev_zoom)
        nw = max(1,int(pil.width*scale)); nh = max(1,int(pil.height*scale))
        p = self._pan.get(id(cnv), [0,0,0,0])
        ox = (cw-nw)//2 + p[0]
        oy = (ch-nh)//2 + p[1]
        px = max(0, min(pil.width-1,  int((event.x-ox)/scale)))
        py = max(0, min(pil.height-1, int((event.y-oy)/scale)))
        self.magnifier.click_magnify(pil, px, py, source_name)
    def _set_placeholder_previews(self):
        for cnv, txt in [(self.cnv_orig,"NO IMAGE"),(self.cnv_proc,"AWAITING…")]:
            ph = Image.new("RGB",(PREV_W,PREV_H),color="#505864")
            ImageDraw.Draw(ph).text((PREV_W//2-30,PREV_H//2-6),txt,fill="#aab4be")
            t = ImageTk.PhotoImage(ph)
            cnv.delete("all"); cnv.create_image(0,0,anchor="nw",image=t); cnv._img=t

    # ── Right panel ─────────────────────────────────────────
    def _build_controls(self, parent):
        bb = tk.Frame(parent, bg=BG_PANEL)
        bb.pack(fill="x", padx=6, pady=(6,0))
        tk.Button(bb, text="\u21ba  Reset Current Tab Values", command=self.reset_tab_sliders,
                  bg=BTN_BG, fg=ACCENT_O, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=8, pady=4,
                  activebackground=BTN_ACTIVE).pack(side="left", padx=(0,4))
        tk.Button(bb, text="\u26a0  Reset All Tab Values", command=self.reset_all,
                  bg=BTN_BG, fg=ACCENT_R, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=8, pady=4,
                  activebackground=BTN_ACTIVE).pack(side="left", padx=(0,8))
        tk.Label(bb, text="Double-click any slider\nto reset to default",
                 bg=BG_PANEL, fg=FG_DIM, font=F_XS, justify="left").pack(side="left")

        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=4)
        self.notebook.bind("<<NotebookTabChanged>>",
                           lambda e: self._on_tab_change())

        self._build_wavelet_tab()
        self._build_fft_tab()
        self._build_color_tab()
        self._build_tools_tab()
        self._build_dering_tab()
        self._build_orbital_tab()
        self._build_batch_tab()
        self._build_derotate_tab()
        self._build_stats_tab()
        # Refresh scroll regions at multiple delays — content may finish laying
        # out at different times especially at larger font sizes
        for delay in (300, 800, 1500):
            self.notebook.after(delay, self._refresh_all_scroll_regions)

    def _on_tab_change(self):
        try:
            self._current_tab_index = self.notebook.index(self.notebook.select())
        except Exception: pass
        # Refresh scroll region of the newly visible tab using actual frame height
        try:
            tab = self.notebook.nametowidget(self.notebook.select())
            for cnv in tab.winfo_children():
                if isinstance(cnv, tk.Canvas):
                    cnv.update_idletasks()
                    for item in cnv.find_all():
                        if cnv.type(item) == "window":
                            w = cnv.nametowidget(cnv.itemcget(item, "window"))
                            w.update_idletasks()
                            cnv.configure(scrollregion=(0, 0, w.winfo_reqwidth(), w.winfo_reqheight()))
                    break
        except Exception: pass

    def _refresh_all_scroll_regions(self):
        """Also stretches card frames in grid cells to equal height."""
        try:
            for frame in getattr(self, "_grid_cells_to_stretch", []):
                _stretch_cards(frame)
        except Exception: pass
        """Force all scrollable tab canvases to recalculate their scroll region."""
        for tab_id in self.notebook.tabs():
            try:
                tab = self.notebook.nametowidget(tab_id)
                for cnv in tab.winfo_children():
                    if isinstance(cnv, tk.Canvas):
                        for item in cnv.find_all():
                            if cnv.type(item) == "window":
                                inner = cnv.nametowidget(cnv.itemcget(item, "window"))
                                inner.update_idletasks()
                                h = inner.winfo_reqheight()
                                w = cnv.winfo_width() or inner.winfo_reqwidth()
                                cnv.configure(scrollregion=(0, 0, w, h))
                                cnv.itemconfig(item, width=w)
                                break
                        break
            except Exception:
                pass

    def _scrollable_tab(self, title):
        outer = tk.Frame(self.notebook, bg=BG_PANEL)
        self.notebook.add(outer, text=title)
        cnv   = tk.Canvas(outer, bg=BG_PANEL, highlightthickness=0)
        vsb   = ttk.Scrollbar(outer, orient="vertical", command=cnv.yview)
        cnv.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y"); cnv.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(cnv, bg=BG_PANEL)
        cw    = cnv.create_window((0,0), window=inner, anchor="nw")

        def _update_scrollregion(*_):
            inner.update_idletasks()
            content_h = inner.winfo_reqheight()
            canvas_h  = cnv.winfo_height() or 100
            w = cnv.winfo_width() or inner.winfo_reqwidth()
            # Scroll region is always at least the content height, min 1400
            scroll_h = max(content_h, 1400)
            cnv.configure(scrollregion=(0, 0, w, scroll_h))
            # Inner frame width always matches canvas width
            # Inner frame height: natural size (top-aligned) — never stretched
            cnv.itemconfig(cw, width=w)

        inner.bind("<Configure>", _update_scrollregion)
        cnv.bind("<Configure>",   _update_scrollregion)

        def _mw(e):  cnv.yview_scroll(int(-1*(e.delta/120)), "units"); return "break"
        def _mw4(e): cnv.yview_scroll(-1, "units"); return "break"
        def _mw5(e): cnv.yview_scroll( 1, "units"); return "break"
        for w in (cnv, inner):
            w.bind("<MouseWheel>", _mw)
            w.bind("<Button-4>",   _mw4)
            w.bind("<Button-5>",   _mw5)
        return inner

    # Helper: register a slider with a tab index
    def _reg(self, tab_idx: int, slider: LabeledSlider) -> LabeledSlider:
        """Register slider in per-tab list and wire it to live processing."""
        self._tab_sliders[tab_idx].append(slider)
        slider.callback = lambda v: self._schedule_live(tab_idx)
        return slider

    def _schedule_live(self, tab_idx: int):
        self._pending_tab = tab_idx
        if self._live_after is not None:
            self.root.after_cancel(self._live_after)
        self._live_after = self.root.after(self.LIVE_DELAY, self._fire_live)

    def _fire_live(self):
        self._live_after = None
        if self.original_arr is None: return
        if self.processing:
            self._live_pending = True
            return
        self._live_pending = False

        # Capture ALL slider values on the main thread before the worker runs.
        # The full pipeline always runs in order: Wavelet → FFT → Color → Dering
        # so each tab operates on the previous tab's output.

        # ── Tab 0: Wavelet ──
        wv_params = self._get_wavelet_params()
        wv_fm     = self.wv_filter.get().split()[0]   # "gaussian", "zgaussian", etc.
        wv_cm     = self.wv_color_model.get().split()[0]  # "oklab", "hsl", "hsv"
        wv_conv   = self.wv_convolve.get().split()[0]      # "L", "RGB", "LRGB"
        wv_sf     = [float(self.wsf1.get()), float(self.wsf2.get()), float(self.wsf3.get())]
        wv_ps     = float(self.wv_pre_smooth.get())        # Pre-Smooth sigma
        wv_cd     = float(self._cd_radius.get()) if self._cd_enabled.get() and wv_conv in ("RGB","LRGB") else 0.0
        # Power function: 1.0 = off (linear)
        wv_pf     = float(self.wv_powerfn_exp.get()) if self.wv_powerfn_enabled.get() else 1.0
        wv_zgf    = float(self.wv_zgauss_factor.get()) if hasattr(self, "wv_zgauss_factor") else 1.0
        wv_br     = float(self.wv_bilateral_radius.get()) if hasattr(self, "wv_bilateral_radius") else 2.0
        wv_bls    = [self.wbl1.get(), self.wbl2.get(), self.wbl3.get()] \
                    if wv_fm == "bilateral" and hasattr(self, "wbl1") else None
        src0      = self.original_arr.copy()
        # RGB channel registration — runs first, so the whole pipeline (sharpen,
        # de-rind reference, moon recovery) works on aligned channels.
        if getattr(self, "_align_rgb", None) is not None and self._align_rgb.get():
            src0 = proc.align_rgb_channels(src0)[0]
        wv_lvls   = wv_params

        # ── Tab 1: FFT ──
        # Independent stages — POST-All and any PRE-Layer may run together.
        fft_pre, fft_post = self._fft_collect()

        # ── Tab 2: Color ──
        r_g  = float(self.rgb_r.get())/100;   g_g  = float(self.rgb_g.get())/100
        b_g  = float(self.rgb_b.get())/100;   r_gm = float(self.gamma_r.get())/100
        g_gm = float(self.gamma_g.get())/100; b_gm = float(self.gamma_b.get())/100
        r_bp = float(self.black_r.get())/100  # black point 0-1
        g_bp = float(self.black_g.get())/100
        b_bp = float(self.black_b.get())/100
        sat  = float(self.saturation.get())/100
        vib  = float(self.vibrance.get())/100
        hue  = float(self.hue_rot.get())
        bri  = float(self.brightness.get())/255
        con  = float(self.contrast.get())/100

        # ── Tab 3: Tools ──
        _dc_on   = bool(self._dc_enabled.get())
        _dc_str  = float(self._dc_strength.get()) if _dc_on else 0.0
        _dc_con  = bool(self._dc_use_contrast.get())
        _dc_cstr = float(self._dc_contrast_str.get())
        _dh_on   = bool(self._dh_enabled.get())
        _dh_bs   = max(1, int(float(self._dh_blocksize.get()))) if _dh_on else 5
        _dh_amt  = float(self._dh_amount.get()) if _dh_on else 0.0
        _lc = self._get_lc_params()
        tl_br,tl_wr,tl_gr, tl_bg,tl_wg,tl_gg, tl_bb,tl_wb,tl_gb,         tl_lr,tl_lg,tl_lb, tl_def = _lc
        tl_clahe_on   = bool(self._clahe_enabled.get())
        tl_clahe_clip = float(self._clahe_clip.get())
        tl_clahe_tile = max(4, int(float(self._clahe_tile.get())))
        tl_clahe_str  = float(self._clahe_strength.get())
        tl_clahe_ch   = self._clahe_channel.get()

        # ── Tab 5: Moon Recovery ──
        mr_on         = bool(self.mr_enabled.get())
        mr_circles    = list(self._mr_circles) if hasattr(self, "_mr_circles") else []
        mr_boost      = float(self.mr_boost.get()) / 10.0
        mr_feather    = float(self.mr_feather.get())
        mr_saturation = float(self.mr_saturation.get()) / 10.0 if hasattr(self, "mr_saturation") else 1.0
        mr_darken     = float(self.mr_darken_edge.get()) / 100.0 if hasattr(self, "mr_darken_edge") else 0.0

        # ── Tab 4: De-rind ──
        dr_enabled = bool(self.dr_enabled.get())
        dr_p = dict(
            edge          = float(self.dr_edge.get()),
            smooth        = float(self.dr_smooth.get()),
            inset         = float(self.dr_inset.get()),
            gap_width     = float(self.dr_gap_width.get()),
            gap_angle     = float(self.dr_gap_angle.get()),
            saturn_mode   = bool(self.dr_saturn.get()),
            dark_edge     = bool(self.dr_dark_edge.get()),
            pre_blur      = float(self.dr_pre_blur.get()),
            show_ring_map = bool(self.dr_show_map.get()),
            ref_lum       = proc.luminance(src0.astype(np.float32)),
            ref_arr       = src0,
        )

        def _full_pipeline(
            src=src0,
            lvls=wv_lvls, fm=wv_fm, cm=wv_cm, conv=wv_conv, sf=wv_sf, ps=wv_ps, pf=wv_pf,
            zgf=wv_zgf, br=wv_br, bls=wv_bls, cd=wv_cd,
            fpre=fft_pre, fpost=fft_post,
            rg=r_g, gg=g_g, bg_=b_g,
            rgm=r_gm, ggm=g_gm, bgm=b_gm,
            rbp=r_bp, gbp=g_bp, bbp=b_bp,
            sat_=sat, vib_=vib, hue_=hue, bri_=bri, con_=con,
            dp=dr_p, dr_on=dr_enabled,
            tl_def=tl_def,
            tl_br=tl_br, tl_wr=tl_wr, tl_gr=tl_gr,
            tl_bg=tl_bg, tl_wg=tl_wg, tl_gg=tl_gg,
            tl_bb=tl_bb, tl_wb=tl_wb, tl_gb=tl_gb,
            tl_lr=tl_lr, tl_lg=tl_lg, tl_lb=tl_lb,
            dc_str=_dc_str, dc_con=_dc_con, dc_cstr=_dc_cstr,
            dh_on=_dh_on, dh_bs=_dh_bs, dh_amt=_dh_amt,
            tl_clahe_on=tl_clahe_on, tl_clahe_clip=tl_clahe_clip,
            tl_clahe_tile=tl_clahe_tile, tl_clahe_str=tl_clahe_str,
            tl_clahe_ch=tl_clahe_ch,
            mr_on=mr_on, mr_circles=mr_circles,
            mr_boost=mr_boost, mr_feather=mr_feather, mr_saturation=mr_saturation,
            mr_darken=mr_darken,
        ):
            img = proc.wavelet_sharpen(src, lvls, fm, color_model=cm, convolve=conv,
                                       pre_fft=fpre, sharp_filter=sf, pre_smooth=ps,
                                       power_fn=pf, zgauss_factor=zgf,
                                       bilateral_radius=br, bilateral_layers=bls,
                                       color_denoise=cd)
            # POST FFT: applied after all wavelet layers (PRE stages already
            # ran inside wavelet_sharpen; both may be active at once)
            if fpost is not None:
                img = proc.fft_denoise(img, fft_start=fpost[0],
                                       fft_width=fpost[1], fft_curve=fpost[2])
            img = proc.apply_color_adjustments(
                img, r_gain=rg, g_gain=gg, b_gain=bg_,
                r_black=rbp, g_black=gbp, b_black=bbp,
                r_gamma=rgm, g_gamma=ggm, b_gamma=bgm,
                saturation=sat_, vibrance=vib_, hue_rotation=hue_,
                brightness=bri_, contrast=con_)
            if dc_str > 0.0:
                img = proc.apply_deconvolution(img, strength=dc_str,
                    use_contrast=dc_con, contrast_strength=dc_cstr)
            if dh_on:
                img = proc.apply_dehaze(img, block_size=dh_bs, amount=dh_amt)
            if not tl_def:
                img = proc.apply_levels_curves(
                    img,
                    black_r=tl_br, white_r=tl_wr, gamma_r=tl_gr,
                    black_g=tl_bg, white_g=tl_wg, gamma_g=tl_gg,
                    black_b=tl_bb, white_b=tl_wb, gamma_b=tl_gb,
                    curve_lut_r=tl_lr, curve_lut_g=tl_lg, curve_lut_b=tl_lb)
            if tl_clahe_on:
                img = proc.apply_clahe(img,
                    clip_limit=tl_clahe_clip, tile_grid=tl_clahe_tile,
                    channel_mode=tl_clahe_ch, strength=tl_clahe_str)
            if dr_on:
                img = proc.apply_derind(img, **dp)
            if mr_on and mr_circles:
                img = proc.apply_moon_recovery(
                    img, src,
                    circles=mr_circles,
                    boost=mr_boost,
                    feather=mr_feather,
                    saturation=mr_saturation,
                    darken_edge=mr_darken,
                )
            return img

        self._run_in_thread(_full_pipeline, "PROCESSING")

    # ── Wavelet Tab ─────────────────────────────────────────
    def _build_wavelet_tab(self):
        TAB = 0
        f = self._scrollable_tab("⟴  Wavelet")

        # ── Sharpen levels ──
        # ── Power Function panel (above layers) ────────────────────────────
        section_header(f, "POWER FUNCTION", ACCENT_P).pack(fill="x", padx=6, pady=(2, 0))
        pf_card = card_frame(f)
        pf_row = tk.Frame(pf_card, bg=BG_CARD)
        pf_row.pack(fill="x", pady=(2, 0))
        self.wv_powerfn_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(pf_row, text="Enable  x^y applied to wavelet coefficients",
                       variable=self.wv_powerfn_enabled, bg=BG_CARD, fg=FG_MID,
                       font=F_SM, activebackground=BG_CARD, selectcolor=BG_RAISED,
                       command=self._on_powerfn_toggle).pack(side="left", anchor="w")
        pf_exp_row = tk.Frame(pf_card, bg=BG_CARD)
        pf_exp_row.pack(fill="x", pady=(2, 2))
        tk.Label(pf_exp_row, text="Exponent (y):", bg=BG_CARD, fg=FG_MID,
                 font=F_SM, anchor="w").pack(side="left", padx=(0, 4))
        self.wv_powerfn_exp = tk.DoubleVar(value=1.0)
        pf_spinbox = tk.Spinbox(
            pf_exp_row, from_=0.10, to=4.00, increment=0.05,
            textvariable=self.wv_powerfn_exp, width=7, font=F_MD,
            format="%.2f", relief="flat", bg=BG_RAISED, fg=ACCENT_P,
            buttonbackground=BG_RAISED, activebackground=BTN_ACTIVE,
            disabledforeground=FG_DIM,
            command=self._on_powerfn_change)
        pf_spinbox.pack(side="left")
        pf_spinbox.bind("<Return>",   lambda e: self._on_powerfn_change())
        pf_spinbox.bind("<FocusOut>", lambda e: self._on_powerfn_change())
        self._bind_spinbox(pf_spinbox, self.wv_powerfn_exp, 0.10, 4.00, 0.05, 0, callback=self._on_powerfn_change)
        self._pf_spinbox = pf_spinbox
        tk.Label(pf_exp_row,
                 text="  >1 = boost large coeff  |  <1 = boost small coeff  |  1.0 = linear",
                 bg=BG_CARD, fg=FG_DIM, font=F_XS).pack(side="left", padx=(8, 0))
        self._pf_spinbox.config(state="disabled")

        section_header(f, "SHARPENING LAYERS", ACCENT_C).pack(fill="x", padx=6)

        def _level_card(parent, title, desc, subtitle, accent, sh_def, dn_def, sigma_def, sf_def=0.0):
            card = card_frame(parent, title, accent, subtitle=subtitle)
            sh = self._reg(TAB, LabeledSlider(card, "Sharpen  (0–200)", 0, 200, sh_def, accent, "{:.0f}"))
            sh.pack(fill="x")
            # Denoise hidden — kept wired but not shown
            dn = self._reg(TAB, LabeledSlider(card, "Denoise  (noise gate)", 0, 100, dn_def, FG_MID, "{:.0f}"))
            # Decomp σ hidden — kept wired but not shown
            sg = self._reg(TAB, LabeledSlider(card, "Decomp σ  (filter radius, px)", 0.5, 30.0, sigma_def, accent, "{:.2f}"))
            # Per-layer SharpenFilter — updated by Autosize when enabled.
            # L2/L3 default to 0.100 (WaveSharp 3); L1 defaults off.
            sf = self._reg(TAB, LabeledSlider(card, "SharpenFilter  (0=off)", 0.0, 2.0, sf_def, accent, "{:.3f}"))
            sf.pack(fill="x")
            # Bilateral checkbox — shown only when bilateral filter is selected
            bl_var = tk.BooleanVar(value=True)
            bl_chk = tk.Checkbutton(card,
                text="Use bilateral on this layer  (uncheck = Gaussian for this layer)",
                variable=bl_var, bg=BG_CARD, fg=FG_MID, font=F_SM,
                activebackground=BG_CARD, selectcolor=BG_RAISED,
                command=lambda: self._schedule_live(0))
            return sh, dn, sg, sf, bl_var, bl_chk

        self.ws1, self.wt1, self.wsz1, self.wsf1, self.wbl1, self._bl_chk1 = _level_card(
            f, "LAYER 1", "Fine Detail  (craters, fine grain)", "~0.5 px", ACCENT_C, 0, 0, 0.5, sf_def=0.0)
        self.ws2, self.wt2, self.wsz2, self.wsf2, self.wbl2, self._bl_chk2 = _level_card(
            f, "LAYER 2", "Mid Detail  (filaments, dust lanes)", "~1 px", ACCENT_G, 0, 0, 1.0, sf_def=0.100)
        self.ws3, self.wt3, self.wsz3, self.wsf3, self.wbl3, self._bl_chk3 = _level_card(
            f, "LAYER 3", "Large Structure  (belts, arms)", "~2 px", ACCENT_P, 0, 0, 2.0, sf_def=0.100)

        # ── Autosize Filter + Estimate Filter ───────────────────────────────
        self.wv_autosize = tk.BooleanVar(value=False)
        # ── Row 1: FILTER SIZING (left) + CONVOLVE CHANNEL (right) side by side ──
        wv_mid = tk.Frame(f, bg=BG_PANEL)
        wv_mid.pack(fill="x")
        wv_mid.columnconfigure(0, weight=1, uniform="wvmid")
        wv_mid.columnconfigure(1, weight=1, uniform="wvmid")
        wv_mid_left  = tk.Frame(wv_mid, bg=BG_PANEL); wv_mid_left.grid(row=0, column=0, sticky="nsew")
        wv_mid_right = tk.Frame(wv_mid, bg=BG_PANEL); wv_mid_right.grid(row=0, column=1, sticky="nsew")

        # Move FILTER SIZING into left column
        section_header(wv_mid_left, "FILTER SIZING", ACCENT_C).pack(fill="x", padx=4, pady=(2, 0))
        as_card2 = card_frame(wv_mid_left)
        tk.Label(as_card2,
                 text="Autosize Filter — auto-adjusts\nSharpenFilter as you move Sharpen sliders.\nEstimate Filter — analyzes image\nand sets SharpenFilter.",
                 bg=BG_CARD, fg=FG_DIM, font=F_XS,
                 justify="left").pack(anchor="w", pady=(2, 2))
        tk.Checkbutton(as_card2, text="Autosize Filter",
                       variable=self.wv_autosize, bg=BG_CARD, fg=FG_MID, font=F_SM,
                       activebackground=BG_CARD, selectcolor=BG_RAISED,
                       command=self._on_autosize_toggle).pack(anchor="w")
        tk.Button(as_card2, text="⟴  Estimate Filter",
                  command=self.run_estimate_sharpen,
                  bg=BTN_BG, fg=ACCENT_C, font=F_MD, relief="groove", bd=2,
                  cursor="hand2", padx=8, pady=4,
                  activebackground=BTN_ACTIVE).pack(anchor="w", pady=(4,0))

        # CONVOLVE CHANNEL in right column
        section_header(wv_mid_right, "CONVOLVE CHANNEL", ACCENT_C).pack(fill="x", padx=4, pady=(2, 0))
        cc = card_frame(wv_mid_right, subtitle="Which channels to sharpen")
        # (old FILTER SIZING section below is now replaced by the above)

        # Stretch cards to equal height in the mid row
        self.root.after(100, lambda: [_stretch_cards(wv_mid_left), _stretch_cards(wv_mid_right)])

        # ── Row 2: SHARPEN FILTER full width ──
        wv_right = f   # SHARPEN FILTER packs directly onto tab frame
        # dummy wv_left for compat
        wv_left = wv_mid_right

        tk.Label(cc, text="Convolve:", bg=BG_CARD, fg=FG_MID, font=F_SM).pack(anchor="w")
        self.wv_convolve = ttk.Combobox(cc,
            values=["L  (luminance only, safest)", "RGB  (all channels)", "LRGB  (L first, then RGB)"],
            state="readonly", font=F_SM)
        self.wv_convolve.set("LRGB  (L first, then RGB)")
        self.wv_convolve.pack(fill="x", pady=2)
        self.wv_convolve.bind("<<ComboboxSelected>>", lambda e: self._on_convolve_change())
        # Show Chroma Denoise immediately since LRGB is the default
        self.root.after(50, self._on_convolve_change)

        # Chroma Denoise — shown only when RGB or LRGB
        self._cd_frame = tk.Frame(cc, bg=BG_CARD)
        cd_row = tk.Frame(self._cd_frame, bg=BG_CARD)
        cd_row.pack(fill="x", pady=(4,0))
        self._cd_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(cd_row, text="Chroma Denoise",
                       variable=self._cd_enabled, bg=BG_CARD, fg=FG_MID, font=F_SM,
                       activebackground=BG_CARD, selectcolor=BG_RAISED,
                       command=lambda: self._schedule_live(0)).pack(side="left")
        self._cd_radius = tk.DoubleVar(value=1.0)
        cd_spin = tk.Spinbox(cd_row, from_=0.0, to=20.0, increment=0.5,
                             textvariable=self._cd_radius, width=5, font=F_SM,
                             format="%.1f", relief="flat", bg=BG_RAISED, fg=ACCENT_C,
                             buttonbackground=BG_RAISED, activebackground=BTN_ACTIVE)
        cd_spin.pack(side="left", padx=(6,0))
        self._bind_spinbox(cd_spin, self._cd_radius, 0.0, 20.0, 0.5, 0)
        self._cd_radius.trace_add("write", lambda *_: (
            None if getattr(self, "_loading_project", False) else
            (self._cd_enabled.set(True), self._schedule_live(0))))
        cd_spin.bind("<Return>",   lambda e: self._schedule_live(0))
        cd_spin.bind("<FocusOut>", lambda e: self._schedule_live(0))
        tk.Label(self._cd_frame,
                 text="Smooths color noise by blurring the saturation channel "
                      "after sharpening —\nhue and luminance detail are kept. "
                      "Value = blur radius in px.  0 = off.",
                 bg=BG_CARD, fg=FG_DIM, font=F_XS,
                 justify="left", wraplength=9999).pack(anchor="w", padx=4, pady=(1,1))

        # ── Filter Type + Estimate ──
        section_header(f, "SHARPEN FILTER", ACCENT_C).pack(fill="x", padx=6, pady=(2, 0))
        ff = card_frame(f, subtitle="Filter type affects character — Z-Gaussian = crispest/most aggressive")

        # Color model + Filter type side by side
        cm_ft_row = tk.Frame(ff, bg=BG_CARD); cm_ft_row.pack(fill="x", pady=(0,2))
        cm_ft_row.columnconfigure(0, weight=1, uniform="cmft")
        cm_ft_row.columnconfigure(1, weight=1, uniform="cmft")
        cm_col = tk.Frame(cm_ft_row, bg=BG_CARD); cm_col.grid(row=0, column=0, sticky="ew", padx=(0,4))
        ft_col = tk.Frame(cm_ft_row, bg=BG_CARD); ft_col.grid(row=0, column=1, sticky="ew")
        tk.Label(cm_col, text="Color model:", bg=BG_CARD, fg=FG_MID, font=F_SM).pack(anchor="w")
        self.wv_color_model = ttk.Combobox(cm_col,
            values=["oklab (default)", "hsl", "hsv"],
            state="readonly", font=F_SM)
        self.wv_color_model.set("oklab (default)")
        self.wv_color_model.pack(fill="x", pady=(2,0))
        self.wv_color_model.bind("<<ComboboxSelected>>", lambda e: self._schedule_live(0))
        tk.Label(ft_col, text="Filter type:", bg=BG_CARD, fg=FG_MID, font=F_SM).pack(anchor="w")
        self.wv_filter = ttk.Combobox(ft_col,
            values=["gaussian", "zgaussian (Z-Gaussian)", "bilateral", "b3 (B3-spline)"],
            state="readonly", font=F_SM)
        self.wv_filter.set("gaussian")
        self.wv_filter.pack(fill="x", pady=(2,0))
        self.wv_filter.bind("<<ComboboxSelected>>", lambda e: self._on_filter_change())

        # Pre-Smooth below the two dropdowns — full panel width
        self.wv_pre_smooth = self._reg(0, LabeledSlider(
            ff, "Pre-Smooth  (0=off, use if moire visible)", 0.0, 5.0, 0.0, ACCENT_C, "{:.2f}"))
        self.wv_pre_smooth.pack(fill="x", pady=(4,0))
        tk.Label(ff, text="Eliminates moire/drizzle artifacts before sharpening.",
                 bg=BG_CARD, fg=FG_DIM, font=F_XS).pack(anchor="w", padx=4, pady=(0,2))

        # ── Bilateral extra controls (hidden until bilateral selected) ──
        self._bilateral_frame = tk.Frame(ff, bg=BG_CARD)
        br_row = tk.Frame(self._bilateral_frame, bg=BG_CARD)
        br_row.pack(fill="x", pady=(2, 0))
        tk.Label(br_row, text="Radius  (0–10):", bg=BG_CARD, fg=FG_MID,
                 font=F_SM, width=20, anchor="w").pack(side="left", padx=(0, 4))
        self.wv_bilateral_radius = tk.DoubleVar(value=2.0)
        br_spin = tk.Spinbox(br_row, from_=0.0, to=10.0, increment=0.5,
                             textvariable=self.wv_bilateral_radius,
                             width=6, font=F_MD, format="%.1f", relief="flat",
                             bg=BG_RAISED, fg=ACCENT_C, buttonbackground=BG_RAISED,
                             activebackground=BTN_ACTIVE,
                             command=lambda: self._schedule_live(0))
        br_spin.pack(side="left")
        br_spin.bind("<Return>",   lambda e: self._schedule_live(0))
        br_spin.bind("<FocusOut>", lambda e: self._schedule_live(0))
        self._bind_spinbox(br_spin, self.wv_bilateral_radius, 0.0, 10.0, 0.5, 0)
        tk.Label(self._bilateral_frame,
                 text="Controls spatial smoothing width. Higher = smoother flat areas.",
                 bg=BG_CARD, fg=FG_DIM, font=F_XS).pack(anchor="w", padx=4, pady=(1, 2))
        tk.Label(self._bilateral_frame, text="Per-layer bilateral:",
                 bg=BG_CARD, fg=FG_MID, font=F_SM).pack(anchor="w", padx=2)

        # ── Z-Gaussian extra controls (hidden until zgaussian selected) ──
        self._zgauss_frame = tk.Frame(ff, bg=BG_CARD)
        zf_row = tk.Frame(self._zgauss_frame, bg=BG_CARD)
        zf_row.pack(fill="x", pady=(2, 0))
        tk.Label(zf_row, text="Factor  (0–1):", bg=BG_CARD, fg=FG_MID,
                 font=F_SM, width=20, anchor="w").pack(side="left", padx=(0, 4))
        self.wv_zgauss_factor = tk.DoubleVar(value=1.0)
        zf_spin = tk.Spinbox(zf_row, from_=0.0, to=1.0, increment=0.05,
                             textvariable=self.wv_zgauss_factor,
                             width=6, font=F_MD, format="%.2f", relief="flat",
                             bg=BG_RAISED, fg=ACCENT_C, buttonbackground=BG_RAISED,
                             activebackground=BTN_ACTIVE,
                             command=lambda: self._schedule_live(0))
        zf_spin.pack(side="left")
        zf_spin.bind("<Return>",   lambda e: self._schedule_live(0))
        zf_spin.bind("<FocusOut>", lambda e: self._schedule_live(0))
        self._bind_spinbox(zf_spin, self.wv_zgauss_factor, 0.0, 1.0, 0.05, 0)
        tk.Label(self._zgauss_frame,
                 text="0 = full Z-Gaussian (strongest, use ~30% of Gaussian slider values)  ·  1 = identical to Gaussian.",
                 bg=BG_CARD, fg=FG_DIM, font=F_XS).pack(anchor="w", padx=4, pady=(1, 2))

        # Autosize checkbox (moved to after Layer 3 — var defined there)

        self.estimate_lbl = tk.Label(ff, text="", bg=BG_CARD, fg=FG_DIM, font=F_XS,
                                     wraplength=600, justify="left")
        # label kept for compatibility but not displayed


        # Initial spinbox state handled in panel above


        # ── Wire wavelet sliders → pipeline ──
        def _sharpen_cb(val):
            if self.wv_autosize.get():
                self._run_autosize()   # updates sigma sliders, which fire their own cb
            self._schedule_live(0)
        def _live_cb(val):
            self._schedule_live(0)
        for _s in [self.ws1, self.ws2, self.ws3]:
            _s.callback = _sharpen_cb
        for _s in [self.wt1, self.wt2, self.wt3, self.wsz1, self.wsz2, self.wsz3]:
            _s.callback = _live_cb

        # _wv_sigmas used by Estimate Filter label; sigma sliders are source of truth
        self._wv_sigmas = [
            float(self.wsz1.get()),
            float(self.wsz2.get()),
            float(self.wsz3.get()),
        ]

    def _run_autosize(self):
        """Recompute sigma and SharpenFilter sliders from sharpen values (Autosize Filter)."""
        img = self.original_arr  # may be None before image is loaded
        for sl_sh, sl_sg, sl_sf, layer_idx in [
            (self.ws1, self.wsz1, self.wsf1, 0),
            (self.ws2, self.wsz2, self.wsf2, 1),
            (self.ws3, self.wsz3, self.wsf3, 2),
        ]:
            sigma = proc.estimate_filter_sigma(sl_sh.get(), layer_idx, img)
            sl_sg.set(round(sigma, 2), fire_callback=False)
            # Pass the image so Layer 1's estimate can adapt to its noise level.
            sf_val = proc.estimate_sharpen_filter(sl_sh.get(), sigma, layer_idx, img)
            sl_sf.set(round(sf_val, 3), fire_callback=False)

    def _update_sigma_labels(self):
        """Sync sigma slider displays from _wv_sigmas (used by Estimate Filter)."""
        for sl_sg, sigma in zip([self.wsz1, self.wsz2, self.wsz3], self._wv_sigmas):
            sl_sg.set(round(sigma, 2), fire_callback=False)

    def _on_convolve_change(self):
        """Show Chroma Denoise controls only when RGB or LRGB is selected."""
        conv = self.wv_convolve.get().split()[0]
        if conv in ("RGB", "LRGB"):
            self._cd_frame.pack(fill="x", pady=(2,0))
        else:
            self._cd_frame.pack_forget()
        self._schedule_live(0)

    def _on_autosize_toggle(self):
        if self.wv_autosize.get():
            self._run_autosize()
            self._schedule_live(0)

    def _on_filter_change(self):
        fm = self.wv_filter.get().split()[0]
        # Show/hide bilateral controls
        if fm == "bilateral":
            self._bilateral_frame.pack(fill="x", pady=(2, 0))
            for chk in [self._bl_chk1, self._bl_chk2, self._bl_chk3]:
                chk.pack(anchor="w", padx=4, pady=(2, 4))
        else:
            self._bilateral_frame.pack_forget()
            for chk in [self._bl_chk1, self._bl_chk2, self._bl_chk3]:
                chk.pack_forget()
        # Show/hide Z-Gaussian factor control
        if fm == "zgaussian":
            self._zgauss_frame.pack(fill="x", pady=(2, 0))
        else:
            self._zgauss_frame.pack_forget()
        self._schedule_live(0)

    def _on_powerfn_toggle(self):
        """Enable / disable the power function spinbox and re-run pipeline."""
        enabled = self.wv_powerfn_enabled.get()
        self._pf_spinbox.config(state="normal" if enabled else "disabled")
        self._schedule_live(0)

    def _on_powerfn_change(self):
        """Validate spinbox value and re-run pipeline."""
        try:
            v = float(self.wv_powerfn_exp.get())
            v = max(0.10, min(4.00, v))
            self.wv_powerfn_exp.set(round(v, 2))
        except (ValueError, tk.TclError):
            self.wv_powerfn_exp.set(1.0)
        if self.wv_powerfn_enabled.get():
            self._schedule_live(0)

    # ── FFT Tab ─────────────────────────────────────────────
    def _build_fft_tab(self):
        TAB = 1
        f = self._scrollable_tab("◈  FFT")

        # FFT state — WaveSharp-3 model: four independent denoise stages
        # (POST-All Layers + PRE-Layer 1/2/3). Any combination may be enabled
        # at once, and each stage carries its own Start / End / Curve. Exactly
        # one stage is "active" — the one the graph markers and Curve slider
        # edit. fft_marker_start/end are properties onto the active stage.
        self.fft_stage_ids   = ["POST", "PRE1", "PRE2", "PRE3"]
        self.fft_stage_names = {          # short — stage rows
            "POST": "POST — All Layers",
            "PRE1": "PRE  — Layer 1",
            "PRE2": "PRE  — Layer 2",
            "PRE3": "PRE  — Layer 3",
        }
        self.fft_stage_desc = {           # long — graph caption / Active line
            "POST": "POST — All Layers  (after the wavelet)",
            "PRE1": "PRE — Layer 1  (before Layer 1 sharpen)",
            "PRE2": "PRE — Layer 2  (before Layer 2 sharpen)",
            "PRE3": "PRE — Layer 3  (before Layer 3 sharpen)",
        }
        self.fft_params    = {s: dict(start=60.0, end=90.0, curve=25.0)
                              for s in self.fft_stage_ids}
        self.fft_stage_on  = {s: tk.BooleanVar(value=(s == "POST"))
                              for s in self.fft_stage_ids}
        self.fft_active    = tk.StringVar(value="POST")
        self._fft_dragging = None      # "start" | "end" | None

        tk.Label(f, text="Drag the teal markers to set the filter band.  "
                         "Pink = unfiltered spectrum · Green = filtered · Blue = filter curve.",
                 bg=BG_PANEL, fg=FG_DIM, font=F_HINT,
                 justify="left", wraplength=600).pack(fill="x", padx=10, pady=(4,2))

        # ── Interactive graph ────────────────────────────────
        section_header(f, "FREQUENCY SPECTRUM  &  FILTER CURVE", ACCENT_O).pack(fill="x", padx=6)
        GH = 324   # graph height px (10% reduced)
        self.fft_canvas = tk.Canvas(f, height=GH, bg="#f5f0e8",
                                    highlightthickness=1, highlightbackground=BORDER,
                                    cursor="crosshair")
        self.fft_canvas.pack(fill="both", expand=True, padx=6, pady=3)
        self.fft_canvas.bind("<ButtonPress-1>",   self._fft_mouse_down)
        self.fft_canvas.bind("<B1-Motion>",       self._fft_mouse_drag)
        self.fft_canvas.bind("<ButtonRelease-1>", self._fft_mouse_up)
        self.fft_canvas.bind("<Configure>",       lambda e: self._redraw_fft_graph())

        # Legend
        leg = tk.Frame(f, bg=BG_PANEL); leg.pack(fill="x", padx=8, pady=(0,4))
        for col, txt in [("#e040fb","Unfiltered"), ("#00c853","Filtered"), ("#2196f3","Filter curve")]:
            tk.Frame(leg, bg=col, width=18, height=3).pack(side="left", padx=(0,2), pady=6)
            tk.Label(leg, text=txt, bg=BG_PANEL, fg=FG_MID, font=F_TINY).pack(side="left", padx=(0,12))

        # ── Controls ─────────────────────────────────────────
        cf = card_frame(f, "FILTER SETTINGS", ACCENT_O)

        # Curve shape slider (replaces dropdown)
        # 0 = concave-down / gentle  ·  50 = linear  ·  100 = concave-up / aggressive
        row1 = tk.Frame(cf, bg=BG_CARD); row1.pack(fill="x", pady=(4,2))
        self.fft_curve_slider = self._reg(TAB, LabeledSlider(
            cf, "Curve  (gentle ◀ 0─────50─────100 ▶ aggressive)",
            0.0, 100.0, 25.0, ACCENT_O, "{:.0f}"))
        self.fft_curve_slider.pack(fill="x", pady=(0,4))
        self.fft_curve_slider.callback = self._fft_curve_changed

        # Start/end readouts (updated by dragging)
        row2 = tk.Frame(cf, bg=BG_CARD); row2.pack(fill="x", pady=2)
        tk.Label(row2, text="▶ Start:", bg=BG_CARD, fg="#009688", font=F_HINT,
                 width=8, anchor="w").pack(side="left")
        self.fft_start_lbl = tk.Label(row2, text="60%", bg=BG_CARD, fg="#009688", font=F_SM)
        self.fft_start_lbl.pack(side="left", padx=(0,20))
        tk.Label(row2, text="◀ End:", bg=BG_CARD, fg="#009688", font=F_SM).pack(side="left")
        self.fft_end_lbl = tk.Label(row2, text="90%", bg=BG_CARD, fg="#009688", font=F_SM)
        self.fft_end_lbl.pack(side="left", padx=4)

        # ON/OFF toggle
        row3 = tk.Frame(cf, bg=BG_CARD); row3.pack(fill="x", pady=4)
        self.fft_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(row3, text="Enable FFT Denoise", variable=self.fft_enabled,
                       bg=BG_CARD, fg=ACCENT_O, font=F_SM,
                       activebackground=BG_CARD, selectcolor=BG_RAISED,
                       command=self._fft_toggle
                       ).pack(side="left")
        self.fft_disabled_lbl = tk.Label(row3,
            text="⚠ bars have no effect until enabled",
            bg=BG_CARD, fg=ACCENT_R, font=F_TINY)
        self.fft_disabled_lbl.pack(side="left", padx=8)

        # Auto Denoise button
        row4 = tk.Frame(cf, bg=BG_CARD); row4.pack(fill="x", pady=(2,4))
        tk.Button(row4, text="⟴  Auto Denoise",
                  command=self._fft_auto_denoise,
                  bg=BTN_BG, fg=ACCENT_O, font=F_MD, relief="groove", bd=2,
                  cursor="hand2", padx=8, pady=4,
                  activebackground=BTN_ACTIVE).pack(side="left")
        self.fft_auto_lbl = tk.Label(row4, text="", bg=BG_CARD, fg=FG_DIM, font=F_HINT)
        self.fft_auto_lbl.pack(side="left", padx=8)

        # Stage selector — any combination of stages may run at once, each
        # with its own filter band (WaveSharp 3 behavior). The "edit" radio
        # picks which stage the graph markers and Curve slider act on.
        lf = card_frame(f, "DENOISE STAGES", ACCENT_O)
        tk.Label(lf, text="Check any combination — each stage keeps its own band. "
                         "“edit” picks the one the graph and Curve slider control.",
                 bg=BG_CARD, fg=FG_DIM, font=F_TINY,
                 justify="left", wraplength=560, anchor="w").pack(fill="x", pady=(0,3))
        self._fft_rows = {}
        for val in self.fft_stage_ids:
            row = tk.Frame(lf, bg=BG_CARD); row.pack(fill="x", pady=0)
            row.columnconfigure(1, weight=1)
            cb = tk.Checkbutton(row, text=self.fft_stage_names[val],
                                variable=self.fft_stage_on[val],
                                bg=BG_CARD, fg=FG_MID, font=F_HINT, anchor="w",
                                activebackground=BG_CARD, selectcolor=FFT_CHECK,
                                command=lambda v=val: self._fft_stage_toggled(v))
            cb.grid(row=0, column=0, sticky="w")
            summ = tk.Label(row, text="", bg=BG_CARD, fg=FG_DIM, font=F_TINY, anchor="e")
            summ.grid(row=0, column=1, sticky="e", padx=(8,6))
            rb = tk.Radiobutton(row, text="edit", variable=self.fft_active,
                                value=val, bg=BG_CARD, fg=ACCENT_O, font=F_TINY,
                                activebackground=BG_CARD, selectcolor=FFT_CHECK,
                                command=self._fft_set_active)
            rb.grid(row=0, column=2, sticky="e")
            self._fft_rows[val] = (row, cb, rb, summ)

        self.fft_active_lbl = tk.Label(lf, text="", bg=BG_CARD, fg=ACCENT_O,
                                       font=F_SM, anchor="w")
        self.fft_active_lbl.pack(fill="x", pady=(3,2))

        self._fft_set_active()

    # ── FFT stage helpers (WaveSharp-3 multi-stage model) ────
    def _fft_stage(self, stage=None):
        """Parameter dict {start, end, curve} for a stage (default: active)."""
        return self.fft_params[stage or self.fft_active.get()]

    # The graph markers always edit the active stage, so expose them as
    # properties onto that stage's parameter dict.
    @property
    def fft_marker_start(self):
        return self._fft_stage()["start"]

    @fft_marker_start.setter
    def fft_marker_start(self, v):
        self._fft_stage()["start"] = float(v)

    @property
    def fft_marker_end(self):
        return self._fft_stage()["end"]

    @fft_marker_end.setter
    def fft_marker_end(self, v):
        self._fft_stage()["end"] = float(v)

    def _fft_collect(self):
        """Resolve the stage checkboxes into engine parameters.

        Returns (pre_fft, post):
          pre_fft — 3-element list for proc.wavelet_sharpen, each entry either
                    (start, width, curve) or None, for PRE-Layer 1/2/3
          post    — (start, width, curve) or None, applied after the wavelet
        Stages are independent: POST-All and any PRE-Layer may be on together,
        matching WaveSharp 3. PRE stages run inside the wavelet cascade, POST
        afterwards.
        """
        def _band(stage):
            if not self.fft_enabled.get() or not self.fft_stage_on[stage].get():
                return None
            p = self.fft_params[stage]
            if p["start"] >= 100.0:
                return None
            return (p["start"], max(0.1, p["end"] - p["start"]), p["curve"])
        return [_band("PRE1"), _band("PRE2"), _band("PRE3")], _band("POST")

    def _fft_curve_changed(self, v):
        """Curve slider moved — it belongs to the active stage."""
        self._fft_stage()["curve"] = float(v)
        self._fft_refresh_rows()
        self._redraw_fft_graph()
        self._schedule_live(1)

    def _fft_stage_toggled(self, stage):
        """Stage checkbox clicked — switching it on also makes it the one
        being edited, so the markers immediately show its band."""
        if self.fft_stage_on[stage].get():
            self.fft_active.set(stage)
        self._fft_set_active()
        self._schedule_live(1)

    def _fft_set_active(self):
        """Load the active stage's band into the markers, slider and readouts."""
        p = self._fft_stage()
        self.fft_curve_slider.set(p["curve"], fire_callback=False)
        self.fft_start_lbl.configure(text=f"{p['start']:.0f}%")
        self.fft_end_lbl.configure(text=f"{p['end']:.0f}%")
        self.fft_active_lbl.configure(
            text=f"Active:  {self.fft_stage_desc[self.fft_active.get()]}")
        self._fft_refresh_rows()
        self._redraw_fft_graph()

    def _fft_refresh_rows(self):
        """Repaint the stage rows — highlight the active one, show each band."""
        act = self.fft_active.get()
        for sid, (row, cb, rb, summ) in getattr(self, "_fft_rows", {}).items():
            on   = self.fft_stage_on[sid].get()
            bgc  = BG_RAISED if sid == act else BG_CARD
            row.configure(bg=bgc)
            cb.configure(bg=bgc, activebackground=bgc,
                         fg=ACCENT_O if sid == act else (FG_MID if on else FG_DIM))
            rb.configure(bg=bgc, activebackground=bgc)
            p = self.fft_params[sid]
            summ.configure(bg=bgc,
                           text=f"{p['start']:.0f}–{p['end']:.0f}%  curve {p['curve']:.0f}"
                                if on else "—")

    # ── FFT graph drag helpers ───────────────────────────────
    def _fft_toggle(self):
        """Called when Enable FFT Denoise checkbox is toggled."""
        if self.fft_enabled.get():
            self.fft_disabled_lbl.pack_forget()
        else:
            self.fft_disabled_lbl.pack(side="left", padx=8)
        self._schedule_live(1)
        self._redraw_fft_graph()

    def _fft_x_to_pct(self, x):
        cw = max(self.fft_canvas.winfo_width(), 400)
        return max(0.0, min(100.0, x / cw * 100.0))

    def _fft_mouse_down(self, event):
        """Start dragging whichever marker is closest."""
        pct  = self._fft_x_to_pct(event.x)
        cw   = max(self.fft_canvas.winfo_width(), 400)
        xs   = self.fft_marker_start / 100.0 * cw
        xe   = self.fft_marker_end   / 100.0 * cw
        self._fft_dragging = "start" if abs(event.x - xs) <= abs(event.x - xe) else "end"

    def _fft_mouse_drag(self, event):
        if not self._fft_dragging: return
        pct = self._fft_x_to_pct(event.x)
        if self._fft_dragging == "start":
            self.fft_marker_start = min(pct, self.fft_marker_end - 2.0)
        else:
            self.fft_marker_end = max(pct, self.fft_marker_start + 2.0)
        self.fft_start_lbl.configure(text=f"{self.fft_marker_start:.0f}%")
        self.fft_end_lbl.configure(text=f"{self.fft_marker_end:.0f}%")
        self._redraw_fft_graph()

    def _fft_mouse_up(self, event):
        if self._fft_dragging:
            self._fft_dragging = None
            self._fft_refresh_rows()
            self._schedule_live(1)

    def _draw_fft_graph(self):
        """Draw FFT graph: unfiltered (pink), filtered (green), filter curve (blue)."""
        c = self.fft_canvas
        c.delete("all")
        cw  = max(c.winfo_width(), 400)
        GH  = max(c.winfo_height(), 324)
        PAD_B = 22   # bottom axis strip
        ph    = GH - PAD_B  # usable plot height

        BINS = 80

        # ── Background ───────────────────────────────────────
        c.create_rectangle(0, 0, cw, GH, fill="#f5f0e8", outline="")
        # Vertical dashed grid lines (every 10%)
        for pct in range(0, 101, 10):
            gx = int(pct / 100.0 * cw)
            c.create_line(gx, 0, gx, ph, fill="#d0c8b0", width=1, dash=(3,3))

        # ── Filter curve ─────────────────────────────────────
        fft_start = getattr(self, "fft_marker_start", 60.0)
        fft_end   = getattr(self, "fft_marker_end",   90.0)
        fft_width = max(0.1, fft_end - fft_start)
        try:    curve_shape = float(self.fft_curve_slider.get())
        except: curve_shape = 25.0
        filt = proc.build_fft_filter_curve(fft_start, fft_width, curve_shape, BINS)

        # ── Power spectra (unfiltered & filtered) ────────────
        raw_spec = getattr(self, "_fft_spectrum", None)
        if raw_spec is not None:
            # Interpolate spectrum to BINS resolution
            import numpy as np
            sp = np.interp(np.linspace(0,1,BINS), np.linspace(0,1,len(raw_spec)), raw_spec)
        else:
            import numpy as np
            x = np.linspace(0, 1, BINS)
            sp = np.exp(-5 * x) * 0.9 + 0.05 * np.exp(-30 * (x - 0.15)**2)
            sp = sp / sp.max()

        # Unfiltered — pink/magenta polyline
        pts_raw = []
        for i, v in enumerate(sp):
            px_ = int((i + 0.5) * cw / BINS)
            py_ = int(ph - 4 - v * (ph - 8))
            pts_raw.extend([px_, py_])
        if len(pts_raw) >= 4:
            c.create_line(pts_raw, fill="#e040fb", width=2, smooth=True)

        # Filtered — green polyline (spectrum × filter)
        filt_sp = sp * filt
        pts_filt = []
        for i, v in enumerate(filt_sp):
            px_ = int((i + 0.5) * cw / BINS)
            py_ = int(ph - 4 - v * (ph - 8))
            pts_filt.extend([px_, py_])
        if len(pts_filt) >= 4:
            c.create_line(pts_filt, fill="#00c853", width=2, smooth=True)

        # Other enabled stages — faint dashed curves, so a multi-stage setup
        # (e.g. POST-All + PRE-Layer 1) is visible at a glance.
        act_id = self.fft_active.get() if hasattr(self, "fft_active") else None
        for sid in getattr(self, "fft_stage_ids", []):
            if sid == act_id or not self.fft_stage_on[sid].get():
                continue
            q = self.fft_params[sid]
            ghost = proc.build_fft_filter_curve(
                q["start"], max(0.1, q["end"] - q["start"]), q["curve"], BINS)
            pts_g = []
            for i, v in enumerate(ghost):
                pts_g.extend([int((i + 0.5) * cw / BINS), int(ph - 4 - v * (ph - 8))])
            if len(pts_g) >= 4:
                c.create_line(pts_g, fill="#90caf9", width=1, dash=(4,3), smooth=True)

        # Filter curve — blue polyline (scaled to full height)
        pts_flt = []
        for i, v in enumerate(filt):
            px_ = int((i + 0.5) * cw / BINS)
            py_ = int(ph - 4 - v * (ph - 8))
            pts_flt.extend([px_, py_])
        if len(pts_flt) >= 4:
            c.create_line(pts_flt, fill="#2196f3", width=2, smooth=True)

        # ── Teal marker lines with arrows ────────────────────
        sx = int(fft_start / 100.0 * cw)
        ex = int(fft_end   / 100.0 * cw)
        # Shaded stopband region
        c.create_rectangle(ex, 0, cw, ph, fill="#e0d8c0", outline="", stipple="gray25")
        # Start marker
        c.create_line(sx, 0, sx, ph, fill="#009688", width=2)
        c.create_polygon(sx-7, ph-2, sx+7, ph-2, sx, ph+10, fill="#009688", outline="")
        # End marker
        c.create_line(ex, 0, ex, ph, fill="#009688", width=2)
        c.create_polygon(ex-7, ph-2, ex+7, ph-2, ex, ph+10, fill="#009688", outline="")
        # ── Curve (middle) marker — orange upward triangle ───
        # Positioned at the x midpoint of the rolloff band,
        # y position reflects the filter value at that point (visual feedback).
        mx = (sx + ex) // 2
        curve_v = float(filt[min(int(mx / cw * BINS), BINS-1)])
        my = int(ph - 4 - curve_v * (ph - 8))
        # Vertical dashed line at midpoint
        c.create_line(mx, 0, mx, ph, fill="#e65100", width=1, dash=(3,3))
        # Upward triangle at curve height
        c.create_polygon(mx-8, my+14, mx+8, my+14, mx, my, fill="#e65100", outline="")
        c.create_text(mx, my-8, text=f"{curve_shape:.0f}", fill="#e65100", font=F_TINY, anchor="s")

        # ── Axis ─────────────────────────────────────────────
        c.create_line(0, ph, cw, ph, fill="#888", width=1)
        for pct, lbl in [(0,"DC"), (25,"25%"), (50,"50%"), (75,"75%"), (100,"Nyquist")]:
            ax = int(pct / 100.0 * cw)
            if pct == 0:
                anchor_, x_ = "sw", max(ax, 2)
            elif pct == 100:
                anchor_, x_ = "se", min(ax, cw - 2)
            else:
                anchor_, x_ = "s", ax
            c.create_text(x_, GH-2, anchor=anchor_, text=lbl, fill=FG_DIM, font=F_TINY)

        # ── Active-stage caption ─────────────────────────────
        if act_id:
            c.create_text(6, 4, anchor="nw", fill="#5d4037", font=F_TINY,
                          text=f"Editing:  {self.fft_stage_desc[act_id]}")

        # ── Enabled indicator ────────────────────────────────
        try:
            enabled = self.fft_enabled.get()
        except: enabled = False
        if enabled:
            c.create_rectangle(cw-52, 4, cw-4, 24, fill=ACCENT_O, outline="")
            c.create_text(cw-28, 14, anchor="center", text="ON", fill="white", font=F_BOLD)
            # Master switch is on but this particular stage is not checked —
            # dragging the markers here will not change the image.
            if act_id and not self.fft_stage_on[act_id].get():
                c.create_text(cw//2, 14, anchor="center", fill="#b45309", font=F_BOLD,
                              text="THIS STAGE IS NOT CHECKED")
        else:
            c.create_rectangle(cw-72, 4, cw-4, 24, fill="#cccccc", outline="")
            c.create_text(cw-38, 14, anchor="center", text="OFF", fill="#666666", font=F_BOLD)
            # Dim overlay to show filter is inactive
            c.create_rectangle(0, 0, cw, ph, fill="#f5f0e8", outline="", stipple="gray50")
            c.create_text(cw//2, ph//2, text="FFT DENOISE DISABLED",
                         fill="#999999", font=F_BOLD, anchor="center")

    def _redraw_fft_graph(self):
        self._draw_fft_graph()

    # ── Color Tab ───────────────────────────────────────────
    def _build_color_tab(self):
        TAB = 2
        f = self._scrollable_tab("◉  RGB")

        # ── Align RGB Channels ──
        section_header(f, "ALIGN RGB CHANNELS", ACCENT_C).pack(fill="x", padx=6, pady=(2,0))
        ca = card_frame(f)
        self._align_rgb = tk.BooleanVar(value=False)
        tk.Checkbutton(ca, text="Align RGB channels  (auto)",
                       variable=self._align_rgb, bg=BG_CARD, fg=FG_MID, font=F_SM,
                       activebackground=BG_CARD, selectcolor=BG_RAISED,
                       command=lambda: self._schedule_live(TAB)).pack(anchor="w")
        tk.Label(ca, text="Registers the red and blue channels to green and shifts them back "
                          "into line — removes\ncolored fringing from atmospheric dispersion or a "
                          "misaligned stack. Runs before sharpening.",
                 bg=BG_CARD, fg=FG_DIM, font=F_HINT, justify="left",
                 wraplength=600).pack(anchor="w", pady=(1,0))

        # ── Histogram ──
        section_header(f, "HISTOGRAM", ACCENT_C).pack(fill="x", padx=6, pady=(2,0))
        self.hist_canvas = tk.Canvas(f, height=300, bg=BG_CARD,
                                     highlightthickness=1, highlightbackground=BORDER)
        self.hist_canvas.pack(fill="both", expand=True, padx=6, pady=(0,2))
        self.hist_canvas.bind("<Configure>", lambda e: (
            draw_line_histogram(self.hist_canvas, self.working_arr, e.height)
            if self.working_arr is not None else None))

        # ── R / G / B channels side by side ──
        rgb_grid = tk.Frame(f, bg=BG_PANEL)
        rgb_grid.pack(fill="x")
        rgb_grid.columnconfigure(0, weight=1, uniform="rgbcol")
        rgb_grid.columnconfigure(1, weight=1, uniform="rgbcol")
        rgb_grid.columnconfigure(2, weight=1, uniform="rgbcol")
        rf = tk.Frame(rgb_grid, bg=BG_PANEL); rf.grid(row=0, column=0, sticky="nsew")
        gf = tk.Frame(rgb_grid, bg=BG_PANEL); gf.grid(row=0, column=1, sticky="nsew")
        bf = tk.Frame(rgb_grid, bg=BG_PANEL); bf.grid(row=0, column=2, sticky="nsew")

        cr = card_frame(rf,"RED",ACCENT_R)
        self.rgb_r   = self._reg(TAB, LabeledSlider(cr,"Gain",    0,200,100,ACCENT_R,lambda v: f"{v/2:.0f}"))
        self.rgb_r.pack(fill="x",pady=1)
        self.black_r = self._reg(TAB, LabeledSlider(cr,"Black %", 0,100,0,ACCENT_R,"{:.1f}"))
        self.black_r.pack(fill="x",pady=1)
        self.gamma_r = self._reg(TAB, LabeledSlider(cr,"Gamma",   50,200,100,ACCENT_R,"{:.0f}"))
        self.gamma_r.pack(fill="x",pady=1)

        cg = card_frame(gf,"GREEN",ACCENT_G)
        self.rgb_g   = self._reg(TAB, LabeledSlider(cg,"Gain",    0,200,100,ACCENT_G,lambda v: f"{v/2:.0f}"))
        self.rgb_g.pack(fill="x",pady=1)
        self.black_g = self._reg(TAB, LabeledSlider(cg,"Black %", 0,100,0,ACCENT_G,"{:.1f}"))
        self.black_g.pack(fill="x",pady=1)
        self.gamma_g = self._reg(TAB, LabeledSlider(cg,"Gamma",   50,200,100,ACCENT_G,"{:.0f}"))
        self.gamma_g.pack(fill="x",pady=1)

        cb2 = card_frame(bf,"BLUE","#2563eb")
        self.rgb_b   = self._reg(TAB, LabeledSlider(cb2,"Gain",   0,200,100,"#2563eb",lambda v: f"{v/2:.0f}"))
        self.rgb_b.pack(fill="x",pady=1)
        self.black_b = self._reg(TAB, LabeledSlider(cb2,"Black %",0,100,0,"#2563eb","{:.1f}"))
        self.black_b.pack(fill="x",pady=1)
        self.gamma_b = self._reg(TAB, LabeledSlider(cb2,"Gamma",  50,200,100,"#2563eb","{:.0f}"))
        self.gamma_b.pack(fill="x",pady=1)

        # ── Saturation & Tone ──
        cs2 = card_frame(f,"SATURATION & TONE",ACCENT_C)
        self.saturation = self._reg(TAB, LabeledSlider(cs2,"Saturation", 0,200,100,ACCENT_C,lambda v: f"{v/2:.0f}"))
        self.saturation.pack(fill="x",pady=1)
        self.vibrance   = self._reg(TAB, LabeledSlider(cs2,"Vibrance",   0,200,100,ACCENT_C,lambda v: f"{v/2:.0f}"))
        self.vibrance.pack(fill="x",pady=1)
        self.brightness = self._reg(TAB, LabeledSlider(cs2,"Brightness", -100,100,0,FG_MID))
        self.brightness.pack(fill="x",pady=1)
        # hue_rot and contrast removed — fixed-zero stubs for pipeline compatibility
        class _ZeroVar:
            def __init__(self): self.var = tk.DoubleVar(value=0.0)
            def get(self): return 0.0
        self.hue_rot  = _ZeroVar()
        self.contrast = _ZeroVar()

        # ── Auto balance buttons side by side ──
        ab_row = tk.Frame(f, bg=BG_PANEL); ab_row.pack(fill="x", padx=6, pady=(3,4))
        tk.Button(ab_row, text="◎  AUTO RGB BALANCE", command=self.auto_balance,
                  bg=BTN_BG, fg=ACCENT_G, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=8, pady=3,
                  activebackground=BTN_ACTIVE).pack(side="left", expand=True, fill="x", padx=(0,3))
        tk.Button(ab_row, text="◎  AUTO WHITE BALANCE", command=self.auto_white_balance,
                  bg=BTN_BG, fg=ACCENT_G, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=8, pady=3,
                  activebackground=BTN_ACTIVE).pack(side="left", expand=True, fill="x", padx=(3,0))

        # Reduce slider label/value fonts in this tab
        F_SM_RGB = ("Consolas", 11)
        F_MD_RGB = ("Consolas", 12)
        F_CARD_RGB = ("Consolas", 13, "bold")
        for slider in [self.rgb_r, self.black_r, self.gamma_r,
                       self.rgb_g, self.black_g, self.gamma_g,
                       self.rgb_b, self.black_b, self.gamma_b,
                       self.saturation, self.vibrance, self.brightness]:
            slider._lbl.configure(font=F_SM_RGB)
            slider.val_lbl.configure(font=F_MD_RGB)
        # Reduce card title fonts
        for card in [cr, cg, cb2, cs2]:
            for w in card.master.winfo_children():
                if isinstance(w, tk.Frame):
                    for lbl in w.winfo_children():
                        if isinstance(lbl, tk.Label):
                            try:
                                if "bold" in str(lbl.cget("font")):
                                    lbl.configure(font=F_CARD_RGB)
                            except: pass

    # ── Tools Tab ───────────────────────────────────────────────────────────
    def _build_tools_tab(self):
        TAB = 3
        f = self._scrollable_tab("⚙  Tools")

        # ── Deconvolution ──
        section_header(f, "DECONVOLUTION", ACCENT_C).pack(fill="x", padx=6, pady=(2, 0))
        dc = card_frame(f, subtitle="Fine luminance sharpening via Richardson-Lucy deconvolution")

        self._dc_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(dc, text="Enable Deconvolution",
                       variable=self._dc_enabled, bg=BG_CARD, fg=FG_BRIGHT, font=F_SM,
                       activebackground=BG_CARD, selectcolor=BG_RAISED,
                       command=lambda: self._schedule_live(0)).pack(anchor="w", pady=(1,2))

        dc_grid = tk.Frame(dc, bg=BG_CARD); dc_grid.pack(fill="x", pady=(2,2))

        # Row 0: Strength
        tk.Label(dc_grid, text="Strength:", bg=BG_CARD, fg=FG_MID, font=F_SM,
                 anchor="w").grid(row=0, column=0, sticky="w", padx=(0,8), pady=2)
        self._dc_strength = tk.DoubleVar(value=0.0)
        dc_str_sb = tk.Spinbox(dc_grid, from_=0.0, to=100.0, increment=1.0,
                               textvariable=self._dc_strength, width=7, font=F_MD,
                               format="%.1f", relief="flat", bg=BG_RAISED, fg=ACCENT_C,
                               buttonbackground=BG_RAISED, activebackground=BTN_ACTIVE)
        dc_str_sb.grid(row=0, column=1, sticky="w", pady=2)
        self._bind_spinbox(dc_str_sb, self._dc_strength, 0.0, 100.0, 1.0, 3)
        self._dc_strength.trace_add("write", lambda *_: (
            None if getattr(self, "_loading_project", False) else
            (self._dc_enabled.set(True), self._schedule_live(0))))
        dc_str_sb.bind("<Return>",   lambda e: self._schedule_live(0))
        dc_str_sb.bind("<FocusOut>", lambda e: self._schedule_live(0))

        # Row 1: Use Contrast — checkbox in col 0, spinbox in col 1
        self._dc_use_contrast = tk.BooleanVar(value=False)
        tk.Checkbutton(dc_grid, text="Use Contrast", variable=self._dc_use_contrast,
                       bg=BG_CARD, fg=FG_MID, font=F_SM, anchor="w",
                       activebackground=BG_CARD, selectcolor=BG_RAISED,
                       command=lambda: self._schedule_live(0)).grid(row=1, column=0, sticky="w", padx=(0,8), pady=2)
        self._dc_contrast_str = tk.DoubleVar(value=0.0)
        dc_con_sb = tk.Spinbox(dc_grid, from_=0.0, to=10.0, increment=0.5,
                               textvariable=self._dc_contrast_str, width=7, font=F_MD,
                               format="%.1f", relief="flat", bg=BG_RAISED, fg=ACCENT_C,
                               buttonbackground=BG_RAISED, activebackground=BTN_ACTIVE)
        dc_con_sb.grid(row=1, column=1, sticky="w", pady=2)
        self._bind_spinbox(dc_con_sb, self._dc_contrast_str, 0.0, 10.0, 0.5, 3)
        self._dc_contrast_str.trace_add("write", lambda *_: (
            None if getattr(self, "_loading_project", False) else
            (self._dc_use_contrast.set(True), self._schedule_live(0))))
        dc_con_sb.bind("<Return>",   lambda e: self._schedule_live(0))
        dc_con_sb.bind("<FocusOut>", lambda e: self._schedule_live(0))

        tk.Label(dc, text="Restores fine luminance detail — apply Wavelet sharpening "
                          "first to see its effect.",
                 bg=BG_CARD, fg=FG_DIM, font=F_HINT, justify="left",
                 wraplength=600).pack(anchor="w", pady=(2, 0))

        dc_footer = tk.Frame(dc, bg=BG_CARD); dc_footer.pack(fill="x", pady=(2,0))
        tk.Button(dc_footer, text="\u2298  Reset Deconvolution",
                  command=self._dc_reset,
                  bg=BTN_BG, fg=FG_MID, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=8, pady=3,
                  activebackground=BTN_ACTIVE).pack(side="left")
        tk.Label(dc_footer, text="Sharpens luminance detail via Moffat PSF deconvolution.  "
                                 "Use Contrast weights sharpening toward high-contrast edges.",
                 bg=BG_CARD, fg=FG_DIM, font=F_HINT,
                 justify="left", wraplength=500).pack(side="left", padx=(8,0), anchor="w")

        # ── Dehaze ──
        section_header(f, "DEHAZE", ACCENT_C).pack(fill="x", padx=6, pady=(2, 0))
        dh = card_frame(f, subtitle="Removes chromatic fringing and haze around planetary edges")

        # Side-by-side: Enable+spinboxes left | Reset+description right
        dh_body = tk.Frame(dh, bg=BG_CARD); dh_body.pack(fill="x")
        dh_left  = tk.Frame(dh_body, bg=BG_CARD); dh_left.pack(side="left", fill="y", anchor="n")
        dh_right = tk.Frame(dh_body, bg=BG_CARD); dh_right.pack(side="left", anchor="n", padx=(12,0))

        self._dh_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(dh_left, text="Enable Dehaze",
                       variable=self._dh_enabled, bg=BG_CARD, fg=FG_BRIGHT, font=F_SM,
                       activebackground=BG_CARD, selectcolor=BG_RAISED,
                       command=lambda: self._schedule_live(0)).pack(anchor="w", pady=(0,2))

        dh_row1 = tk.Frame(dh_left, bg=BG_CARD); dh_row1.pack(fill="x", pady=(2,2))
        tk.Label(dh_row1, text="Block Size:", bg=BG_CARD, fg=FG_MID, font=F_SM,
                 width=14, anchor="w").pack(side="left")
        self._dh_blocksize = tk.DoubleVar(value=5.0)
        dh_bs_sb = tk.Spinbox(dh_row1, from_=1.0, to=100.0, increment=1.0,
                               textvariable=self._dh_blocksize, width=7, font=F_MD,
                               format="%.0f", relief="flat", bg=BG_RAISED, fg=ACCENT_C,
                               buttonbackground=BG_RAISED, activebackground=BTN_ACTIVE)
        dh_bs_sb.pack(side="left")
        self._bind_spinbox(dh_bs_sb, self._dh_blocksize, 1.0, 100.0, 1.0, 3)
        self._dh_blocksize.trace_add("write", lambda *_: (
            None if getattr(self, "_loading_project", False) else
            (self._dh_enabled.set(True), self._schedule_live(0))))
        dh_bs_sb.bind("<Return>",   lambda e: self._schedule_live(0))
        dh_bs_sb.bind("<FocusOut>", lambda e: self._schedule_live(0))

        dh_row2 = tk.Frame(dh_left, bg=BG_CARD); dh_row2.pack(fill="x", pady=(2,2))
        tk.Label(dh_row2, text="Amount:", bg=BG_CARD, fg=FG_MID, font=F_SM,
                 width=14, anchor="w").pack(side="left")
        self._dh_amount = tk.DoubleVar(value=0.5)
        dh_amt_sb = tk.Spinbox(dh_row2, from_=0.0, to=1.0, increment=0.05,
                               textvariable=self._dh_amount, width=7, font=F_MD,
                               format="%.2f", relief="flat", bg=BG_RAISED, fg=ACCENT_C,
                               buttonbackground=BG_RAISED, activebackground=BTN_ACTIVE)
        dh_amt_sb.pack(side="left")
        self._bind_spinbox(dh_amt_sb, self._dh_amount, 0.0, 1.0, 0.05, 3)
        self._dh_amount.trace_add("write", lambda *_: (
            None if getattr(self, "_loading_project", False) else
            (self._dh_enabled.set(True), self._schedule_live(0))))
        dh_amt_sb.bind("<Return>",   lambda e: self._schedule_live(0))
        dh_amt_sb.bind("<FocusOut>", lambda e: self._schedule_live(0))

        tk.Button(dh_right, text="\u2298  Reset Dehaze",
                  command=self._dh_reset,
                  bg=BTN_BG, fg=FG_MID, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=8, pady=3,
                  activebackground=BTN_ACTIVE).pack(anchor="w", pady=(0,4))
        tk.Label(dh_right,
                 text="Reduces chromatic fringing\naround planetary edges.\n"
                      "Block Size = local window.\nAmount blends result.",
                 bg=BG_CARD, fg=FG_DIM, font=F_XS,
                 justify="left").pack(anchor="w")

        # ── LEVELS & CURVES ─────────────────────────────────────────────────
        section_header(f, "LEVELS & CURVES", ACCENT_C).pack(fill="x", padx=6)
        lc = card_frame(f, subtitle="Per-channel black/white point, gamma, and tone curve")

        # Side-by-side: spinboxes left, curve right
        lc_body = tk.Frame(lc, bg=BG_CARD)
        lc_body.pack(fill="x")
        lc_left  = tk.Frame(lc_body, bg=BG_CARD)
        lc_left.pack(side="left", fill="y", padx=(0,6))
        lc_right = tk.Frame(lc_body, bg=BG_CARD)
        lc_right.pack(side="left", anchor="n")

        # Channel selector (left column)
        chan_row = tk.Frame(lc_left, bg=BG_CARD); chan_row.pack(fill="x", pady=(2,2))
        tk.Label(chan_row, text="Channel:", bg=BG_CARD, fg=FG_MID,
                 font=F_SM).pack(side="left")
        self._lc_channel = tk.StringVar(value="All")
        for ch in ["All", "R", "G", "B"]:
            tk.Radiobutton(chan_row, text=ch, variable=self._lc_channel, value=ch,
                           bg=BG_CARD, fg=FG_BRIGHT, selectcolor=BG_RAISED,
                           activebackground=BG_CARD, font=F_SM,
                           command=self._on_lc_channel_change).pack(side="left", padx=(6,0))

        def _lc_sb(parent, label, var, from_, to, inc, fmt, accent):
            row = tk.Frame(parent, bg=BG_CARD); row.pack(fill="x", pady=1)
            tk.Label(row, text=label, bg=BG_CARD, fg=FG_MID, font=F_SM,
                     width=14, anchor="w").pack(side="left")
            sb = tk.Spinbox(row, from_=from_, to=to, increment=inc,
                            textvariable=var, width=7, font=F_MD, format=fmt,
                            relief="flat", bg=BG_RAISED, fg=accent,
                            buttonbackground=BG_RAISED, activebackground=BTN_ACTIVE)
            sb.pack(side="left")
            def _on_write(*_, _var=var, _lbl=label):
                self._lc_spinbox_to_curve(_var, _lbl)
                self._schedule_live(0)
            var.trace_add("write", _on_write)
            sb.bind("<Return>",   lambda e: self._schedule_live(0))
            sb.bind("<FocusOut>", lambda e: self._schedule_live(0))
            self._bind_spinbox(sb, var, from_, to, inc, 3)

        self._lc_spinbox_container = tk.Frame(lc_left, bg=BG_CARD)
        self._lc_spinbox_container.pack(fill="x")
        self._lc_all_frame = tk.Frame(self._lc_spinbox_container, bg=BG_CARD)
        self._lc_black_all = tk.DoubleVar(value=0.0)
        self._lc_white_all = tk.DoubleVar(value=1.0)
        self._lc_gamma_all = tk.DoubleVar(value=1.0)
        _lc_sb(self._lc_all_frame, "Black point:", self._lc_black_all, 0.0, 0.99, 0.01, "%.2f", ACCENT_C)
        _lc_sb(self._lc_all_frame, "White point:", self._lc_white_all, 0.01, 1.0,  0.01, "%.2f", ACCENT_C)
        _lc_sb(self._lc_all_frame, "Gamma:",       self._lc_gamma_all, 0.1,  4.0,  0.05, "%.2f", ACCENT_C)

        self._lc_rgb_frame = tk.Frame(self._lc_spinbox_container, bg=BG_CARD)
        self._lc_black = {c: tk.DoubleVar(value=0.0) for c in "RGB"}
        self._lc_white = {c: tk.DoubleVar(value=1.0) for c in "RGB"}
        self._lc_gamma = {c: tk.DoubleVar(value=1.0) for c in "RGB"}
        _acc = {"R": ACCENT_O, "G": ACCENT_G, "B": ACCENT_C}
        for ch in "RGB":
            _lc_sb(self._lc_rgb_frame, f"{ch} Black:", self._lc_black[ch], 0.0,  0.99, 0.01, "%.2f", _acc[ch])
            _lc_sb(self._lc_rgb_frame, f"{ch} White:", self._lc_white[ch], 0.01, 1.0,  0.01, "%.2f", _acc[ch])
            _lc_sb(self._lc_rgb_frame, f"{ch} Gamma:", self._lc_gamma[ch], 0.1,  4.0,  0.05, "%.2f", _acc[ch])
        self._lc_all_frame.pack(fill="x")

        # Reset buttons (left column, below spinboxes)
        btn_row = tk.Frame(lc_left, bg=BG_CARD); btn_row.pack(fill="x", pady=(4,0))
        tk.Button(btn_row, text="\u2298  Reset Curve", command=self._lc_reset_curve,
                  bg=BTN_BG, fg=FG_MID, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=8, pady=3,
                  activebackground=BTN_ACTIVE).pack(side="left")
        tk.Button(btn_row, text="\u2298  Reset All", command=self._lc_reset_all,
                  bg=BTN_BG, fg=FG_MID, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=8, pady=3,
                  activebackground=BTN_ACTIVE).pack(side="left", padx=(4,0))

        # Curve canvas (right column)
        tk.Label(lc_right, text="Tone Curve  (click=add  \u00b7  right-click=remove)",
                 bg=BG_CARD, fg=FG_DIM, font=F_HINT).pack(anchor="w", pady=(0,2))
        CURVE_SZ = 180
        curve_outer = tk.Frame(lc_right, bg=BORDER, padx=1, pady=1,
                               width=CURVE_SZ, height=CURVE_SZ)
        curve_outer.pack(anchor="w")
        curve_outer.pack_propagate(False)
        self._curve_canvas = tk.Canvas(curve_outer, width=CURVE_SZ, height=CURVE_SZ,
                                       bg="#1a1e26", highlightthickness=0)
        self._curve_canvas.pack(fill="both", expand=True)
        def _curve_resize(e):
            sz = min(e.width, e.height)
            if sz > 50:
                self._curve_size = sz
                self._draw_curve_canvas()
        self._curve_canvas.bind("<Configure>", _curve_resize)
        self._curve_size = CURVE_SZ
        self._curve_pts  = {c: [[0.0,0.0],[1.0,1.0]] for c in ["All","R","G","B"]}
        self._lc_updating_curve = False
        self._curve_dragging = None
        self._curve_canvas.bind("<Button-1>",       self._curve_click)
        self._curve_canvas.bind("<B1-Motion>",       self._curve_drag)
        self._curve_canvas.bind("<ButtonRelease-1>", self._curve_release)
        self._curve_canvas.bind("<Button-3>",        self._curve_remove)
        self._draw_curve_canvas()

        # ── CLAHE ────────────────────────────────────────────────────────────
        section_header(f, "CLAHE", ACCENT_P).pack(fill="x", padx=6, pady=(6,0))
        cl = card_frame(f, subtitle="Contrast Limited Adaptive Histogram Equalisation")

        self._clahe_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(cl, text="Enable CLAHE",
                       variable=self._clahe_enabled, bg=BG_CARD, fg=FG_BRIGHT,
                       font=F_SM, activebackground=BG_CARD, selectcolor=BG_RAISED,
                       command=lambda: self._schedule_live(0)).pack(anchor="w", pady=(1,2))

        def _cl_sb(label, var, from_, to, inc, fmt):
            row = tk.Frame(cl, bg=BG_CARD); row.pack(fill="x", pady=1)
            tk.Label(row, text=label, bg=BG_CARD, fg=FG_MID, font=F_SM,
                     width=16, anchor="w").pack(side="left")
            sb = tk.Spinbox(row, from_=from_, to=to, increment=inc,
                            textvariable=var, width=7, font=F_MD, format=fmt,
                            relief="flat", bg=BG_RAISED, fg=ACCENT_P,
                            buttonbackground=BG_RAISED, activebackground=BTN_ACTIVE)
            sb.pack(side="left")
            def _on_write(*_):
                if getattr(self, "_loading_project", False): return
                self._clahe_enabled.set(True)   # auto-enable when tweaking values
                self._schedule_live(0)
            var.trace_add("write", _on_write)
            sb.bind("<Return>",   lambda e: (self._clahe_enabled.set(True), self._schedule_live(0)))
            sb.bind("<FocusOut>", lambda e: self._schedule_live(0))
            self._bind_spinbox(sb, var, from_, to, inc, 3)

        # Side-by-side: spinboxes+Apply to (left) | description+Reset (right)
        cl_body = tk.Frame(cl, bg=BG_CARD); cl_body.pack(fill="x")
        cl_left  = tk.Frame(cl_body, bg=BG_CARD); cl_left.pack(side="left", fill="y", anchor="n")
        cl_right = tk.Frame(cl_body, bg=BG_CARD); cl_right.pack(side="left", anchor="n", padx=(12,0))

        self._clahe_clip     = tk.DoubleVar(value=0.5)
        self._clahe_tile     = tk.DoubleVar(value=4.0)
        self._clahe_strength = tk.DoubleVar(value=0.05)
        # temporarily redirect _cl_sb parent to cl_left
        _orig_cl = cl; cl = cl_left  # redirect spinbox parent
        _cl_sb("Clip Limit:",  self._clahe_clip,     0.01, 10.0, 0.01, "%.2f")
        _cl_sb("Tile Grid:",   self._clahe_tile,     4,    64,   4,    "%.0f")
        _cl_sb("Strength:",    self._clahe_strength, 0.0,  1.0,  0.01, "%.2f")
        cl = _orig_cl  # restore

        cr2 = tk.Frame(cl_left, bg=BG_CARD); cr2.pack(fill="x", pady=(4,0))
        tk.Label(cr2, text="Apply to:", bg=BG_CARD, fg=FG_MID, font=F_SM).pack(side="left")
        self._clahe_channel = tk.StringVar(value="luminance")
        for val, lbl in [("luminance","Luminance only"),("rgb","RGB channels")]:
            tk.Radiobutton(cr2, text=lbl, variable=self._clahe_channel, value=val,
                           bg=BG_CARD, fg=FG_BRIGHT, selectcolor=BG_RAISED,
                           activebackground=BG_CARD, font=F_SM,
                           command=lambda: self._schedule_live(0)).pack(side="left", padx=(6,0))

        tk.Label(cl_right,
                 text="Clip Limit: higher = stronger contrast (default 0.5).\n"
                      "Tile Grid: smaller = more local contrast (default 4).\n"
                      "Luminance only is safer \u2014 RGB mode can shift colors.",
                 bg=BG_CARD, fg=FG_DIM, font=F_XS,
                 justify="left", wraplength=300).pack(anchor="w", pady=(0,4))
        tk.Button(cl_right, text="\u2298  Reset CLAHE",
                  command=self._clahe_reset,
                  bg=BTN_BG, fg=FG_MID, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=8, pady=3,
                  activebackground=BTN_ACTIVE).pack(anchor="w")

    # ── Tools Tab helpers ────────────────────────────────────────────────────

    def _on_lc_channel_change(self):
        ch = self._lc_channel.get()
        if ch == "All":
            self._lc_rgb_frame.pack_forget()
            self._lc_all_frame.pack(fill="x")
        else:
            self._lc_all_frame.pack_forget()
            self._lc_rgb_frame.pack(fill="x")
        self._draw_curve_canvas()
        self._schedule_live(0)

    def _curve_pt_to_canvas(self, px, py):
        sz = self._curve_size
        return int(px * (sz-1)), int((1.0-py) * (sz-1))

    def _canvas_to_curve_pt(self, cx, cy):
        sz = self._curve_size
        return cx / (sz-1), 1.0 - cy / (sz-1)

    def _lc_spinbox_to_curve(self, var, label):
        """When a black/white spinbox changes, move the corresponding curve endpoint."""
        if not hasattr(self, "_curve_pts") or self._lc_updating_curve:
            return
        try:
            val = float(var.get())
        except (ValueError, tk.TclError):
            return
        ch = self._lc_channel.get()

        # Determine which channel's curve and whether it's black or white
        is_black = "Black" in label
        is_white = "White" in label
        if not (is_black or is_white):
            return  # Gamma has no curve endpoint

        # Map label → curve channel(s) to update
        if ch == "All":
            channels = ["All"]
        else:
            # In RGB mode, label is "R Black:", "G White:" etc.
            lbl_ch = label[0] if label[0] in "RGB" else None
            channels = [lbl_ch] if lbl_ch else []

        self._lc_updating_curve = True
        try:
            for c in channels:
                pts = self._curve_pts[c]
                if is_black:
                    # Black point = x of first control point (clamped below white)
                    white_x = pts[-1][0]
                    pts[0][0] = max(0.0, min(white_x - 0.01, val))
                else:
                    # White point = x of last control point (clamped above black)
                    black_x = pts[0][0]
                    pts[-1][0] = max(black_x + 0.01, min(1.0, val))
            self._draw_curve_canvas()
        finally:
            self._lc_updating_curve = False

    def _lc_curve_to_spinbox(self, ch, pts):
        """When first/last curve point is dragged, update the black/white spinboxes."""
        if self._lc_updating_curve:
            return
        black_x = round(pts[0][0], 2)
        white_x = round(pts[-1][0], 2)
        self._lc_updating_curve = True
        try:
            if ch == "All":
                self._lc_black_all.set(black_x)
                self._lc_white_all.set(white_x)
            elif ch in "RGB":
                self._lc_black[ch].set(black_x)
                self._lc_white[ch].set(white_x)
        finally:
            self._lc_updating_curve = False

    def _draw_curve_canvas(self):
        import numpy as _np
        cnv = self._curve_canvas
        sz  = self._curve_size
        ch  = self._lc_channel.get()
        pts = self._curve_pts[ch]
        cnv.delete("all")
        for i in range(1, 4):
            v = int(i * sz / 4)
            cnv.create_line(v, 0, v, sz, fill="#2a2e38", width=1)
            cnv.create_line(0, v, sz, v, fill="#2a2e38", width=1)
        cnv.create_line(0, sz-1, sz-1, 0, fill="#2a2e38", width=1, dash=(4,4))
        arr = self.working_arr if self.working_arr is not None else self.original_arr
        if arr is not None:
            cidx = {"All":None,"R":0,"G":1,"B":2}.get(ch)
            data = (0.299*arr[...,0]+0.587*arr[...,1]+0.114*arr[...,2]).flatten() \
                   if cidx is None else arr[...,cidx].flatten()
            hist, _ = _np.histogram(data, bins=64, range=(0,1))
            hmax = hist.max() if hist.max() > 0 else 1
            bw = sz / 64
            hcol = {"All":"#6a8aaa","R":"#aa5555","G":"#55aa55","B":"#5577cc"}.get(ch,"#6a8aaa")
            for i, h in enumerate(hist):
                bh = int(h / hmax * (sz * 0.7))
                cnv.create_rectangle(int(i*bw), sz, int((i+1)*bw), sz-bh,
                                     fill=hcol, outline="")
        if len(pts) >= 2:
            lut = proc._build_curve_lut(pts)
            coords = []
            for i in range(sz):
                t = i / (sz-1)
                idx = min(int(t*255), 255)
                _, cy = self._curve_pt_to_canvas(t, float(lut[idx]))
                coords += [i, cy]
            lcol = {"All":"#60a0d0","R":"#e06060","G":"#60c060","B":"#6080e0"}.get(ch,"#60a0d0")
            cnv.create_line(coords, fill=lcol, width=2)
        for i, (px, py) in enumerate(pts):
            cx, cy = self._curve_pt_to_canvas(px, py)
            cnv.create_oval(cx-5,cy-5,cx+5,cy+5, fill=ACCENT_C, outline="white", width=1)

    def _curve_hit_test(self, cx, cy):
        ch = self._lc_channel.get()
        best, best_d = None, 12
        for i, (px, py) in enumerate(self._curve_pts[ch]):
            ox, oy = self._curve_pt_to_canvas(px, py)
            d = ((cx-ox)**2+(cy-oy)**2)**0.5
            if d < best_d:
                best, best_d = i, d
        return best

    def _curve_click(self, event):
        ch  = self._lc_channel.get()
        pts = self._curve_pts[ch]
        hit = self._curve_hit_test(event.x, event.y)
        if hit is not None:
            self._curve_dragging = hit
        else:
            nx, ny = self._canvas_to_curve_pt(event.x, event.y)
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            pts.append([nx, ny])
            pts.sort(key=lambda p: p[0])
            self._curve_dragging = next(i for i,(x,y) in enumerate(pts) if x==nx and y==ny)
            self._draw_curve_canvas()
            self._schedule_live(0)

    def _curve_drag(self, event):
        if self._curve_dragging is None: return
        ch  = self._lc_channel.get()
        pts = self._curve_pts[ch]
        i   = self._curve_dragging
        nx, ny = self._canvas_to_curve_pt(event.x, event.y)
        xmin = pts[i-1][0]+0.01 if i > 0 else 0.0
        xmax = pts[i+1][0]-0.01 if i < len(pts)-1 else 1.0
        pts[i][0] = max(xmin, min(xmax, nx))
        # Lock first point to y=0 (black point) and last to y=1 (white point)
        # so dragging along the bottom/top axis stays a straight line
        if i == 0:
            pts[i][1] = 0.0
        elif i == len(pts) - 1:
            pts[i][1] = 1.0
        else:
            pts[i][1] = max(0.0, min(1.0, ny))
        # Sync first/last point x-position back to black/white spinboxes
        self._lc_curve_to_spinbox(ch, pts)
        self._draw_curve_canvas()
        self._schedule_live(0)

    def _curve_release(self, event):
        self._curve_dragging = None

    def _curve_remove(self, event):
        ch  = self._lc_channel.get()
        pts = self._curve_pts[ch]
        hit = self._curve_hit_test(event.x, event.y)
        if hit is not None and 0 < hit < len(pts)-1:
            pts.pop(hit)
            self._draw_curve_canvas()
            self._schedule_live(0)

    def _dc_reset(self):
        self._dc_enabled.set(False)
        self._dc_strength.set(0.0)
        self._dc_use_contrast.set(False)
        self._dc_contrast_str.set(0.0)
        self._schedule_live(0)

    def _dh_reset(self):
        self._dh_enabled.set(False)
        self._dh_blocksize.set(5.0)
        self._dh_amount.set(0.5)
        self._schedule_live(0)

    def _clahe_reset(self):
        """Reset all CLAHE controls to defaults."""
        self._clahe_enabled.set(False)
        self._clahe_clip.set(0.5)
        self._clahe_tile.set(4.0)
        self._clahe_strength.set(0.05)
        self._clahe_channel.set("luminance")
        self._schedule_live(0)

    def _lc_reset_curve(self):
        ch = self._lc_channel.get()
        self._curve_pts[ch] = [[0.0,0.0],[1.0,1.0]]
        # Reset black/white spinboxes for the current channel to defaults
        if ch == "All":
            self._lc_black_all.set(0.0); self._lc_white_all.set(1.0)
        elif ch in "RGB":
            self._lc_black[ch].set(0.0); self._lc_white[ch].set(1.0)
        self._draw_curve_canvas()
        self._schedule_live(0)

    def _lc_reset_all(self):
        for ch in ["All","R","G","B"]:
            self._curve_pts[ch] = [[0.0,0.0],[1.0,1.0]]
        self._lc_black_all.set(0.0); self._lc_white_all.set(1.0); self._lc_gamma_all.set(1.0)
        for ch in "RGB":
            self._lc_black[ch].set(0.0); self._lc_white[ch].set(1.0); self._lc_gamma[ch].set(1.0)
        self._draw_curve_canvas()
        self._schedule_live(0)

    def _get_lc_params(self):
        ch = self._lc_channel.get()
        if ch == "All":
            bl=float(self._lc_black_all.get()); wh=float(self._lc_white_all.get()); gm=float(self._lc_gamma_all.get())
            br=bl; wr=wh; gr=gm; bg=bl; wg=wh; gg=gm; bb=bl; wb=wh; gb=gm
            lut=proc._build_curve_lut(self._curve_pts["All"]); lr=lut; lg=lut; lb=lut
        else:
            br=float(self._lc_black["R"].get()); wr=float(self._lc_white["R"].get()); gr=float(self._lc_gamma["R"].get())
            bg=float(self._lc_black["G"].get()); wg=float(self._lc_white["G"].get()); gg=float(self._lc_gamma["G"].get())
            bb=float(self._lc_black["B"].get()); wb=float(self._lc_white["B"].get()); gb=float(self._lc_gamma["B"].get())
            lr=proc._build_curve_lut(self._curve_pts["R"])
            lg=proc._build_curve_lut(self._curve_pts["G"])
            lb=proc._build_curve_lut(self._curve_pts["B"])
        def _curve_is_identity(pts):
            """True only if the curve is the exact identity: two points at (0,0) and (1,1)."""
            return (len(pts) == 2
                    and abs(pts[0][0]) < 0.001 and abs(pts[0][1]) < 0.001
                    and abs(pts[1][0] - 1.0) < 0.001 and abs(pts[1][1] - 1.0) < 0.001)
        is_default=(br==0 and wr==1 and gr==1 and bg==0 and wg==1 and gg==1 and bb==0 and wb==1 and gb==1
                    and all(_curve_is_identity(self._curve_pts[c]) for c in ["All","R","G","B"]))
        return br,wr,gr,bg,wg,gg,bb,wb,gb,lr,lg,lb,is_default


    # ── De-rind Tab ────────────────────────────────────────────
    def _build_dering_tab(self):
        TAB = 4
        f = self._scrollable_tab("◎  De-rind")

        # ── Enable + Auto ────────────────────────────────────────
        en_row = tk.Frame(f, bg=BG_PANEL)
        en_row.pack(fill="x", padx=10, pady=(8, 2))
        self.dr_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(
            en_row, text="ENABLE DE-RIND", variable=self.dr_enabled,
            bg=BG_PANEL, fg=ACCENT_O, selectcolor=BG_RAISED,
            activebackground=BG_PANEL, activeforeground=ACCENT_O,
            font=F_BOLD, anchor="w", indicatoron=True,
            command=lambda: self._schedule_live(TAB),
        ).pack(side="left")
        tk.Button(en_row, text="⟴  Auto De-rind",
                  command=self._auto_dering,
                  bg=BTN_BG, fg=ACCENT_P, font=F_MD, relief="groove", bd=2,
                  cursor="hand2", padx=8, pady=3,
                  activebackground=BTN_ACTIVE).pack(side="left", padx=(12, 0))
        self.dr_auto_lbl = tk.Label(en_row, text="", bg=BG_PANEL, fg=FG_DIM, font=F_HINT)
        self.dr_auto_lbl.pack(side="left", padx=8)

        tk.Label(f, text="Removes the bright or dark rind — the glowing ring heavy sharpening leaves around a "
                         "planet's edge. Click Auto De-rind for a good starting point, then nudge Edge and Feather.",
                 bg=BG_PANEL, fg=FG_DIM, font=F_HINT,
                 justify="left", wraplength=600).pack(fill="x", padx=10, pady=(2, 6))

        # ── Main controls: Edge + Feather (the two that matter) ───
        cm = card_frame(f, "ADJUST", ACCENT_P)
        self.dr_edge   = self._reg(TAB, LabeledSlider(cm, "Edge  (how far out, px)", 1, 60, 20, ACCENT_P, "{:.0f}"))
        self.dr_edge.pack(fill="x", pady=2)
        self.dr_smooth = self._reg(TAB, LabeledSlider(cm, "Feather  (softness, px)", 0, 20, 2,  FG_MID,   "{:.0f}"))
        self.dr_smooth.pack(fill="x", pady=2)

        # Saturn mode — the common ring case, one click
        sat_row = tk.Frame(cm, bg=BG_CARD); sat_row.pack(fill="x", pady=(6, 2))
        self.dr_saturn = tk.BooleanVar(value=False)
        tk.Checkbutton(sat_row, text="Saturn mode  (keep the rings out of the correction)",
                       variable=self.dr_saturn, bg=BG_CARD, fg=FG_MID,
                       selectcolor=BG_RAISED, activebackground=BG_CARD,
                       font=F_SM, command=lambda: self._schedule_live(TAB)
                       ).pack(side="left")

        # ── Show mask overlay ─────────────────────────────────────
        map_row = tk.Frame(f, bg=BG_PANEL); map_row.pack(fill="x", padx=10, pady=(6, 2))
        self.dr_show_map = tk.BooleanVar(value=False)
        tk.Checkbutton(map_row, text="Show De-rind Mask overlay  (see exactly where it acts)",
                       variable=self.dr_show_map, bg=BG_PANEL, fg=FG_MID,
                       selectcolor=BG_RAISED, activebackground=BG_PANEL,
                       font=F_SM, command=lambda: self._schedule_live(TAB)
                       ).pack(side="left")

        # ── Advanced (collapsible) — rarely needed ────────────────
        adv_hdr = tk.Frame(f, bg=BG_PANEL); adv_hdr.pack(fill="x", padx=6, pady=(2, 0))
        self._dr_adv_open = tk.BooleanVar(value=False)
        self._dr_adv_btn  = tk.Button(adv_hdr, text="▶  Advanced",
                  command=self._toggle_dr_advanced,
                  bg=BTN_BG, fg=FG_MID, font=F_SM, relief="raised", bd=2,
                  cursor="hand2", anchor="w", activebackground=BTN_ACTIVE)
        self._dr_adv_btn.pack(side="left")

        self._dr_adv_frame = tk.Frame(f, bg=BG_PANEL)
        ca = card_frame(self._dr_adv_frame, "ADVANCED", ACCENT_C)
        ca.pack(fill="x", padx=6, pady=(4, 2))

        # Inset — shift the band in/out
        self.dr_inset = self._reg(TAB, LabeledSlider(ca, "Inset  (shift band in/out, px)", -30, 30, 0, FG_MID, "{:.0f}"))
        self.dr_inset.pack(fill="x", pady=2)
        tk.Label(ca, text="Move the correction band toward the disc center (−) or out past the limb (+).",
                 bg=BG_CARD, fg=FG_DIM, font=F_HINT, wraplength=600, justify="left").pack(anchor="w", pady=(0, 6))

        # Manual ring gap (Arc Start / End)
        tk.Label(ca, text="Manual ring gap — Arc Start/End mark a wedge to leave out (clockwise from Start to End); "
                          "the correction is kept everywhere else. Both 0 = full circle. Saturn mode does this for you. "
                          "Steps by 1°; hold Shift while scrolling for 0.1° fine adjust.",
                 bg=BG_CARD, fg=FG_DIM, font=F_HINT, wraplength=600, justify="left").pack(anchor="w", pady=(0, 4))

        gw_row = tk.Frame(ca, bg=BG_CARD); gw_row.pack(fill="x", pady=2)
        tk.Label(gw_row, text="Arc Start (°)", bg=BG_CARD, fg=FG_MID, font=F_SM, width=18, anchor="w").pack(side="left")
        self.dr_gap_width = tk.DoubleVar(value=0.0)
        self._dr_gap_width_lbl = tk.Label(gw_row, textvariable=self.dr_gap_width,
                                           bg=BG_CARD, fg=ACCENT_P, font=F_MD, width=6, anchor="e")
        self._dr_gap_width_lbl.pack(side="right")
        gw_spin = tk.Spinbox(gw_row, from_=-180, to=180, increment=1,
                              textvariable=self.dr_gap_width, width=6,
                              font=F_SM, relief="flat", bg=BG_RAISED,
                              command=lambda: self._schedule_live(TAB))
        gw_spin.pack(side="right", padx=4)
        gw_spin.bind("<MouseWheel>",       lambda e: self._spinbox_wheel(e, self.dr_gap_width, 1, -180, 180, TAB))
        gw_spin.bind("<Shift-MouseWheel>", lambda e: self._spinbox_wheel(e, self.dr_gap_width, 1, -180, 180, TAB))
        gw_spin.bind("<Button-4>",   lambda e: self._spinbox_wheel(e, self.dr_gap_width, 1, -180, 180, TAB))
        gw_spin.bind("<Button-5>",   lambda e: self._spinbox_wheel(e, self.dr_gap_width, 1, -180, 180, TAB))
        gw_spin.bind("<FocusOut>",   lambda e: self._schedule_live(TAB))
        gw_spin.bind("<Return>",     lambda e: self._schedule_live(TAB))

        ga_row = tk.Frame(ca, bg=BG_CARD); ga_row.pack(fill="x", pady=2)
        tk.Label(ga_row, text="Arc End (°)", bg=BG_CARD, fg=FG_MID, font=F_SM, width=18, anchor="w").pack(side="left")
        self.dr_gap_angle = tk.DoubleVar(value=0.0)
        tk.Label(ga_row, textvariable=self.dr_gap_angle,
                 bg=BG_CARD, fg=ACCENT_P, font=F_MD, width=6, anchor="e").pack(side="right")
        ga_spin = tk.Spinbox(ga_row, from_=-180, to=180, increment=1,
                              textvariable=self.dr_gap_angle, width=6,
                              font=F_SM, relief="flat", bg=BG_RAISED,
                              command=lambda: self._schedule_live(TAB))
        ga_spin.pack(side="right", padx=4)
        ga_spin.bind("<MouseWheel>",       lambda e: self._spinbox_wheel(e, self.dr_gap_angle, 1, -180, 180, TAB))
        ga_spin.bind("<Shift-MouseWheel>", lambda e: self._spinbox_wheel(e, self.dr_gap_angle, 1, -180, 180, TAB))
        ga_spin.bind("<Button-4>",   lambda e: self._spinbox_wheel(e, self.dr_gap_angle, 1, -180, 180, TAB))
        ga_spin.bind("<Button-5>",   lambda e: self._spinbox_wheel(e, self.dr_gap_angle, 1, -180, 180, TAB))
        ga_spin.bind("<FocusOut>",   lambda e: self._schedule_live(TAB))
        ga_spin.bind("<Return>",     lambda e: self._schedule_live(TAB))

        # Dark edge toggle (on by default)
        self.dr_dark_edge = tk.BooleanVar(value=True)
        tk.Checkbutton(ca, text="Dark Edge  (also suppress diffraction rings outside the limb)",
                       variable=self.dr_dark_edge, bg=BG_CARD, fg=FG_MID,
                       selectcolor=BG_RAISED, activebackground=BG_CARD,
                       font=F_SM, command=lambda: self._schedule_live(TAB)
                       ).pack(anchor="w", pady=(6, 2))
        tk.Label(ca, text="Especially useful on Mars to suppress bright rings well outside the limb.",
                 bg=BG_CARD, fg=FG_DIM, font=F_HINT, wraplength=600, justify="left").pack(anchor="w", pady=(0, 6))

        # Pre-blur
        self.dr_pre_blur = self._reg(TAB, LabeledSlider(ca, "Pre-blur  (0=off)", 0, 5, 0, FG_MID, "{:.1f}"))
        self.dr_pre_blur.pack(fill="x", pady=2)
        tk.Label(ca, text="Blur the image before de-rinding. Useful when ringing is very strong.",
                 bg=BG_CARD, fg=FG_DIM, font=F_HINT, wraplength=600, justify="left").pack(anchor="w", pady=(0, 4))


    def _spinbox_wheel(self, event, var, step, lo, hi, tab):
        """Mouse-wheel handler for Gap spinboxes.
        Hold Shift while scrolling for a 10x finer step."""
        if event.state & 0x0001:            # Shift held → fine adjust
            step = step / 10.0
        delta = -step if (event.num == 5 or getattr(event, "delta", 0) < 0) else step
        new_val = (var.get() + delta - lo) % (hi - lo + step) + lo
        var.set(round(new_val, 2))
        self._schedule_live(tab)

    def _bind_spinbox(self, sb, var, lo, hi, inc, tab, callback=None):
        """Bind Return, FocusOut, and MouseWheel to a spinbox uniformly.
        Uses spinbox increment for wheel step. callback overrides _schedule_live."""
        def _apply(*_):
            try:
                v = float(var.get())
                v = max(lo, min(hi, v))
                var.set(round(v, 4) if inc < 0.1 else (round(v, 2) if inc < 1 else round(v, 1)))
            except (ValueError, tk.TclError):
                var.set(lo)
            if callback:
                callback()
            elif tab >= 0:
                self._schedule_live(tab)
        def _wheel(e):
            try: cur = float(var.get())
            except (ValueError, tk.TclError): cur = lo
            delta = -inc if (e.num == 5 or getattr(e, "delta", 0) < 0) else inc
            new = max(lo, min(hi, cur + delta))
            var.set(round(new, 4) if inc < 0.1 else (round(new, 2) if inc < 1 else round(new, 1)))
            if callback:
                callback()
            elif tab >= 0:
                self._schedule_live(tab)
        sb.bind("<Return>",    _apply)
        sb.bind("<FocusOut>",  _apply)
        sb.bind("<MouseWheel>", _wheel)
        sb.bind("<Button-4>",   _wheel)
        sb.bind("<Button-5>",   _wheel)

    def _toggle_dr_advanced(self):
        if self._dr_adv_open.get():
            self._dr_adv_frame.pack_forget()
            self._dr_adv_open.set(False)
            self._dr_adv_btn.configure(text="▶  Advanced")
        else:
            self._dr_adv_frame.pack(fill="x", padx=0, pady=(0, 4))
            self._dr_adv_open.set(True)
            self._dr_adv_btn.configure(text="▼  Advanced")

    def _auto_dering(self):
        """Estimate best-guess de-rind parameters from the current image."""
        import numpy as np
        src = self.working_arr if self.working_arr is not None else self.original_arr
        if src is None:
            self.dr_auto_lbl.configure(text="Load an image first"); return

        import numpy as np
        from scipy.ndimage import distance_transform_edt
        lum = (0.2126*src[...,0] + 0.7152*src[...,1] + 0.0722*src[...,2]).astype(np.float64)

        # Robust disk detection: use fraction of image peak
        lum_max   = float(np.percentile(lum, 99.5))
        disk_thresh = max(lum_max * 0.15, 0.01)
        disk_mask   = lum > disk_thresh
        dist_sky    = distance_transform_edt(~disk_mask)

        # Estimate edge extent: find where sky brightness drops to near-background
        deep_sky_mask = dist_sky > 40
        if deep_sky_mask.sum() < 100:
            deep_sky_mask = dist_sky > 20
        sky_bg    = float(np.percentile(lum[deep_sky_mask], 85)) if deep_sky_mask.any() else 0.0
        ring_thresh = sky_bg + (lum_max - sky_bg) * 0.015

        edge_px = 10
        for d in range(1, 40):
            shell = lum[(dist_sky >= d) & (dist_sky < d + 1)]
            if len(shell) < 10:
                continue
            if float(np.mean(shell)) <= ring_thresh:
                edge_px = max(d + 2, 4)
                break
        else:
            edge_px = 16

        edge_px    = int(np.clip(edge_px, 4, 35))
        feather_px = max(2, edge_px // 3)

        self.dr_edge.set(edge_px, fire_callback=False)
        self.dr_smooth.set(feather_px, fire_callback=False)
        self.dr_inset.set(0, fire_callback=False)
        self.dr_gap_width.set(0.0)
        self.dr_gap_angle.set(0.0)
        self.dr_dark_edge.set(True)
        self.dr_pre_blur.set(0.0, fire_callback=False)
        self.dr_enabled.set(True)
        # Saturn mode and the mask overlay are the user's choices — which planet
        # this is, and whether they want to see the mask. Auto estimates the
        # numeric parameters only; clearing these silently undid their setup.
        self.dr_auto_lbl.configure(text=f"edge={edge_px}px  feather={feather_px}px")
        self._schedule_live(3)
    def _build_moon_recovery_ui(self, parent_frame):
        """Build the Moon Recovery panel inside the Orbital tab."""
        f = parent_frame
        section_header(f, "MOON RECOVERY", ACCENT_C).pack(fill="x", padx=6, pady=(2, 0))

        mr_card = card_frame(f, subtitle="Recover faint moons suppressed by sharpening")
        mr_card.pack(fill="x", pady=(4, 0), padx=0)

        # State: list of (cx_img, cy_img, r_img) in full image pixel coords
        self._mr_circles = []          # committed circles
        self._mr_drawing  = False      # currently drawing
        self._mr_draw_start = None     # (canvas_x, canvas_y) of press
        self._mr_draw_item  = None     # canvas oval item id being dragged
        self._mr_mode = False          # True = place-circle mode active
        self._mr_overlays_hidden = False  # True after finalise — circles hidden but active

        # Enable checkbox
        self.mr_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(mr_card, text="Enable Moon Recovery",
                       variable=self.mr_enabled, bg=BG_CARD, fg=FG_MID, font=F_SM,
                       activebackground=BG_CARD, selectcolor=BG_RAISED,
                       command=lambda: self._schedule_live(5)).pack(anchor="w")

        tk.Label(mr_card,
                 text="Draw circle masks over faint moons in the processed preview.\n"
                      "Each circle blends a boosted copy of the original into that region.",
                 bg=BG_CARD, fg=FG_DIM, font=F_HINT,
                 justify="left", wraplength=600).pack(anchor="w", pady=(2, 6))

        # Place / Clear buttons
        btn_row = tk.Frame(mr_card, bg=BG_CARD)
        btn_row.pack(fill="x", pady=(0, 4))
        self._mr_place_btn = tk.Button(
            btn_row, text="✛  Place Moon Circle",
            bg=BTN_BG, fg=ACCENT_C, font=F_MD, relief="groove", bd=2,
            cursor="hand2", padx=8, pady=4,
            activebackground=BTN_ACTIVE,
            command=self._mr_toggle_draw_mode)
        self._mr_place_btn.pack(side="left")
        tk.Button(btn_row, text="✕  Clear All",
                  bg=BG_RAISED, fg=ACCENT_O, font=F_MD, relief="groove", bd=2,
                  cursor="hand2", padx=8, pady=4,
                  activebackground=BTN_ACTIVE,
                  command=self._mr_clear_all).pack(side="left", padx=(6, 0))

        self._mr_circle_lbl = tk.Label(mr_card, text="No circles placed.",
                                        bg=BG_CARD, fg=FG_DIM, font=F_HINT)
        self._mr_circle_lbl.pack(anchor="w", pady=(0, 4))

        # Boost slider
        self.mr_boost = self._reg(5, LabeledSlider(
            mr_card, "Boost  (×)", 10, 500, 25, ACCENT_C, lambda v: f"{v/10:.1f}×"))
        self.mr_boost.pack(fill="x", pady=2)

        # Feather slider
        self.mr_feather = self._reg(5, LabeledSlider(
            mr_card, "Feather  (px)", 0, 30, 5, FG_MID, "{:.0f}"))
        self.mr_feather.pack(fill="x", pady=2)

        # Darken outer edge — suppresses the oblong background outside the circle
        self.mr_darken_edge = self._reg(5, LabeledSlider(
            mr_card, "Darken Outer Edge  (%)", 0, 100, 0, FG_MID, "{:.0f}"))
        self.mr_darken_edge.pack(fill="x", pady=2)
        tk.Label(mr_card,
                 text="Darkens the region just outside the circle to suppress the oblong\n"
                      "background shape. 0 = off.",
                 bg=BG_CARD, fg=FG_DIM, font=F_HINT,
                 justify="left", wraplength=600).pack(anchor="w", pady=(0, 4))

        # Saturation — desaturate inside circles to remove color cast
        self.mr_saturation = self._reg(5, LabeledSlider(
            mr_card, "Moon Saturation", 0, 10, 10, FG_MID, "{:.0f}"))
        self.mr_saturation.pack(fill="x", pady=2)
        tk.Label(mr_card,
                 text="0 = fully desaturated inside circles. Reduces color cast\n"
                      "on brightness-boosted moons.",
                 bg=BG_CARD, fg=FG_DIM, font=F_HINT,
                 justify="left", wraplength=600).pack(anchor="w", pady=(0, 4))

        tk.Label(mr_card,
                 text="Tip: in Place mode, right-click a circle to remove it.",
                 bg=BG_CARD, fg=FG_DIM, font=F_HINT,
                 justify="left", wraplength=600).pack(anchor="w", pady=(2, 4))

        # Moon preview boost
        section_header(mr_card, "PREVIEW BRIGHTNESS", FG_MID).pack(fill="x", pady=(2, 0))
        tk.Label(mr_card,
                 text="Temporarily brightens the PROCESSED preview so faint moons\n"
                      "become visible. Display only — does not affect export.",
                 bg=BG_CARD, fg=FG_DIM, font=F_HINT,
                 justify="left", wraplength=600).pack(anchor="w", pady=(2, 4))
        self.mr_preview_boost = self._reg(5, LabeledSlider(
            mr_card, "Preview Boost  (×)", 10, 200, 10, ACCENT_O,
            lambda v: f"{v/10:.1f}×"))
        self.mr_preview_boost.pack(fill="x", pady=2)
        tk.Button(mr_card, text="↺  Reset Preview",
                  bg=BG_RAISED, fg=FG_MID, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=6, pady=2,
                  activebackground=BTN_ACTIVE,
                  command=self._mr_reset_preview_boost).pack(anchor="w", pady=(2, 4))

        # Finalize button
        section_header(mr_card, "FINALIZE", FG_MID).pack(fill="x", pady=(2, 0))
        tk.Label(mr_card,
                 text="When satisfied with the result, remove the circle overlays\n"
                      "so you can continue processing on other tabs.",
                 bg=BG_CARD, fg=FG_DIM, font=F_HINT,
                 justify="left", wraplength=600).pack(anchor="w", pady=(2, 4))
        tk.Button(mr_card, text="✔  Moon Recovery Complete — Remove Circles",
                  bg=BG_RAISED, fg=ACCENT_C, font=F_MD, relief="groove", bd=2,
                  cursor="hand2", padx=8, pady=4,
                  activebackground=BTN_ACTIVE,
                  command=self._mr_finalise).pack(fill="x", pady=(0, 6))

        # Override preview boost slider callback
        def _pb_redraw(*_):
            pil = self._get_processed_pil()
            if pil is not None:
                self._draw_on_canvas(self.cnv_proc, pil)
                self._mr_update_overlays()
        self.mr_preview_boost.callback = _pb_redraw

    # ── Moon Recovery interaction ──────────────────────────────────────────────

    def _mr_toggle_draw_mode(self):
        """Enter or exit circle-drawing mode on the processed preview canvas."""
        self._mr_mode = not self._mr_mode
        if self._mr_mode:
            # Entering draw mode — show overlays again
            self._mr_overlays_hidden = False
            self._mr_place_btn.configure(text="⏹  Done Placing", fg=ACCENT_O)
            self.cnv_proc.configure(cursor="crosshair")
            # Button-1 for drawing circles
            self.cnv_proc.bind("<ButtonPress-1>",  self._mr_press)
            self.cnv_proc.bind("<B1-Motion>",       self._mr_drag)
            self.cnv_proc.bind("<ButtonRelease-1>", self._mr_release)
            # Use ButtonRelease-3 for removing circles — this is a DISTINCT event
            # from ButtonPress-3 so it does NOT overwrite the pan binding
            self.cnv_proc.bind("<ButtonRelease-3>", self._mr_pan_end_and_remove)
        else:
            self._mr_place_btn.configure(text="✛  Place Moon Circle", fg=ACCENT_C)
            self.cnv_proc.configure(cursor="")
            self.cnv_proc.bind("<ButtonPress-1>",
                lambda e: self._magnify_at(e, self.cnv_proc,
                                           self._get_processed_pil, "PROC"))
            self.cnv_proc.unbind("<B1-Motion>")
            self.cnv_proc.unbind("<ButtonRelease-1>")
            # Restore ButtonRelease-3 to pan_end
            self.cnv_proc.bind("<ButtonRelease-3>",
                lambda e: self._pan_end(e, self.cnv_proc))

    def _mr_finalise(self):
        """Hide circle overlays — the moon recovery effect is kept.
        Circles remain active in the pipeline so Enable toggle works as before/after.
        """
        # Don't clear _mr_circles — keep them so the pipeline can still apply
        # the effect and mr_enabled toggles as a before/after
        self._mr_update_overlays()  # will redraw with existing circles (no change visually
        # but clears any in-progress drawing artifacts)
        # Exit draw mode if active
        if getattr(self, "_mr_mode", False):
            self._mr_toggle_draw_mode()
        # Suppress overlay display so circles are invisible but still active
        self._mr_overlays_hidden = True
        # Clear the overlay tags from canvas without clearing _mr_circles
        self.cnv_proc.delete("mr_circle")
        # Redraw canvas cleanly
        pil = self._get_processed_pil()
        if pil is not None:
            self._draw_on_canvas(self.cnv_proc, pil)

    def _mr_pan_end_and_remove(self, event):
        """ButtonRelease-3 in draw mode: restore cursor then optionally remove circle.
        Called after a right-click — could be end of a pan drag or a stationary click.
        Only remove a circle if the mouse barely moved (not a drag)."""
        # Always restore cursor to crosshair after right-button release
        self._is_panning = False
        self.cnv_proc.configure(cursor="crosshair")
        # Check if this was a stationary click (not a pan) by comparing
        # release position to the press position stored in pan state
        p = self._pan.get(id(self.cnv_proc), [0, 0, 0, 0])
        # p[2]/p[3] hold the last recorded drag position — if release is close
        # to where ButtonPress-3 fired, treat as a click (remove circle)
        import math
        drag_dist = math.hypot(event.x - p[2], event.y - p[3])
        if drag_dist < 8:
            self._mr_right_click_release(event)

    def _mr_canvas_to_img(self, cx, cy):
        """Convert processed-preview canvas coords to full image pixel coords.
        Accounts for preview rotation so circles land correctly in working_arr."""
        if self.working_arr is None:
            return 0, 0
        import math as _math
        # The canvas shows the ROTATED image — use rotated PIL for scale/offset
        pil_rot = self._get_processed_pil()
        if pil_rot is None:
            return 0, 0
        cnv = self.cnv_proc
        cw = cnv.winfo_width() or 1
        ch = cnv.winfo_height() or 1
        scale = (min(cw / pil_rot.width, ch / pil_rot.height)
                 if self._zoom_pct_var.get() == "Fit" else self._prev_zoom)
        nw = max(1, int(pil_rot.width  * scale))
        nh = max(1, int(pil_rot.height * scale))
        p  = self._pan.get(id(cnv), [0, 0, 0, 0])
        ox = (cw - nw) // 2 + p[0]
        oy = (ch - nh) // 2 + p[1]
        # Position in rotated image space
        rx = (cx - ox) / scale
        ry = (cy - oy) / scale
        # Reverse-rotate back to unrotated image space
        try:
            deg = float(self._preview_rot_deg.get())
        except Exception:
            deg = 0.0
        if deg != 0.0:
            W0 = self.working_arr.shape[1]
            H0 = self.working_arr.shape[0]
            # Center of rotated image
            rcx = pil_rot.width  / 2.0
            rcy = pil_rot.height / 2.0
            rad = _math.radians(deg)   # PIL rotates by +deg CCW
            dx = rx - rcx
            dy = ry - rcy
            # Rotate point back by -deg
            ix = dx * _math.cos(rad) + dy * _math.sin(rad) + W0 / 2.0
            iy = -dx * _math.sin(rad) + dy * _math.cos(rad) + H0 / 2.0
        else:
            ix, iy = rx, ry
        return ix, iy

    def _mr_img_to_canvas(self, ix, iy):
        """Convert full image pixel coords to processed-preview canvas coords.
        Accounts for preview rotation so circle overlays appear correctly."""
        if self.working_arr is None:
            return 0, 0
        import math as _math
        pil_rot = self._get_processed_pil()
        if pil_rot is None:
            return 0, 0
        cnv = self.cnv_proc
        cw = cnv.winfo_width() or 1
        ch = cnv.winfo_height() or 1
        scale = (min(cw / pil_rot.width, ch / pil_rot.height)
                 if self._zoom_pct_var.get() == "Fit" else self._prev_zoom)
        nw = max(1, int(pil_rot.width  * scale))
        nh = max(1, int(pil_rot.height * scale))
        p  = self._pan.get(id(cnv), [0, 0, 0, 0])
        ox = (cw - nw) // 2 + p[0]
        oy = (ch - nh) // 2 + p[1]
        # Forward-rotate image coords into rotated display space
        try:
            deg = float(self._preview_rot_deg.get())
        except Exception:
            deg = 0.0
        if deg != 0.0:
            W0 = self.working_arr.shape[1]
            H0 = self.working_arr.shape[0]
            rcx = pil_rot.width  / 2.0
            rcy = pil_rot.height / 2.0
            rad = _math.radians(deg)
            dx = ix - W0 / 2.0
            dy = iy - H0 / 2.0
            rx = dx * _math.cos(rad) - dy * _math.sin(rad) + rcx
            ry = dx * _math.sin(rad) + dy * _math.cos(rad) + rcy
        else:
            rx, ry = ix, iy
        cx = rx * scale + ox
        cy = ry * scale + oy
        return cx, cy

    def _mr_press(self, event):
        self._mr_drawing    = True
        self._mr_draw_start = (event.x, event.y)
        self._mr_suppress_redraw = True   # block _finish_processing from wiping canvas
        self._mr_draw_item  = self.cnv_proc.create_oval(
            event.x, event.y, event.x, event.y,
            outline="#00ffcc", width=2, dash=(4, 3), tags="mr_drawing")

    def _mr_drag(self, event):
        if not self._mr_drawing or self._mr_draw_item is None:
            return
        x0, y0 = self._mr_draw_start
        # Radius = distance from start to current; draw circle centered on press point
        import math
        r = math.hypot(event.x - x0, event.y - y0)
        self.cnv_proc.coords(self._mr_draw_item,
                             x0 - r, y0 - r, x0 + r, y0 + r)

    def _mr_release(self, event):
        if not self._mr_drawing:
            return
        self._mr_drawing = False
        import math
        x0, y0 = self._mr_draw_start
        r_canvas = math.hypot(event.x - x0, event.y - y0)
        if r_canvas < 1:
            # Too small — discard
            if self._mr_draw_item:
                self.cnv_proc.delete(self._mr_draw_item)
            self._mr_draw_item = None
            self._mr_suppress_redraw = False
            return
        # Convert to image coords
        ix, iy = self._mr_canvas_to_img(x0, y0)
        # Radius: convert canvas pixels to image pixels using display scale
        pil_rot = self._get_processed_pil()
        if pil_rot is not None:
            cnv = self.cnv_proc
            cw = cnv.winfo_width() or 1
            ch = cnv.winfo_height() or 1
            disp_scale = (min(cw / pil_rot.width, ch / pil_rot.height)
                          if self._zoom_pct_var.get() == "Fit" else self._prev_zoom)
        else:
            disp_scale = self._prev_zoom
        r_img = r_canvas / disp_scale if disp_scale > 0 else r_canvas
        self._mr_circles.append((ix, iy, r_img))
        self._mr_draw_item = None
        self._mr_suppress_redraw = False   # allow redraws again
        self._mr_update_overlays()
        self._schedule_live(5)

    def _mr_right_click(self, event):
        """Remove the circle nearest to the right-click point (legacy, kept for compat)."""
        self._mr_right_click_release(event)

    def _mr_right_click_release(self, event):
        """Remove the circle nearest to the right-click release point."""
        import math
        # Only act if we haven't panned (drag distance < 5px)
        p = self._pan.get(id(self.cnv_proc), [0,0,0,0])
        best, best_d = None, float("inf")
        for i, (cx, cy, r) in enumerate(self._mr_circles):
            ccx, ccy = self._mr_img_to_canvas(cx, cy)
            d = math.hypot(event.x - ccx, event.y - ccy)
            if d < best_d and d < r * self._prev_zoom + 10:
                best, best_d = i, d
        if best is not None:
            self._mr_circles.pop(best)
            self._mr_update_overlays()
            self._schedule_live(5)

    def _mr_clear_all(self):
        self._mr_circles.clear()
        self._mr_update_overlays()
        self._schedule_live(5)

    def _mr_reset_preview_boost(self):
        """Reset the moon preview boost to 1.0× (display only)."""
        if hasattr(self, "mr_preview_boost"):
            self.mr_preview_boost.set(10, fire_callback=False)  # 10 = 1.0×
        # Redraw without triggering pipeline
        pil = self._get_processed_pil()
        if pil is not None:
            self._draw_on_canvas(self.cnv_proc, pil)

    def _mr_update_overlays(self):
        """Redraw all committed circle overlays on the processed preview."""
        self.cnv_proc.delete("mr_circle")
        # If overlays are hidden (after finalise), don't redraw them
        if getattr(self, "_mr_overlays_hidden", False):
            n = len(self._mr_circles)
            lbl = f"{n} circle{'s' if n != 1 else ''} placed (hidden)." if n else "No circles placed."
            if hasattr(self, "_mr_circle_lbl"):
                self._mr_circle_lbl.configure(text=lbl)
            return
        n = len(self._mr_circles)
        # Compute the actual display scale to convert image-space radius to canvas pixels
        pil_rot = self._get_processed_pil()
        if pil_rot is not None:
            cnv = self.cnv_proc
            cw = cnv.winfo_width() or 1
            ch = cnv.winfo_height() or 1
            disp_scale = (min(cw / pil_rot.width, ch / pil_rot.height)
                          if self._zoom_pct_var.get() == "Fit" else self._prev_zoom)
        else:
            disp_scale = self._prev_zoom
        for cx, cy, r in self._mr_circles:
            ccx, ccy = self._mr_img_to_canvas(cx, cy)
            rc = r * disp_scale
            self.cnv_proc.create_oval(
                ccx - rc, ccy - rc, ccx + rc, ccy + rc,
                outline="#00ffcc", width=2, tags="mr_circle")
        lbl = f"{n} circle{'s' if n != 1 else ''} placed." if n else "No circles placed."
        if hasattr(self, "_mr_circle_lbl"):
            self._mr_circle_lbl.configure(text=lbl)

    # ── Stats Tab ────────────────────────────────────────────
    # ── Batch Tab ────────────────────────────────────────────
    def _build_orbital_tab(self):
        TAB = 5
        f = self._scrollable_tab("⊕  Orbital")
        tk.Label(f,
                 text="Recover faint moons and satellites suppressed by sharpening.",
                 bg=BG_PANEL, fg=FG_DIM, font=F_HINT,
                 justify="left", wraplength=600).pack(fill="x", padx=10, pady=(6, 2))
        self._build_moon_recovery_ui(f)

    def _build_batch_tab(self):
        f = self._scrollable_tab("⬡  Batch")

        # ── INPUT FILES ────────────────────────────────────
        cf = card_frame(f, "INPUT FILES", ACCENT_C)

        # Buttons + count on one row
        btn_row = tk.Frame(cf, bg=BG_CARD); btn_row.pack(fill="x", pady=(0,1))
        tk.Button(btn_row, text="＋ Add",
                  command=self._batch_add_files,
                  bg=BTN_BG, fg=ACCENT_C, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=7, pady=2,
                  activebackground=BTN_ACTIVE).pack(side="left")
        tk.Button(btn_row, text="✕ Remove",
                  command=self._batch_remove_selected,
                  bg=BTN_BG, fg=ACCENT_O, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=7, pady=2,
                  activebackground=BTN_ACTIVE).pack(side="left", padx=(4,0))
        tk.Button(btn_row, text="⊘ Clear",
                  command=self._batch_clear,
                  bg=BTN_BG, fg=FG_MID, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=7, pady=2,
                  activebackground=BTN_ACTIVE).pack(side="left", padx=(4,0))
        self._batch_count_lbl = tk.Label(btn_row, text="0 files",
                                          bg=BG_CARD, fg=FG_DIM, font=F_HINT)
        self._batch_count_lbl.pack(side="left", padx=10)

        # Compact listbox — 5 rows
        lb_frame = tk.Frame(cf, bg=BG_CARD); lb_frame.pack(fill="x")
        lb_scroll = tk.Scrollbar(lb_frame, orient="vertical")
        self._batch_listbox = tk.Listbox(
            lb_frame, height=4 if sys.platform.startswith("linux") else 5, font=F_SM,
            bg=BG_RAISED, fg=FG_MID, selectbackground=SEL_BG,
            relief="flat", bd=0, activestyle="none",
            yscrollcommand=lb_scroll.set, selectmode="extended")
        lb_scroll.config(command=self._batch_listbox.yview)
        lb_scroll.pack(side="right", fill="y")
        self._batch_listbox.pack(side="left", fill="x", expand=True)

        # ── ALIGNMENT + ROTATION (merged card) ─────────────
        car = card_frame(f, "ALIGNMENT & ROTATION", ACCENT_C)

        # Top row: align checkbox
        self._batch_align = tk.BooleanVar(value=False)
        tk.Checkbutton(car,
                       text="Align planet disks to common center  (shifts each image to match first file)",
                       variable=self._batch_align,
                       bg=BG_CARD, fg=FG_MID, selectcolor=BG_RAISED,
                       activebackground=BG_CARD, font=F_HINT).pack(anchor="w")

        tk.Frame(car, bg=BORDER, height=1).pack(fill="x", pady=(2,2))

        # Rotation header row
        rot_hdr = tk.Frame(car, bg=BG_CARD); rot_hdr.pack(fill="x")
        self._batch_rotate_on = tk.BooleanVar(value=False)
        tk.Checkbutton(rot_hdr, text="Rotate all images",
                       variable=self._batch_rotate_on,
                       bg=BG_CARD, fg=FG_MID, selectcolor=BG_RAISED,
                       activebackground=BG_CARD, font=F_HINT,
                       command=self._batch_rotate_toggle).pack(side="left")
        tk.Label(rot_hdr, text="(negative = counter-clockwise)",
                 bg=BG_CARD, fg=FG_DIM, font=F_HINT).pack(side="left", padx=8)

        # Two-column layout: controls left, preview right (preview adds no extra height)
        rot_outer = tk.Frame(car, bg=BG_CARD); rot_outer.pack(fill="x", pady=(2,0))
        rot_controls = tk.Frame(rot_outer, bg=BG_CARD); rot_controls.pack(side="left", anchor="n", fill="x", expand=True)
        rot_prev_col  = tk.Frame(rot_outer, bg=BG_CARD); rot_prev_col.pack(side="right", anchor="n", padx=(8,0))

        # Presets + spinbox row
        rot_row1 = tk.Frame(rot_controls, bg=BG_CARD); rot_row1.pack(fill="x", pady=(0,1))
        self._batch_rot_preset_frame = rot_row1
        for deg in [45, 90, 180, -45, -90]:
            tk.Button(rot_row1, text=f"{deg}°",
                      command=lambda d=deg: self._batch_set_rotation(d),
                      bg=BTN_BG, fg=FG_MID, font=F_SM, relief="groove", bd=2,
                      cursor="hand2", padx=7, pady=2,
                      activebackground=BTN_ACTIVE).pack(side="left", padx=(0,3))
        tk.Label(rot_row1, text="Custom:", bg=BG_CARD, fg=FG_MID, font=F_HINT).pack(side="left", padx=(8,0))
        self._batch_rot_deg = tk.DoubleVar(value=0.0)
        self._batch_rot_entry = tk.Spinbox(
            rot_row1, from_=-360, to=360, increment=1,
            textvariable=self._batch_rot_deg,
            font=F_SM, width=5, relief="groove", bd=2, bg=BTN_BG)
        self._batch_rot_entry.pack(side="left", padx=(4,0))
        self._bind_spinbox(self._batch_rot_entry, self._batch_rot_deg, -360, 360, 1, -1)
        tk.Label(rot_row1, text="°", bg=BG_CARD, fg=FG_MID, font=F_HINT).pack(side="left", padx=(2,0))

        # Expand checkbox row
        rot_row2 = tk.Frame(rot_controls, bg=BG_CARD); rot_row2.pack(fill="x", pady=(0,1))
        self._batch_rot_expand = tk.BooleanVar(value=False)
        self._batch_rot_expand_cb = tk.Checkbutton(
            rot_row2, text="Expand canvas to fit  (may change image size)",
            variable=self._batch_rot_expand,
            bg=BG_CARD, fg=FG_MID, selectcolor=BG_RAISED,
            activebackground=BG_CARD, font=F_HINT,
            command=self._batch_update_rot_preview)
        self._batch_rot_expand_cb.pack(side="left")

        # Rotation preview — right column, spans both rows
        PREV_SZ = 120 if sys.platform == "darwin" else 90
        prev_outer = tk.Frame(rot_prev_col, bg=BORDER, padx=1, pady=1)
        prev_outer.pack()
        self._batch_rot_canvas = tk.Canvas(
            prev_outer, width=PREV_SZ, height=PREV_SZ,
            bg="#303030", highlightthickness=0)
        self._batch_rot_canvas.pack()
        self._batch_rot_canvas_lbl = tk.Label(
            rot_prev_col, text="",
            bg=BG_CARD, fg=FG_DIM, font=F_HINT, anchor="center")
        self._batch_rot_canvas_lbl.pack(fill="x")

        # Trace spinbox changes for live update
        self._batch_rot_deg.trace_add("write", lambda *_: self._batch_update_rot_preview())
        self._batch_rot_deg.trace_add("write", lambda *_: self._sync_batch_rot_to_preview())

        self._batch_rotate_toggle()

        # ── 2x2 grid: OUTPUT | ANIMATION / SETTINGS SUMMARY | RUN ──
        batch_grid = tk.Frame(f, bg=BG_PANEL)
        batch_grid.pack(fill="both", expand=True)
        batch_grid.columnconfigure(0, weight=1, uniform="batchcol")
        batch_grid.columnconfigure(1, weight=1, uniform="batchcol")
        batch_grid.rowconfigure(0, weight=1)
        batch_grid.rowconfigure(1, weight=1)
        batch_grid.pack(fill="both", expand=True)
        oa_left  = tk.Frame(batch_grid, bg=BG_PANEL); oa_left.grid(row=0, column=0, sticky="nsew")
        oa_right = tk.Frame(batch_grid, bg=BG_PANEL); oa_right.grid(row=0, column=1, sticky="nsew")
        batch_left  = tk.Frame(batch_grid, bg=BG_PANEL); batch_left.grid(row=1, column=0, sticky="nsew")
        batch_right = tk.Frame(batch_grid, bg=BG_PANEL); batch_right.grid(row=1, column=1, sticky="nsew")

        co = card_frame(oa_left, "OUTPUT", ACCENT_G)

        tk.Label(co, text="Save to:", bg=BG_CARD, fg=FG_MID, font=F_HINT, anchor="w").pack(anchor="w")
        self._batch_out_dir = tk.StringVar(value="")
        of_row = tk.Frame(co, bg=BG_CARD); of_row.pack(fill="x", pady=(0,1))
        tk.Entry(of_row, textvariable=self._batch_out_dir,
                 font=F_SM, bg=BTN_BG, relief="groove", bd=2).pack(side="left", fill="x", expand=True, padx=(0,4))
        tk.Button(of_row, text="Browse…", command=self._batch_browse_output,
                  bg=BTN_BG, fg=FG_MID, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=6, pady=2,
                  activebackground=BTN_ACTIVE).pack(side="left")

        fn_row = tk.Frame(co, bg=BG_CARD); fn_row.pack(fill="x", pady=(0,1))
        tk.Label(fn_row, text="Folder name:", bg=BG_CARD, fg=FG_MID, font=F_HINT, anchor="w").pack(side="left")
        self._batch_subfolder = tk.StringVar(value="")
        tk.Entry(fn_row, textvariable=self._batch_subfolder,
                 font=F_SM, bg=BG_RAISED, relief="flat", bd=2, width=26).pack(side="left", padx=(4,0))
        self._batch_path_lbl = tk.Label(co, text="", bg=BG_CARD, fg=FG_DIM,
                                         font=("Consolas",10), anchor="w")
        self._batch_path_lbl.pack(fill="x")
        self._batch_out_dir.trace_add("write", lambda *_: self._batch_update_path_lbl())
        self._batch_subfolder.trace_add("write", lambda *_: self._batch_update_path_lbl())

        self._batch_fmt = tk.StringVar(value="TIFF 16-bit")
        fmt_row = tk.Frame(co, bg=BG_CARD); fmt_row.pack(fill="x", pady=(1,0))
        tk.Label(fmt_row, text="Format:", bg=BG_CARD, fg=FG_MID, font=F_HINT, anchor="w").pack(side="left")
        for fmt_opt, fmt_lbl in [("TIFF 16-bit","TIFF 16-bit"), ("PNG 8-bit","PNG"), ("JPEG","JPEG")]:
            tk.Radiobutton(fmt_row, text=fmt_lbl, variable=self._batch_fmt, value=fmt_opt,
                           bg=BG_CARD, fg=FG_MID, selectcolor=BG_RAISED,
                           activebackground=BG_CARD, font=F_HINT).pack(side="left", padx=(4,0))

        # ── ANIMATION ──────────────────────────────────────
        can = card_frame(oa_right, "ANIMATION", ACCENT_P)

        # Row 1: enable + format + delay + loop — all on one line
        # Row 1: Create + Format + Loop
        anim_r1 = tk.Frame(can, bg=BG_CARD); anim_r1.pack(fill="x", pady=(0,1))
        self._batch_anim_on = tk.BooleanVar(value=False)
        tk.Checkbutton(anim_r1, text="Create animation",
                       variable=self._batch_anim_on,
                       bg=BG_CARD, fg=FG_MID, selectcolor=BG_RAISED,
                       activebackground=BG_CARD, font=F_HINT).pack(side="left")
        self._batch_anim_fmt = tk.StringVar(value="WebP")
        for fmt_opt in ["WebP", "GIF"]:
            tk.Radiobutton(anim_r1, text=fmt_opt, variable=self._batch_anim_fmt, value=fmt_opt,
                           bg=BG_CARD, fg=FG_MID, selectcolor=BG_RAISED,
                           activebackground=BG_CARD, font=F_HINT).pack(side="left", padx=(6,0))
        # Delay + Loop on second row
        anim_delay_row = tk.Frame(can, bg=BG_CARD); anim_delay_row.pack(fill="x", pady=(0,2))
        tk.Label(anim_delay_row, text="Delay:", bg=BG_CARD, fg=FG_MID, font=F_HINT).pack(side="left")
        self._batch_anim_delay = tk.DoubleVar(value=0.5)
        _anim_sb = tk.Spinbox(anim_delay_row, from_=0.05, to=10.0, increment=0.05,
                   textvariable=self._batch_anim_delay,
                   font=F_SM, width=6, relief="flat", bg=BG_RAISED,
                   format="%.2f")
        _anim_sb.pack(side="left", padx=(4,0))
        self._bind_spinbox(_anim_sb, self._batch_anim_delay, 0.05, 10.0, 0.05, -1)
        tk.Label(anim_delay_row, text="s", bg=BG_CARD, fg=FG_MID, font=F_HINT).pack(side="left", padx=(2,8))
        self._batch_anim_loop = tk.BooleanVar(value=True)
        tk.Checkbutton(anim_delay_row, text="Loop",
                       variable=self._batch_anim_loop,
                       bg=BG_CARD, fg=FG_MID, selectcolor=BG_RAISED,
                       activebackground=BG_CARD, font=F_HINT).pack(side="left")

        # Row 2: filename + export button
        tk.Label(can, text="Filename:", bg=BG_CARD, fg=FG_MID, font=F_HINT, anchor="w").pack(anchor="w")
        self._batch_anim_name = tk.StringVar(value="kepler_animation")
        anim_r2 = tk.Frame(can, bg=BG_CARD); anim_r2.pack(fill="x", pady=(0,2))
        tk.Entry(anim_r2, textvariable=self._batch_anim_name,
                 font=F_SM, bg=BG_RAISED, relief="flat", bd=2).pack(side="left", fill="x", expand=True, padx=(0,4))
        tk.Button(anim_r2, text="▶ Export Now",
                  command=self._batch_export_animation_standalone,
                  bg=BTN_BG, fg=ACCENT_P, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=8, pady=2,
                  activebackground=BTN_ACTIVE).pack(side="left")

        # ── SETTINGS SUMMARY ───────────────────────────────
        cs = card_frame(batch_left, "SETTINGS SUMMARY", ACCENT_C)
        refresh_row = tk.Frame(cs, bg=BG_CARD); refresh_row.pack(fill="x", pady=(0,2))
        tk.Button(refresh_row, text="↻  Refresh Summary",
                  command=self._batch_refresh_summary,
                  bg=BTN_BG, fg=ACCENT_C, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=8, pady=2,
                  activebackground=BTN_ACTIVE).pack(side="left")
        tk.Label(cs,
                 text="Shows Wavelet / FFT / RGB / Tools / De-rind\nsettings that will be applied at Run time.",
                 bg=BG_CARD, fg=FG_DIM, font=F_XS,
                 justify="left").pack(anchor="w", pady=(0,2))
        self._batch_summary_txt = tk.Text(
            cs, height=4, font=("Consolas", 11),
            bg=BG_RAISED, fg=FG_MID, relief="flat",
            state="disabled", wrap="none")
        self._batch_summary_txt.pack(fill="x")

        # ── RUN ────────────────────────────────────────────
        cr = card_frame(batch_right, "RUN", ACCENT_G)

        run_row = tk.Frame(cr, bg=BG_CARD); run_row.pack(fill="x", pady=(0,2))
        self._batch_run_btn = tk.Button(run_row, text="▶  Run Batch",
                  command=self._batch_run,
                  bg=ACCENT_G, fg="white", font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=10, pady=4,
                  activebackground="#0d5a2a")
        self._batch_run_btn.pack(side="left")
        self._batch_cancel_btn = tk.Button(run_row, text="■  Cancel",
                  command=self._batch_cancel,
                  bg=BTN_BG, fg=ACCENT_O, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=8, pady=4,
                  activebackground=BTN_ACTIVE, state="disabled")
        self._batch_cancel_btn.pack(side="left", padx=(8,0))

        self._batch_progress_var = tk.DoubleVar(value=0.0)
        self._batch_progress_bar = ttk.Progressbar(
            cr, variable=self._batch_progress_var,
            maximum=100, length=400, mode="determinate")
        self._batch_progress_bar.pack(fill="x", pady=(2,1))

        self._batch_status_lbl = tk.Label(cr, text="Ready.", bg=BG_CARD,
                                           fg=FG_DIM, font=F_HINT, anchor="w")
        self._batch_status_lbl.pack(fill="x")

        log_frame = tk.Frame(cr, bg=BG_CARD); log_frame.pack(fill="x", pady=(4,0))
        log_scroll = tk.Scrollbar(log_frame, orient="vertical")
        self._batch_log = tk.Text(log_frame, height=3, font=("Consolas", 11),
                                   bg=BG_RAISED, fg=FG_MID, relief="flat",
                                   state="disabled", wrap="none",
                                   yscrollcommand=log_scroll.set)
        log_scroll.config(command=self._batch_log.yview)
        log_scroll.pack(side="right", fill="y")
        # Stretch all batch grid cards to equal height
        self.root.after(100, lambda: [
            _stretch_cards(oa_left), _stretch_cards(oa_right),
            _stretch_cards(batch_left), _stretch_cards(batch_right)])
        self._batch_log.pack(side="left", fill="x", expand=True)

        # Internal state
        self._batch_files   = []
        self._batch_running = False
        self._batch_stop    = False

    # ── Batch helpers ─────────────────────────────────────────

    def _batch_log_msg(self, msg):
        """Append a line to the batch log widget (thread-safe via root.after)."""
        def _append():
            self._batch_log.config(state="normal")
            self._batch_log.insert("end", msg + "\n")
            self._batch_log.see("end")
            self._batch_log.config(state="disabled")
        self.root.after(0, _append)

    def _batch_add_files(self):
        paths = filedialog.askopenfilenames(
            title="Add Files to Batch",
            filetypes=[("Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                       ("All files", "*.*")])
        for p in paths:
            if p not in self._batch_files:
                self._batch_files.append(p)
                self._batch_listbox.insert("end", os.path.basename(p))
        self._batch_count_lbl.configure(text=f"{len(self._batch_files)} file{'s' if len(self._batch_files)!=1 else ''}")

    def _batch_remove_selected(self):
        selected = sorted(self._batch_listbox.curselection(), reverse=True)
        for idx in selected:
            self._batch_listbox.delete(idx)
            del self._batch_files[idx]
        self._batch_count_lbl.configure(text=f"{len(self._batch_files)} file{'s' if len(self._batch_files)!=1 else ''}")

    def _batch_clear(self):
        self._batch_listbox.delete(0, "end")
        self._batch_files.clear()
        self._batch_count_lbl.configure(text="0 files")

    def _batch_browse_output(self):
        d = filedialog.askdirectory(title="Choose Parent Folder for Output")
        if d:
            self._batch_out_dir.set(d)
            self._batch_update_path_lbl()

    def _batch_update_path_lbl(self):
        base = self._batch_out_dir.get().strip()
        sub  = self._batch_subfolder.get().strip()
        if base and sub:
            full = os.path.join(base, sub)
            self._batch_path_lbl.configure(text=f"→  {full}")
        elif base:
            self._batch_path_lbl.configure(text=f"→  {base}  (enter a folder name above)")
        else:
            self._batch_path_lbl.configure(text="")

    def _batch_cancel(self):
        self._batch_stop = True
        self._batch_status_lbl.configure(text="Canceling…")

    def _on_preview_rotate(self):
        """Called when preview rotation spinbox changes — refresh preview and sync to batch."""
        try:
            deg = float(self._preview_rot_deg.get())
            deg = max(-360.0, min(360.0, deg))
            self._preview_rot_deg.set(round(deg, 1))
        except (ValueError, tk.TclError):
            self._preview_rot_deg.set(0.0)
        # Reset channel view to RGB on rotation change
        if hasattr(self, "_chan_view"):
            self._set_chan_view("RGB")
        # Redraw current processed result with new rotation (no re-pipeline needed)
        pil = self._get_processed_pil()
        if pil is not None:
            self._draw_on_canvas(self.cnv_proc, pil)
        # Sync to batch tab
        self._sync_preview_rot_to_batch(deg)

    def _preview_set_rotation(self, deg):
        """Set preview rotation to a specific value (preset or reset)."""
        self._preview_rot_deg.set(float(deg))
        self._on_preview_rotate()

    def _sync_batch_rot_to_preview(self):
        """Mirror batch rotation back to the preview rotation spinbox."""
        try:
            if not self._batch_rotate_on.get():
                return
            deg = float(self._batch_rot_deg.get())
            current = float(self._preview_rot_deg.get())
            if abs(deg - current) > 0.05:   # avoid feedback loop
                self._preview_rot_deg.set(round(deg, 1))
                # Redraw without triggering another sync
                pil = self._get_processed_pil()
                if pil is not None:
                    self._draw_on_canvas(self.cnv_proc, pil)
        except (AttributeError, ValueError, tk.TclError):
            pass

    def _sync_preview_rot_to_batch(self, deg):
        """Mirror the preview rotation into the Batch ALIGNMENT & ROTATION section."""
        try:
            self._batch_rot_deg.set(float(deg))
            if deg != 0.0:
                self._batch_rotate_on.set(True)
            self._batch_rotate_toggle()
            self._batch_update_rot_preview()
        except AttributeError:
            pass   # Batch tab not yet built

    def _batch_rotate_toggle(self):
        """Enable or disable rotation sub-controls based on checkbox."""
        state = "normal" if self._batch_rotate_on.get() else "disabled"
        for w in self._batch_rot_preset_frame.winfo_children():
            try: w.configure(state=state)
            except Exception: pass
        self._batch_rot_entry.configure(state=state)
        self._batch_rot_expand_cb.configure(state=state)
        self._batch_update_rot_preview()

    def _batch_update_rot_preview(self, *_):
        """Render rotated thumbnail live into the inset canvas."""
        import numpy as np
        from PIL import Image as PILImage, ImageTk as PILImageTk

        cnv = self._batch_rot_canvas
        lbl = self._batch_rot_canvas_lbl
        SIZE = int(cnv.cget("width"))

        if not self._batch_rotate_on.get():
            cnv.delete("all")
            cnv.create_rectangle(0, 0, SIZE, SIZE, fill="#303030", outline="")
            cnv.create_text(SIZE//2, SIZE//2, text="Enable\nrotation",
                            fill="#666666", font=F_HINT, justify="center")
            lbl.configure(text="")
            return

        src = self.working_arr if self.working_arr is not None else self.original_arr
        if src is None:
            cnv.delete("all")
            cnv.create_text(SIZE//2, SIZE//2, text="Load an\nimage first",
                            fill="#888888", font=F_HINT, justify="center")
            lbl.configure(text="")
            return

        try:
            deg    = float(self._batch_rot_deg.get())
        except (ValueError, tk.TclError):
            return

        expand = bool(self._batch_rot_expand.get())
        arr8   = np.clip(src * 255, 0, 255).astype(np.uint8)
        pil    = PILImage.fromarray(arr8)

        # Fit source into SIZE×SIZE first, then rotate
        scale_fit = min(SIZE / pil.width, SIZE / pil.height)
        thumb = pil.resize((max(1, int(pil.width*scale_fit)),
                             max(1, int(pil.height*scale_fit))),
                            PILImage.LANCZOS)
        rotated = thumb.rotate(-deg, expand=expand,
                               resample=PILImage.BICUBIC,
                               fillcolor=(0, 0, 0))

        # Fit rotated result into SIZE×SIZE canvas
        rw, rh = rotated.size
        scale2 = min(SIZE / rw, SIZE / rh)
        final  = rotated.resize((max(1, int(rw*scale2)),
                                  max(1, int(rh*scale2))),
                                 PILImage.LANCZOS)

        # Composite onto dark background
        bg_img = PILImage.new("RGB", (SIZE, SIZE), (0x30, 0x30, 0x30))
        ox = (SIZE - final.width)  // 2
        oy = (SIZE - final.height) // 2
        bg_img.paste(final, (ox, oy))

        tk_img = PILImageTk.PhotoImage(bg_img)
        cnv.delete("all")
        cnv.create_image(0, 0, anchor="nw", image=tk_img)
        cnv._rot_img = tk_img   # prevent GC

        dir_str = "CW" if deg > 0 else "CCW" if deg < 0 else "no rotation"
        lbl.configure(text=f"{deg:+.1f}°  {dir_str}"
                      + (f"  expanded to {rotated.width}×{rotated.height}px" if expand else ""))

    def _batch_set_rotation(self, deg):
        """Set rotation spinbox to a preset value and enable rotation."""
        self._batch_rotate_on.set(True)
        self._batch_rot_deg.set(float(deg))
        self._batch_rotate_toggle()
        self._batch_update_rot_preview()

    def _batch_export_animation_standalone(self):
        """Export animation from already-processed files in the output folder."""
        out_dir  = self._batch_out_dir.get().strip()
        subfolder = self._batch_subfolder.get().strip()
        if not out_dir or not subfolder:
            messagebox.showwarning("Animation", "Please set an output folder and folder name first.")
            return
        out_path = os.path.join(out_dir, subfolder)
        if not os.path.isdir(out_path):
            messagebox.showwarning("Animation", f"Output folder not found:\n{out_path}")
            return
        fmt = self._batch_anim_fmt.get()
        ext = ".webp" if fmt == "WebP" else ".gif"
        exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp")
        anim_name = self._batch_anim_name.get().strip() or "kepler_animation"
        # Exclude any existing animation files
        files = sorted([
            os.path.join(out_path, fn)
            for fn in os.listdir(out_path)
            if os.path.splitext(fn)[1].lower() in exts
               and not fn.startswith(anim_name)
        ])
        if not files:
            messagebox.showwarning("Animation", "No processed image files found in the output folder.")
            return
        self._build_animation(files, out_path, anim_name, fmt, ext)

    def _build_animation(self, image_paths, out_path, anim_name, fmt, ext):
        """Build and save animated WebP or GIF from a list of image file paths."""
        import numpy as np
        from PIL import Image as PILImage

        delay_ms  = max(50, int(round(self._batch_anim_delay.get() * 1000)))
        loop      = 0 if self._batch_anim_loop.get() else 1
        out_file  = os.path.join(out_path, anim_name + ext)

        try:
            frames = []
            for fp in image_paths:
                img = PILImage.open(fp).convert("RGB")
                frames.append(img)

            if not frames:
                messagebox.showwarning("Animation", "No frames to animate.")
                return

            first = frames[0]
            if fmt == "WebP":
                first.save(out_file, format="WEBP", save_all=True,
                           append_images=frames[1:],
                           duration=delay_ms, loop=loop, quality=90)
            else:  # GIF — convert to palette mode
                pal_frames = [fr.quantize(colors=256, method=PILImage.Quantize.MEDIANCUT)
                              for fr in frames]
                pal_frames[0].save(out_file, format="GIF", save_all=True,
                                   append_images=pal_frames[1:],
                                   duration=delay_ms, loop=loop, optimize=True)

            self._batch_log_msg(f"Animation saved: {os.path.basename(out_file)}  "
                                f"({len(frames)} frames, {delay_ms}ms/frame)")
            messagebox.showinfo("Animation", f"Saved: {out_file}")
        except Exception as e:
            messagebox.showerror("Animation error", str(e))

    def _batch_refresh_summary(self):
        """Read current tab settings and display a human-readable summary."""
        lines = []

        # Wavelet
        lvls = self._get_wavelet_params()
        fm   = self.wv_filter.get().split()[0]
        cm   = self.wv_color_model.get().split()[0]
        conv = self.wv_convolve.get().split()[0]
        ps   = float(self.wv_pre_smooth.get())
        pf   = float(self.wv_powerfn_exp.get()) if self.wv_powerfn_enabled.get() else None
        sf   = [float(self.wsf1.get()), float(self.wsf2.get()), float(self.wsf3.get())]
        lines.append("─── Wavelet ───────────────────────────────")
        for i,(sh,th,sz) in enumerate(lvls,1):
            active = sh > 0 or th > 0
            tag = "" if active else "  (off)"
            lines.append(f"  L{i}  Sharpen={sh}  Threshold={th}  σ={sz:.2f}px  SharpenFilter={sf[i-1]:.3f}{tag}")
        lines.append(f"  Filter={fm}  Color={cm}  Convolve={conv}"
                     + (f"  Pre-Smooth={ps:.2f}" if ps>0 else "")
                     + (f"  PowerFn={pf:.2f}" if pf else ""))

        # FFT
        fft_on = self.fft_enabled.get()
        lines.append("─── FFT Denoise ───────────────────────────")
        if fft_on:
            _act = [s for s in self.fft_stage_ids if self.fft_stage_on[s].get()]
            if _act:
                for s in _act:
                    p = self.fft_params[s]
                    lines.append(f"  {s:<5} Start={p['start']:.0f}%  "
                                 f"End={p['end']:.0f}%  Curve={p['curve']:.0f}")
            else:
                lines.append("  Enabled, but no stage checked")
        else:
            lines.append("  Disabled")

        # RGB
        rg  = float(self.rgb_r.get())/2;  gg  = float(self.rgb_g.get())/2;  bg_ = float(self.rgb_b.get())/2
        rgm = float(self.gamma_r.get())/100; ggm = float(self.gamma_g.get())/100; bgm = float(self.gamma_b.get())/100
        sat = float(self.saturation.get())/2; vib = float(self.vibrance.get())/2
        hue = float(self.hue_rot.get()); bri = float(self.brightness.get()); con = float(self.contrast.get())/100
        lines.append("─── RGB / Color ──────────────────────────")
        lines.append(f"  Gain  R={rg:.0f}  G={gg:.0f}  B={bg_:.0f}"
                     + f"    Gamma  R={rgm:.2f}  G={ggm:.2f}  B={bgm:.2f}")
        lines.append(f"  Sat={sat:.0f}  Vib={vib:.0f}  Hue={hue:.0f}°"
                     + f"  Brightness={bri:.0f}  Contrast={con:.2f}")

        # De-rind
        dr_on = self.dr_enabled.get()
        lines.append("─── De-rind ───────────────────────────────")
        if dr_on:
            lines.append(f"  Enabled  Edge={self.dr_edge.get():.0f}px"
                         + f"  Smooth={self.dr_smooth.get():.0f}  Inset={self.dr_inset.get():.0f}px")
        else:
            lines.append("  Disabled")

        # Moon Recovery
        lines.append("─── Moon Recovery ─────────────────────────")
        if self.mr_enabled.get():
            nc = len(self._mr_circles) if hasattr(self, "_mr_circles") else 0
            lines.append(f"  Enabled  {nc} circle(s)"
                         + f"  Boost={self.mr_boost.get()/10:.1f}\u00d7"
                         + f"  Feather={self.mr_feather.get():.0f}px"
                         + (f"  Darken={self.mr_darken_edge.get():.0f}%" if hasattr(self, "mr_darken_edge") and self.mr_darken_edge.get() > 0 else ""))
        else:
            lines.append("  Disabled")

        # Write to text widget
        self._batch_summary_txt.config(state="normal")
        self._batch_summary_txt.delete("1.0", "end")
        self._batch_summary_txt.insert("end", "\n".join(lines))
        self._batch_summary_txt.config(state="disabled")

    def _batch_align_image(self, arr, ref_cy, ref_cx):
        """Shift arr so its disk center aligns to (ref_cy, ref_cx). Returns shifted array."""
        import numpy as np
        lum = 0.299*arr[...,0]+0.587*arr[...,1]+0.114*arr[...,2]
        lum_max = float(np.percentile(lum, 99.5))
        thresh  = max(lum_max * 0.15, 0.01)
        disk    = lum > thresh
        ys, xs  = np.where(disk)
        if len(ys) == 0:
            return arr
        cy, cx = float(ys.mean()), float(xs.mean())
        dy = int(round(ref_cy - cy))
        dx = int(round(ref_cx - cx))
        if dy == 0 and dx == 0:
            return arr
        from scipy.ndimage import shift
        shifted = np.stack([
            shift(arr[...,c], (dy, dx), mode="constant", cval=0.0)
            for c in range(3)
        ], axis=-1)
        return np.clip(shifted, 0.0, 1.0).astype(arr.dtype)

    def _batch_run(self):
        """Collect current pipeline params and launch batch in a background thread."""
        import threading, numpy as np

        if self._batch_running:
            return
        if not self._batch_files:
            messagebox.showwarning("Batch", "No files added to batch."); return
        out_dir   = self._batch_out_dir.get().strip()
        subfolder = self._batch_subfolder.get().strip()
        if not out_dir:
            messagebox.showwarning("Batch", "Please choose a save-to folder."); return
        if not subfolder:
            messagebox.showwarning("Batch", "Please enter a name for the output folder."); return
        out_path = os.path.join(out_dir, subfolder)

        # Capture ALL current pipeline settings on main thread
        wv_params = self._get_wavelet_params()
        wv_fm   = self.wv_filter.get().split()[0]
        wv_cm   = self.wv_color_model.get().split()[0]
        wv_conv = self.wv_convolve.get().split()[0]
        wv_sf   = [float(self.wsf1.get()), float(self.wsf2.get()), float(self.wsf3.get())]
        wv_ps   = float(self.wv_pre_smooth.get())
        wv_cd   = float(self._cd_radius.get()) if self._cd_enabled.get() and wv_conv in ("RGB","LRGB") else 0.0
        wv_pf   = float(self.wv_powerfn_exp.get()) if self.wv_powerfn_enabled.get() else 1.0
        wv_zgf  = float(self.wv_zgauss_factor.get()) if hasattr(self, "wv_zgauss_factor") else 1.0
        wv_br   = float(self.wv_bilateral_radius.get()) if hasattr(self, "wv_bilateral_radius") else 2.0
        wv_bls  = [self.wbl1.get(), self.wbl2.get(), self.wbl3.get()]                   if wv_fm == "bilateral" and hasattr(self, "wbl1") else None
        fft_pre, fft_post = self._fft_collect()
        r_g  = float(self.rgb_r.get())/100;  g_g  = float(self.rgb_g.get())/100
        b_g  = float(self.rgb_b.get())/100;  r_gm = float(self.gamma_r.get())/100
        g_gm = float(self.gamma_g.get())/100; b_gm = float(self.gamma_b.get())/100
        r_bp = float(self.black_r.get())/100; g_bp = float(self.black_g.get())/100
        b_bp = float(self.black_b.get())/100
        sat  = float(self.saturation.get())/100
        vib  = float(self.vibrance.get())/100
        hue  = float(self.hue_rot.get())
        bri  = float(self.brightness.get())/255
        con  = float(self.contrast.get())/100
        # ── Tools ──
        b_dc_on   = bool(self._dc_enabled.get())
        b_dc_str  = float(self._dc_strength.get()) if b_dc_on else 0.0
        b_dc_con  = bool(self._dc_use_contrast.get())
        b_dc_cstr = float(self._dc_contrast_str.get())
        b_dh_on   = bool(self._dh_enabled.get())
        b_dh_bs   = max(1, int(float(self._dh_blocksize.get()))) if b_dh_on else 5
        b_dh_amt  = float(self._dh_amount.get()) if b_dh_on else 0.0
        _lc_b = self._get_lc_params()
        b_tl_br,b_tl_wr,b_tl_gr, b_tl_bg,b_tl_wg,b_tl_gg, b_tl_bb,b_tl_wb,b_tl_gb,         b_tl_lr,b_tl_lg,b_tl_lb, b_tl_def = _lc_b
        b_clahe_on   = bool(self._clahe_enabled.get())
        b_clahe_clip = float(self._clahe_clip.get())
        b_clahe_tile = max(4, int(float(self._clahe_tile.get())))
        b_clahe_str  = float(self._clahe_strength.get())
        b_clahe_ch   = self._clahe_channel.get()
        dr_on = bool(self.dr_enabled.get())
        dr_p  = dict(
            edge=float(self.dr_edge.get()),
            smooth=float(self.dr_smooth.get()), inset=float(self.dr_inset.get()),
            gap_width=float(self.dr_gap_width.get()), gap_angle=float(self.dr_gap_angle.get()),
            saturn_mode=bool(self.dr_saturn.get()), dark_edge=bool(self.dr_dark_edge.get()),
            pre_blur=float(self.dr_pre_blur.get()), show_ring_map=False,
        )
        do_align    = bool(self._batch_align.get())
        align_rgb_on = bool(self._align_rgb.get()) if getattr(self, "_align_rgb", None) is not None else False
        do_rotate   = bool(self._batch_rotate_on.get())
        rot_deg     = float(self._batch_rot_deg.get()) if do_rotate else 0.0
        rot_expand  = bool(self._batch_rot_expand.get())
        do_anim     = bool(self._batch_anim_on.get())
        anim_fmt    = self._batch_anim_fmt.get()
        anim_ext    = ".webp" if anim_fmt == "WebP" else ".gif"
        anim_name   = self._batch_anim_name.get().strip() or "kepler_animation"
        fmt         = self._batch_fmt.get()
        suffix      = "_KEP"
        files    = list(self._batch_files)
        n        = len(files)

        # Update UI
        self._batch_running = True
        self._batch_stop    = False
        self._batch_run_btn.configure(state="disabled")
        self._batch_cancel_btn.configure(state="normal")
        self._batch_progress_var.set(0)
        self._batch_log.config(state="normal"); self._batch_log.delete("1.0","end"); self._batch_log.config(state="disabled")
        self._batch_status_lbl.configure(text=f"Starting batch of {n} files…")

        def _worker():
            import tifffile
            from PIL import Image as PILImage

            try:
                os.makedirs(out_path, exist_ok=True)
                self._batch_log_msg(f"Output: {out_path}")

                # Determine reference disk center (first file) for alignment
                ref_cy, ref_cx  = None, None
                processed_files = []   # (img_array, original_filename) for animation

                for i, fpath in enumerate(files):
                    if self._batch_stop:
                        self._batch_log_msg("— Canceled —")
                        break

                    fname = os.path.basename(fpath)
                    self.root.after(0, lambda i=i,n=n,fname=fname:
                        self._batch_status_lbl.configure(text=f"[{i+1}/{n}]  {fname}"))
                    self._batch_log_msg(f"[{i+1}/{n}]  {fname}")

                    # Load
                    ext = os.path.splitext(fpath)[1].lower()
                    try:
                        if ext in (".tif",".tiff"):
                            raw = tifffile.imread(fpath)
                            if raw.ndim == 2: raw = np.stack([raw]*3, axis=-1)
                            elif raw.shape[-1] > 3: raw = raw[...,:3]
                            if raw.dtype == np.uint16:
                                arr = raw.astype(np.float32)/65535.0
                            else:
                                arr = raw.astype(np.float32)
                                mn,mx=arr.min(),arr.max()
                                if mx>mn: arr=(arr-mn)/(mx-mn)
                        elif ext == ".png":
                            arr = proc.load_png(fpath)
                        else:
                            pil = PILImage.open(fpath).convert("RGB")
                            arr = np.array(pil).astype(np.float32)/255.0
                    except Exception as e:
                        self._batch_log_msg(f"  ✕ Load error: {e}"); continue

                    # RGB channel registration (before everything else)
                    if align_rgb_on:
                        arr, _rgb_sh = proc.align_rgb_channels(arr)
                        self._batch_log_msg(
                            "  RGB align: R({:+.1f},{:+.1f}) B({:+.1f},{:+.1f})px".format(
                                _rgb_sh["R"][1], _rgb_sh["R"][0],
                                _rgb_sh["B"][1], _rgb_sh["B"][0]))

                    # Alignment
                    if do_align:
                        lum = 0.299*arr[...,0]+0.587*arr[...,1]+0.114*arr[...,2]
                        lmax = float(np.percentile(lum,99.5))
                        disk = lum > max(lmax*0.15,0.01)
                        ys,xs = np.where(disk)
                        if len(ys)>0:
                            cy,cx = float(ys.mean()),float(xs.mean())
                            if ref_cy is None:
                                ref_cy,ref_cx = cy,cx
                                self._batch_log_msg(f"  Reference center: ({cx:.1f},{cy:.1f})")
                            else:
                                arr = self._batch_align_image(arr, ref_cy, ref_cx)
                                self._batch_log_msg(f"  Aligned: shift ({cx-ref_cx:+.1f},{cy-ref_cy:+.1f})px")

                    # Pipeline
                    img = proc.wavelet_sharpen(arr, wv_params, wv_fm,
                                               color_model=wv_cm, convolve=wv_conv,
                                               pre_fft=fft_pre, sharp_filter=wv_sf,
                                               pre_smooth=wv_ps, power_fn=wv_pf,
                                               zgauss_factor=wv_zgf,
                                               bilateral_radius=wv_br,
                                               bilateral_layers=wv_bls,
                                               color_denoise=wv_cd)
                    if fft_post is not None:
                        img = proc.fft_denoise(img, fft_start=fft_post[0],
                                               fft_width=fft_post[1],
                                               fft_curve=fft_post[2])
                    img = proc.apply_color_adjustments(
                        img, r_gain=r_g, g_gain=g_g, b_gain=b_g,
                        r_black=r_bp, g_black=g_bp, b_black=b_bp,
                        r_gamma=r_gm, g_gamma=g_gm, b_gamma=b_gm,
                        saturation=sat, vibrance=vib, hue_rotation=hue,
                        brightness=bri, contrast=con)
                    if b_dc_str > 0.0:
                        img = proc.apply_deconvolution(img, strength=b_dc_str,
                            use_contrast=b_dc_con, contrast_strength=b_dc_cstr)
                    if b_dh_on:
                        img = proc.apply_dehaze(img, block_size=b_dh_bs, amount=b_dh_amt)
                    if not b_tl_def:
                        img = proc.apply_levels_curves(
                            img,
                            black_r=b_tl_br, white_r=b_tl_wr, gamma_r=b_tl_gr,
                            black_g=b_tl_bg, white_g=b_tl_wg, gamma_g=b_tl_gg,
                            black_b=b_tl_bb, white_b=b_tl_wb, gamma_b=b_tl_gb,
                            curve_lut_r=b_tl_lr, curve_lut_g=b_tl_lg, curve_lut_b=b_tl_lb)
                    if b_clahe_on:
                        img = proc.apply_clahe(img,
                            clip_limit=b_clahe_clip, tile_grid=b_clahe_tile,
                            channel_mode=b_clahe_ch, strength=b_clahe_str)
                    if dr_on:
                        _dr_p = {**dr_p, 'ref_lum': proc.luminance(arr.astype(np.float32)),
                                 'ref_arr': arr}
                        img = proc.apply_derind(img, **_dr_p)

                    # Rotation (applied after pipeline)
                    if do_rotate and rot_deg != 0.0:
                        pil_img = PILImage.fromarray(
                            np.clip(img*255,0,255).astype(np.uint8))
                        # PIL rotate: positive = CCW internally; negate to match convention
                        # (positive input = CW, negative = CCW)
                        pil_img = pil_img.rotate(
                            -rot_deg, expand=rot_expand,
                            resample=PILImage.BICUBIC,
                            fillcolor=(0,0,0))
                        img = np.array(pil_img).astype(np.float32) / 255.0

                    processed_files.append((img, fname))

                    # Save
                    stem = os.path.splitext(fname)[0]
                    if fmt == "TIFF 16-bit":
                        out_fname = stem + suffix + ".tif"
                        arr16 = np.clip(img*65535.0,0,65535).astype(np.uint16)
                        tifffile.imwrite(os.path.join(out_path, out_fname), arr16)
                    elif fmt == "PNG 8-bit":
                        out_fname = stem + suffix + ".png"
                        arr8 = np.clip(img*255.0,0,255).astype(np.uint8)
                        PILImage.fromarray(arr8).save(os.path.join(out_path, out_fname))
                    else:  # JPEG
                        out_fname = stem + suffix + ".jpg"
                        arr8 = np.clip(img*255.0,0,255).astype(np.uint8)
                        PILImage.fromarray(arr8).save(os.path.join(out_path, out_fname), quality=95)

                    self._batch_log_msg(f"  ✔ Saved: {out_fname}")
                    pct = (i+1)/n*100
                    self.root.after(0, lambda p=pct: self._batch_progress_var.set(p))

                # Build animation from processed files
                if do_anim and processed_files and not self._batch_stop:
                    self.root.after(0, lambda: self._batch_status_lbl.configure(
                        text="Building animation…"))
                    self._batch_log_msg(f"Building {anim_fmt} animation ({len(processed_files)} frames)…")
                    saved_paths = []
                    for _, orig_fname in processed_files:
                        stem = os.path.splitext(orig_fname)[0]
                        ext_map = {"TIFF 16-bit": ".tif", "PNG 8-bit": ".png", "JPEG": ".jpg"}
                        saved_paths.append(os.path.join(out_path, stem + suffix + ext_map.get(fmt, ".tif")))
                    self._build_animation(saved_paths, out_path, anim_name, anim_fmt, anim_ext)

                if not self._batch_stop:
                    done_msg = f"Done — {n} file{'s' if n!=1 else ''} processed."
                    self._batch_log_msg(done_msg)
                    self.root.after(0, lambda: self._batch_status_lbl.configure(text=done_msg))

            except Exception as e:
                self._batch_log_msg(f"Batch error: {e}")
                self.root.after(0, lambda: self._batch_status_lbl.configure(text=f"Error: {e}"))
            finally:
                self._batch_running = False
                self._batch_stop    = False
                self.root.after(0, lambda: self._batch_run_btn.configure(state="normal"))
                self.root.after(0, lambda: self._batch_cancel_btn.configure(state="disabled"))

        import threading
        threading.Thread(target=_worker, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    #  🪐 DE-ROTATE TAB
    # ══════════════════════════════════════════════════════════════════════════

    def _build_derotate_tab(self):
        f = self._scrollable_tab("🪐 De-rotate")

        # ── Info card ─────────────────────────────────────────────────────────
        ci = card_frame(f, "PLANETARY DE-ROTATION", ACCENT_P)
        tk.Label(ci,
            text=("Uses IAU-2018 geometry + JPL Horizons. Each frame's UT is read from its filename.\n"
                  "Click Measure images to fit a wireframe to each frame — that supplies the disc\n"
                  "center, scale and the true pole orientation on your sensor, so the images do NOT\n"
                  "need to be rotated pole-up first. Run alone will auto-fit, but measuring is safer."),
            bg=BG_CARD, fg=FG_DIM, font=F_XS,
            justify="left", anchor="w").pack(anchor="w", pady=(0, 2))

        # ── Input files card ──────────────────────────────────────────────────
        cif = card_frame(f, "INPUT FILES", ACCENT_P)
        btn_row = tk.Frame(cif, bg=BG_CARD); btn_row.pack(fill="x", pady=(0, 2))
        tk.Button(btn_row, text="＋ Add",
                  command=self._dr2_add,
                  bg=BTN_BG, fg=ACCENT_P, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=7, pady=2,
                  activebackground=BTN_ACTIVE).pack(side="left")
        tk.Button(btn_row, text="✕ Remove",
                  command=self._dr2_remove,
                  bg=BTN_BG, fg=ACCENT_O, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=7, pady=2,
                  activebackground=BTN_ACTIVE).pack(side="left", padx=(4, 0))
        tk.Button(btn_row, text="⊘ Clear",
                  command=self._dr2_clear,
                  bg=BTN_BG, fg=FG_MID, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=7, pady=2,
                  activebackground=BTN_ACTIVE).pack(side="left", padx=(4, 0))
        self._dr2_count_lbl = tk.Label(btn_row, text="0 files",
                                        bg=BG_CARD, fg=FG_DIM, font=F_HINT)
        self._dr2_count_lbl.pack(side="left", padx=10)

        lb_frame = tk.Frame(cif, bg=BG_CARD); lb_frame.pack(fill="x")
        lb_scroll = tk.Scrollbar(lb_frame, orient="vertical")
        self._dr2_lb = tk.Listbox(
            lb_frame, height=5, font=F_SM, bg=BG_RAISED, fg=FG_MID,
            selectbackground=SEL_BG, relief="flat", bd=0,
            activestyle="none", yscrollcommand=lb_scroll.set,
            selectmode="extended")
        lb_scroll.config(command=self._dr2_lb.yview)
        lb_scroll.pack(side="right", fill="y")
        self._dr2_lb.pack(side="left", fill="x", expand=True)

        # ── Options ───────────────────────────────────────────────────────────
        co = card_frame(f, "OPTIONS", ACCENT_C)

        # Planet
        row1 = tk.Frame(co, bg=BG_CARD); row1.pack(fill="x", pady=(0, 4))
        tk.Label(row1, text="Planet:", bg=BG_CARD, fg=FG_MID,
                 font=F_HINT).pack(side="left")
        _saved_planet = load_prefs().get("derotate_planet", "JUPITER")
        if _saved_planet not in ("JUPITER", "SATURN", "MARS"):
            _saved_planet = "JUPITER"
        self._dr2_planet = tk.StringVar(value=_saved_planet)
        for planet in ("JUPITER", "SATURN", "MARS"):
            tk.Radiobutton(row1, text=planet.capitalize(),
                           variable=self._dr2_planet, value=planet,
                           bg=BG_CARD, fg=FG_MID, font=F_HINT,
                           selectcolor=BG_RAISED,
                           activebackground=BG_CARD).pack(side="left", padx=(8, 0))

        # De-rotation LD — algorithmic edge cleanup (separate from the
        # measurement-window LD visual aid).  1.00 = off; 0.5–0.9 darkens the
        # limb to stop bright fringes / dark halos on the stacked result.
        ldrow = tk.Frame(co, bg=BG_CARD); ldrow.pack(fill="x", pady=(0, 4))
        tk.Label(ldrow, text="De-rotation LD:", bg=BG_CARD, fg=FG_MID,
                 font=F_HINT).pack(side="left")
        _dr_ld = load_prefs().get("derotate_ld", {})
        self._dr2_ld_value = tk.StringVar(
            value=f"{float(_dr_ld.get('value', 1.0)):.2f}")
        self._dr2_ld_angle = tk.StringVar(
            value=f"{int(_dr_ld.get('angle', 65))}")
        tk.Label(ldrow, text=" value", bg=BG_CARD, fg=FG_MID,
                 font=F_HINT).pack(side="left")
        tk.Spinbox(ldrow, from_=0.50, to=1.00, increment=0.05, width=5,
                   textvariable=self._dr2_ld_value, format="%.2f",
                   justify="right", bg=BG_CARD, fg=FG_MID, relief="groove", bd=2,
                   font=F_HINT).pack(side="left", padx=(2, 8))
        tk.Label(ldrow, text="angle", bg=BG_CARD, fg=FG_MID,
                 font=F_HINT).pack(side="left")
        tk.Spinbox(ldrow, from_=0, to=95, increment=5, width=4,
                   textvariable=self._dr2_ld_angle, justify="right",
                   bg=BG_CARD, fg=FG_MID, relief="groove", bd=2,
                   font=F_HINT).pack(side="left", padx=(2, 8))
        tk.Label(ldrow, text="1.00 = off · lower cleans up edge fringes",
                 bg=BG_CARD, fg=FG_DIM, font=F_XS).pack(side="left")

        # Output folder
        row2 = tk.Frame(co, bg=BG_CARD); row2.pack(fill="x", pady=(0, 4))
        tk.Label(row2, text="Output folder:", bg=BG_CARD, fg=FG_MID,
                 font=F_HINT).pack(side="left")
        self._dr2_out_dir = tk.StringVar(value="")
        tk.Entry(row2, textvariable=self._dr2_out_dir, font=F_SM,
                 bg=BTN_BG, relief="groove", bd=2, width=38).pack(
            side="left", padx=(4, 4))
        tk.Button(row2, text="Browse…",
                  command=self._dr2_browse_out,
                  bg=BTN_BG, fg=FG_MID, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=6, pady=2,
                  activebackground=BTN_ACTIVE).pack(side="left")
        tk.Label(row2, text="(blank = auto de-rotated/ subfolder)",
                 bg=BG_CARD, fg=FG_DIM, font=F_XS).pack(side="left", padx=(8, 0))

        # ── Observer location ─────────────────────────────────────────────────
        cloc = card_frame(f, "OBSERVER LOCATION", ACCENT_C)
        tk.Label(cloc,
            text="Enter your observing site coordinates and click Save. Stored in ~/.kepler_ephem/config.json.",
            bg=BG_CARD, fg=FG_DIM, font=F_XS).pack(anchor="w", pady=(0, 4))

        # Load saved site
        try:
            from pvol_derotate import load_config as _lc
            _site = _lc()
        except Exception:
            _site = {"lat": 0.0, "lon": 0.0, "elev": 0.0}

        loc_row = tk.Frame(cloc, bg=BG_CARD); loc_row.pack(fill="x")
        self._dr2_lat  = tk.StringVar(value=f"{_site['lat']:.6f}")
        self._dr2_lon  = tk.StringVar(value=f"{_site['lon']:.6f}")
        self._dr2_elev = tk.StringVar(value=f"{_site['elev']:.1f}")
        for lbl, var, width in [
                ("Lat (°):", self._dr2_lat, 12),
                ("Lon (°):", self._dr2_lon, 12),
                ("Elev (m):", self._dr2_elev, 8)]:
            tk.Label(loc_row, text=lbl, bg=BG_CARD, fg=FG_MID,
                     font=F_HINT).pack(side="left", padx=(0, 2))
            tk.Entry(loc_row, textvariable=var, font=F_SM,
                     bg=BTN_BG, relief="groove", bd=2,
                     width=width).pack(side="left", padx=(0, 12))
        tk.Button(loc_row, text="Save",
                  command=self._dr2_save_loc,
                  bg=BTN_BG, fg=FG_MID, font=F_SM, relief="groove", bd=2,
                  cursor="hand2", padx=6, pady=2,
                  activebackground=BTN_ACTIVE).pack(side="left")

        # Offline pre-download: cache the whole current year for all planets so
        # de-rotation works without internet later.  Live fetch already covers
        # any date on demand; this is the offline convenience the installer
        # runs once, exposed here so a user can top up without reinstalling.
        dl_row = tk.Frame(cloc, bg=BG_CARD); dl_row.pack(fill="x", pady=(6, 0))
        self._dr2_dl_btn = tk.Button(
            dl_row, text="Download this year's ephemeris",
            command=self._dr2_download_ephem,
            bg=BTN_BG, fg=ACCENT_C, font=F_SM, relief="groove", bd=2,
            cursor="hand2", padx=8, pady=2, activebackground=BTN_ACTIVE)
        self._dr2_dl_btn.pack(side="left")
        tk.Label(dl_row,
            text="Jupiter, Saturn & Mars — for offline de-rotation at your saved location.",
            bg=BG_CARD, fg=FG_DIM, font=F_XS).pack(side="left", padx=(8, 0))

        # ── Run controls ──────────────────────────────────────────────────────
        run_row = tk.Frame(f, bg=BG_PANEL); run_row.pack(fill="x", padx=4, pady=(4, 2))
        self._dr2_run_btn = tk.Button(
            run_row, text="🪐  Run De-Rotation",
            command=self._dr2_run,
            bg=ACCENT_P, fg="white", font=F_SM, relief="groove", bd=2,
            cursor="hand2", padx=12, pady=5,
            activebackground="#3b1a8a")
        self._dr2_run_btn.pack(side="left")
        self._dr2_cancel_btn = tk.Button(
            run_row, text="■  Cancel",
            command=self._dr2_cancel,
            bg=BTN_BG, fg=ACCENT_O, font=F_SM, relief="groove", bd=2,
            cursor="hand2", padx=8, pady=5,
            activebackground=BTN_ACTIVE, state="disabled")
        self._dr2_cancel_btn.pack(side="left", padx=(8, 0))
        self._dr2_measure_btn = tk.Button(
            run_row, text="◉  Measure images…",
            command=self._dr2_measure,
            bg=BTN_BG, fg=ACCENT_C, font=F_SM, relief="groove", bd=2,
            cursor="hand2", padx=8, pady=5,
            activebackground=BTN_ACTIVE)
        self._dr2_measure_btn.pack(side="left", padx=(8, 0))

        self._dr2_prog_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(f, variable=self._dr2_prog_var,
                         maximum=100, mode="determinate").pack(
            fill="x", padx=4, pady=(2, 1))

        self._dr2_meas = None            # per-frame wireframe measurements
        self._dr2_meas_status = tk.Label(
            f, text="Not measured — Run will auto-fit the wireframe.",
            bg=BG_PANEL, fg=FG_DIM, font=F_HINT, anchor="w")
        self._dr2_meas_status.pack(fill="x", padx=4)

        self._dr2_status = tk.Label(f, text="Ready.",
                                     bg=BG_PANEL, fg=FG_DIM,
                                     font=F_HINT, anchor="w")
        self._dr2_status.pack(fill="x", padx=4)

        # Log
        log_frame  = tk.Frame(f, bg=BG_PANEL); log_frame.pack(fill="both", expand=True, padx=4)
        log_scroll = tk.Scrollbar(log_frame, orient="vertical")
        self._dr2_log = tk.Text(
            log_frame, height=8, font=("Consolas", 10),
            bg=BG_RAISED, fg=FG_MID, relief="flat",
            state="disabled", wrap="none",
            yscrollcommand=log_scroll.set)
        log_scroll.config(command=self._dr2_log.yview)
        log_scroll.pack(side="right", fill="y")
        self._dr2_log.pack(side="left", fill="both", expand=True)

        # Internal state
        self._dr2_files   = []
        self._dr2_running = False
        self._dr2_stop    = False

    # ── De-rotate helpers ─────────────────────────────────────────────────────

    def _dr2_log_msg(self, msg: str):
        def _append():
            self._dr2_log.config(state="normal")
            self._dr2_log.insert("end", msg + "\n")
            self._dr2_log.see("end")
            self._dr2_log.config(state="disabled")
        self.root.after(0, _append)

    def _dr2_add(self):
        from tkinter import filedialog as _fd
        paths = _fd.askopenfilenames(
            title="Add Files for De-Rotation",
            filetypes=[("Image files", "*.tif *.tiff *.png"),
                       ("All files", "*.*")])
        for p in sorted(paths):
            if p not in self._dr2_files:
                self._dr2_files.append(p)
                self._dr2_lb.insert("end", os.path.basename(p))
        self._dr2_invalidate_meas()
        n = len(self._dr2_files)
        self._dr2_count_lbl.configure(
            text=f"{n} file{'s' if n != 1 else ''}")
        if self._dr2_files and not self._dr2_out_dir.get():
            auto = os.path.join(
                os.path.dirname(self._dr2_files[0]), "de-rotated")
            self._dr2_out_dir.set(auto)

    def _dr2_remove(self):
        for idx in sorted(self._dr2_lb.curselection(), reverse=True):
            self._dr2_lb.delete(idx)
            del self._dr2_files[idx]
        self._dr2_invalidate_meas()
        n = len(self._dr2_files)
        self._dr2_count_lbl.configure(
            text=f"{n} file{'s' if n != 1 else ''}")

    def _dr2_clear(self):
        self._dr2_lb.delete(0, "end")
        self._dr2_files.clear()
        self._dr2_invalidate_meas()
        self._dr2_count_lbl.configure(text="0 files")
        self._dr2_out_dir.set("")

    def _dr2_browse_out(self):
        from tkinter import filedialog as _fd
        d = _fd.askdirectory(title="Choose Output Folder")
        if d:
            self._dr2_out_dir.set(d)

    def _dr2_save_loc(self):
        try:
            site = {"lat":  float(self._dr2_lat.get()),
                    "lon":  float(self._dr2_lon.get()),
                    "elev": float(self._dr2_elev.get())}
        except ValueError:
            messagebox.showerror("De-Rotation", "Lat/lon/elev must be numeric.")
            return
        try:
            from pvol_derotate import save_config as _sc
            _sc(site)
            self._dr2_log_msg(
                f"Location saved: lat={site['lat']:.6f}  "
                f"lon={site['lon']:.6f}  elev={site['elev']:.0f} m")
        except Exception as e:
            messagebox.showerror("De-Rotation", f"Could not save location: {e}")

    def _dr2_cancel(self):
        self._dr2_stop = True
        self._dr2_status.configure(text="Canceling…")

    def _dr2_download_ephem(self):
        """Pre-download the current year's ephemeris for all three planets."""
        if self._dr2_running:
            return
        try:
            site = {"lat":  float(self._dr2_lat.get()),
                    "lon":  float(self._dr2_lon.get()),
                    "elev": float(self._dr2_elev.get())}
        except ValueError:
            messagebox.showerror("Ephemeris",
                                 "Observer lat/lon/elev must be numeric.")
            return
        try:
            import ephem_cache
            from pvol_derotate import save_config as _sc
        except Exception as e:
            messagebox.showerror("Ephemeris", f"ephem_cache not available:\n{e}")
            return

        import datetime as _dt
        year    = _dt.datetime.now(_dt.timezone.utc).year
        planets = ["JUPITER", "SATURN", "MARS"]

        # Persist the site first so de-rotation later uses the same coordinates
        # the cache was built for (Horizons vectors are topocentric).
        try:
            _sc(site)
        except Exception:
            pass

        self._dr2_running = True
        self._dr2_stop    = False
        self._dr2_dl_btn.configure(state="disabled")
        self._dr2_run_btn.configure(state="disabled")
        self._dr2_measure_btn.configure(state="disabled")
        self._dr2_cancel_btn.configure(state="normal")
        self._dr2_prog_var.set(0.0)
        self._dr2_status.configure(text=f"Downloading {year} ephemeris…")
        self._dr2_log_msg(
            f"Downloading {year} ephemeris for Jupiter, Saturn & Mars "
            f"at lat={site['lat']:.4f}, lon={site['lon']:.4f} — a few minutes.")

        def _worker():
            ok_all = True
            try:
                span = 100.0 / len(planets)
                for i, planet in enumerate(planets):
                    if self._dr2_stop:
                        ok_all = False
                        break
                    base = i * span

                    def _prog(pct, _b=base, _s=span):
                        self.root.after(
                            0, lambda: self._dr2_prog_var.set(_b + pct * _s / 100.0))

                    self.root.after(0, lambda p=planet: self._dr2_status.configure(
                        text=f"Downloading {p.title()} {year}…"))
                    ok = ephem_cache.populate_cache(
                        planet, year, site,
                        log_callback=self._dr2_log_msg,
                        progress_callback=_prog,
                        should_stop=lambda: self._dr2_stop)
                    ok_all = ok_all and ok
                    if not ok and not self._dr2_stop:
                        self._dr2_log_msg(
                            f"  ⚠ {planet.title()} failed — check your "
                            f"internet connection.")

                if self._dr2_stop:
                    self._dr2_log_msg("Canceled.")
                    self.root.after(0, lambda: self._dr2_status.configure(
                        text="Ephemeris download canceled."))
                elif ok_all:
                    mb = ephem_cache.cache_size_mb()
                    self.root.after(0, lambda: self._dr2_prog_var.set(100.0))
                    self.root.after(0, lambda: self._dr2_status.configure(
                        text=f"{year} ephemeris ready — cache {mb:.1f} MB. "
                             f"Offline de-rotation available."))
                    self._dr2_log_msg(f"✔ Done. Cache is now {mb:.1f} MB.")
                else:
                    self.root.after(0, lambda: self._dr2_status.configure(
                        text="Ephemeris download incomplete — see log."))
            except Exception as e:
                import traceback
                self._dr2_log_msg(f"\nError: {e}\n{traceback.format_exc()}")
                self.root.after(0, lambda: self._dr2_status.configure(
                    text=f"Error: {e}"))
            finally:
                self._dr2_running = False
                self._dr2_stop    = False
                self.root.after(0, lambda: self._dr2_dl_btn.configure(state="normal"))
                self.root.after(0, lambda: self._dr2_run_btn.configure(state="normal"))
                self.root.after(0, lambda: self._dr2_measure_btn.configure(
                    state="normal"))
                self.root.after(0, lambda: self._dr2_cancel_btn.configure(
                    state="disabled"))

        import threading
        threading.Thread(target=_worker, daemon=True).start()

    def _dr2_invalidate_meas(self):
        self._dr2_meas = None
        if hasattr(self, "_dr2_meas_status"):
            self._dr2_meas_status.configure(
                text="Not measured — Run will auto-fit the wireframe.")

    def _dr2_measure(self):
        """Open the wireframe measurement editor.

        Each frame's UT is parsed from its filename, so the ephemeris is
        looked up per frame — no manual time entry."""
        from pathlib import Path as _Path
        if len(self._dr2_files) < 1:
            messagebox.showerror("Measure", "Add some images first.")
            return
        try:
            from pvol_derotate import (MeasurementWindow, parse_ut_from_filename,
                                       fetch_horizons_vectors, compute_cm_iau2018,
                                       compute_sub_earth_lat,
                                       compute_pole_position_angle)
        except Exception as e:
            messagebox.showerror("Measure", f"pvol_derotate.py not found:\n{e}")
            return
        planet = self._dr2_planet.get().upper()
        files  = [_Path(p) for p in self._dr2_files]
        site   = {"lat":  float(self._dr2_lat.get()),
                  "lon":  float(self._dr2_lon.get()),
                  "elev": float(self._dr2_elev.get())}
        self._dr2_meas_status.configure(text="Fetching ephemeris for each frame…")
        self.root.update_idletasks()
        try:
            eph = []
            for f in files:
                dt  = parse_ut_from_filename(f)
                row = fetch_horizons_vectors(planet, [dt], site, log=None)[0]
                eph.append(dict(
                    ut=dt,
                    cm=compute_cm_iau2018(planet, row.jd, row.r_vec_au),
                    de=compute_sub_earth_lat(planet, row.jd, row.r_vec_au),
                    pa=compute_pole_position_angle(planet, row.jd, row.r_vec_au)))
        except Exception as e:
            self._dr2_invalidate_meas()
            messagebox.showerror("Measure", f"Ephemeris lookup failed:\n{e}")
            return

        def done(ms):
            self._dr2_meas = ms
            mid = ms[len(ms) // 2]
            self._dr2_meas_status.configure(
                text=f"Measured {len(ms)}/{len(files)} frames  ·  "
                     f"reference tilt {mid.tilt_deg:+.2f}°, "
                     f"r_eq {mid.r_eq:.1f} px")

        MeasurementWindow(self.root, files, planet, site, eph, on_done=done)

    def _dr2_run(self):
        if self._dr2_running:
            return
        if len(self._dr2_files) < 2:
            messagebox.showerror("De-Rotation",
                                 "Add at least 2 image files to de-rotate.")
            return

        try:
            from pvol_derotate import run_derotation
            from pathlib import Path as _Path
        except ImportError as e:
            messagebox.showerror("De-Rotation",
                                 f"pvol_derotate.py not found:\n{e}")
            return

        try:
            site = {"lat":  float(self._dr2_lat.get()),
                    "lon":  float(self._dr2_lon.get()),
                    "elev": float(self._dr2_elev.get())}
        except ValueError:
            messagebox.showerror("De-Rotation",
                                 "Observer lat/lon/elev must be numeric.")
            return

        planet  = self._dr2_planet.get()
        out_str = self._dr2_out_dir.get().strip()
        out_dir = _Path(out_str) if out_str else None
        files   = [_Path(p) for p in self._dr2_files]

        # De-rotation LD (edge cleanup), persisted for next time
        try:
            ld_value = float(self._dr2_ld_value.get())
            ld_angle = float(self._dr2_ld_angle.get())
        except ValueError:
            ld_value, ld_angle = 1.0, 65.0
        save_prefs({"derotate_ld": {"value": ld_value, "angle": ld_angle}})

        self._dr2_running = True
        self._dr2_stop    = False
        self._dr2_run_btn.configure(state="disabled")
        self._dr2_cancel_btn.configure(state="normal")
        self._dr2_prog_var.set(0.0)
        self._dr2_log.config(state="normal")
        self._dr2_log.delete("1.0", "end")
        self._dr2_log.config(state="disabled")
        self._dr2_status.configure(
            text=f"Starting — {len(files)} files  planet={planet}")

        def _worker():
            try:
                run_derotation(
                    input_files=files,
                    output_dir=out_dir,
                    planet=planet,
                    site=site,
                    measurements=self._dr2_meas,
                    log_callback=self._dr2_log_msg,
                    ld_value=ld_value, ld_angle=ld_angle)
                self.root.after(0, lambda: self._dr2_prog_var.set(100.0))
                self.root.after(0, lambda: self._dr2_status.configure(
                    text="Done."))
            except Exception as e:
                import traceback
                self._dr2_log_msg(f"\nError: {e}\n{traceback.format_exc()}")
                self.root.after(0, lambda: self._dr2_status.configure(
                    text=f"Error: {e}"))
            finally:
                self._dr2_running = False
                self._dr2_stop    = False
                self.root.after(0, lambda: self._dr2_run_btn.configure(
                    state="normal"))
                self.root.after(0, lambda: self._dr2_cancel_btn.configure(
                    state="disabled"))

        import threading
        threading.Thread(target=_worker, daemon=True).start()

    def _build_stats_tab(self):
        outer = tk.Frame(self.notebook, bg=BG_PANEL)
        self.notebook.add(outer, text="📊 Stats")
        gf = tk.Frame(outer, bg=BG_PANEL, padx=8, pady=8); gf.pack(fill="x")
        self.stat_vars = {}
        for i,(k,lbl) in enumerate([("width","WIDTH"),("height","HEIGHT"),
                                     ("channels","CHANNELS"),("mean","MEAN L"),
                                     ("std","STD DEV"),("file","FILE")]):
            r,c = divmod(i,3)
            cell = tk.Frame(gf, bg=BG_CARD, padx=8, pady=6)
            cell.grid(row=r, column=c, padx=2, pady=2, sticky="nsew")
            self.stat_vars[k] = tk.StringVar(value="—")
            tk.Label(cell, textvariable=self.stat_vars[k],
                     bg=BG_CARD, fg=ACCENT_C, font=F_BOLD).pack()
            tk.Label(cell, text=lbl, bg=BG_CARD, fg=FG_DIM, font=F_HINT).pack()
        for c in range(3): gf.columnconfigure(c, weight=1)

        tk.Frame(outer, bg=BORDER, height=1).pack(fill="x", padx=8, pady=4)
        tk.Label(outer, text="Per-Channel Statistics",
                 bg=BG_PANEL, fg=FG_MID, font=F_MD).pack(padx=8, anchor="w")
        tbl = tk.Frame(outer, bg=BG_PANEL, padx=8, pady=4); tbl.pack(fill="x")
        for c,h in enumerate(["Channel","Min","Max","Mean","Std Dev"]):
            tk.Label(tbl, text=h, bg=BG_RAISED, fg=FG_MID, font=F_SM,
                     padx=6, pady=3, anchor="center"
                     ).grid(row=0, column=c, sticky="nsew", padx=1, pady=1)
        self.ch_stat_vars = {}
        for r,(ch,col) in enumerate([("Red","#dc2626"),("Green","#15803d"),
                                      ("Blue","#2563eb")], 1):
            tk.Label(tbl, text=ch, bg=BG_CARD, fg=col,
                     font=F_SM, padx=6, pady=3, anchor="w"
                     ).grid(row=r, column=0, sticky="nsew", padx=1, pady=1)
            rv = {}
            for c,key in enumerate(["min","max","mean","std"],1):
                v = tk.StringVar(value="—"); rv[key]=v
                tk.Label(tbl, textvariable=v, bg=BG_CARD, fg=FG_BRIGHT,
                         font=F_SM, padx=6, pady=3, anchor="center"
                         ).grid(row=r, column=c, sticky="nsew", padx=1, pady=1)
            self.ch_stat_vars[ch.lower()] = rv
        for c in range(5): tbl.columnconfigure(c, weight=1)

        tk.Frame(outer, bg=BORDER, height=1).pack(fill="x", padx=8, pady=4)
        tk.Label(outer, text="Histogram  (log scale, 0–65535)",
                 bg=BG_PANEL, fg=FG_MID, font=F_MD).pack(padx=8, anchor="w")
        self.hist_canvas2 = tk.Canvas(outer, height=320, bg=BG_CARD,
                                      highlightthickness=1, highlightbackground=BORDER)
        self.hist_canvas2.pack(fill="both", expand=True, padx=8, pady=(3,10))
        self.hist_canvas2.bind("<Configure>", lambda e: (
            draw_line_histogram(self.hist_canvas2, self.working_arr, e.height)
            if self.working_arr is not None else None))
        self.hist_canvas2.bind("<Configure>", lambda e: 
            self._update_histogram(self.working_arr if self.working_arr is not None
                                   else self.original_arr)
            if self.original_arr is not None else None)

    # ── Image I/O ────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════
    #  Projects — save / open / helpers
    # ══════════════════════════════════════════════════════════════════════

    def _fmt_proj_dir(self):
        d = self._project_dir
        if not d:
            return "📁 No folder set — click 📁 to choose"
        return f"📁 {d}"

    def _project_choose_dir(self):
        import tkinter.filedialog as fd
        d = fd.askdirectory(title="Choose project folder",
                            initialdir=self._project_dir or os.path.expanduser("~"))
        if d:
            self._project_dir = d
            self._proj_dir_lbl.configure(text=self._fmt_proj_dir())

    def _collect_project(self):
        """Return a dict of all current tab settings."""
        import datetime as _dt

        def _slider(s):
            return float(s.get())

        # ── Wavelet ──────────────────────────────────────────────────────
        wv = {
            "convolve":        self.wv_convolve.get(),
            "autosize":        bool(self.wv_autosize.get()),
            "cd_enabled":      bool(self._cd_enabled.get()),
            "cd_radius":       float(self._cd_radius.get()),
            "powerfn_enabled": bool(self.wv_powerfn_enabled.get()),
            "powerfn_exp":     float(self.wv_powerfn_exp.get()),
            "bilateral_radius":float(self.wv_bilateral_radius.get()),
            "zgauss_factor":   float(self.wv_zgauss_factor.get()),
            "pre_smooth":      _slider(self.wv_pre_smooth),
            "layers": [
                {"sharpen": _slider(self.ws1), "denoise": _slider(self.wt1),
                 "sigma":   _slider(self.wsz1), "sharpenfilt": _slider(self.wsf1),
                 "enabled": bool(self.wbl1.get())},
                {"sharpen": _slider(self.ws2), "denoise": _slider(self.wt2),
                 "sigma":   _slider(self.wsz2), "sharpenfilt": _slider(self.wsf2),
                 "enabled": bool(self.wbl2.get())},
                {"sharpen": _slider(self.ws3), "denoise": _slider(self.wt3),
                 "sigma":   _slider(self.wsz3), "sharpenfilt": _slider(self.wsf3),
                 "enabled": bool(self.wbl3.get())},
            ],
        }

        # ── FFT ───────────────────────────────────────────────────────────
        fft = {
            "enabled":    bool(self.fft_enabled.get()),
            "active":     self.fft_active.get(),
            # Per-stage bands. "stages" supersedes the old single-layer format;
            # layer/start/end/curve are still written so presets saved here can
            # be read by older builds.
            "stages":     {s: {"on": bool(self.fft_stage_on[s].get()),
                               "start": float(self.fft_params[s]["start"]),
                               "end":   float(self.fft_params[s]["end"]),
                               "curve": float(self.fft_params[s]["curve"])}
                           for s in self.fft_stage_ids},
            "layer":      self.fft_active.get(),
            "start":      float(self.fft_marker_start),
            "end":        float(self.fft_marker_end),
            "curve":      _slider(self.fft_curve_slider),
        }

        # ── RGB / Tools ───────────────────────────────────────────────────
        rgb = {
            "align_rgb": bool(self._align_rgb.get()) if hasattr(self, "_align_rgb") else False,
            "r_gain":  _slider(self.rgb_r),  "r_black": _slider(self.black_r),
            "r_gamma": _slider(self.gamma_r),
            "g_gain":  _slider(self.rgb_g),  "g_black": _slider(self.black_g),
            "g_gamma": _slider(self.gamma_g),
            "b_gain":  _slider(self.rgb_b),  "b_black": _slider(self.black_b),
            "b_gamma": _slider(self.gamma_b),
            "sat":     _slider(self.saturation),
            "dc_enabled":      bool(self._dc_enabled.get()),
            "dc_strength":     float(self._dc_strength.get()),
            "dc_use_contrast": bool(self._dc_use_contrast.get()),
            "dc_contrast_str": float(self._dc_contrast_str.get()),
            "dh_enabled":   bool(self._dh_enabled.get()),
            "dh_blocksize": float(self._dh_blocksize.get()),
            "dh_amount":    float(self._dh_amount.get()),
            "lc_channel":   self._lc_channel.get(),
            "lc_black_all": float(self._lc_black_all.get()),
            "lc_white_all": float(self._lc_white_all.get()),
            "lc_gamma_all": float(self._lc_gamma_all.get()),
            "lc_black": {c: float(self._lc_black[c].get()) for c in "RGB"},
            "lc_white": {c: float(self._lc_white[c].get()) for c in "RGB"},
            "lc_gamma": {c: float(self._lc_gamma[c].get()) for c in "RGB"},
            "clahe_enabled":  bool(self._clahe_enabled.get()),
            "clahe_clip":     float(self._clahe_clip.get()),
            "clahe_tile":     float(self._clahe_tile.get()),
            "clahe_strength": float(self._clahe_strength.get()),
            "clahe_channel":  self._clahe_channel.get(),
        }

        # ── De-rind ───────────────────────────────────────────────────────
        dr = {
            "enabled":      bool(self.dr_enabled.get()),
            "edge":         _slider(self.dr_edge),
            "smooth":       _slider(self.dr_smooth),
            "inset":        _slider(self.dr_inset),
            "gap_width":    float(self.dr_gap_width.get()),
            "gap_angle":    float(self.dr_gap_angle.get()),
            "saturn":       bool(self.dr_saturn.get()),
            "dark_edge":    bool(self.dr_dark_edge.get()),
            "pre_blur":     _slider(self.dr_pre_blur),
        }

        # ── Moon Recovery ────────────────────────────────────────────────
        mr = {
            "enabled":      bool(self.mr_enabled.get()),
            "circles":      list(self._mr_circles) if hasattr(self, "_mr_circles") else [],
            "boost":        float(self.mr_boost.get()),
            "feather":      float(self.mr_feather.get()),
            "saturation":   float(self.mr_saturation.get()) if hasattr(self, "mr_saturation") else 10,
            "darken_edge":  float(self.mr_darken_edge.get()) if hasattr(self, "mr_darken_edge") else 0,
            "preview_boost": float(self.mr_preview_boost.get()) if hasattr(self, "mr_preview_boost") else 10,
        }

        return {
            "kepler_project_version": 1,
            "rotate_deg": float(self._preview_rot_deg.get()),
            "object":   self._proj_object.get().strip(),
            "date":     self._proj_date.get().strip(),
            "wavelet":  wv,
            "fft":      fft,
            "rgb":      rgb,
            "derind":   dr,
            "moon_recovery": mr,
        }

    def _apply_project(self, data):
        """Apply a loaded project dict to all controls, then re-run pipeline."""
        self._loading_project = True
        def _s(slider, val):
            try: slider.set(float(val), fire_callback=False)
            except Exception: pass

        wv = data.get("wavelet", {})
        try: self.wv_convolve.set(wv.get("convolve", "LRGB  (L first, then RGB)"))
        except Exception: pass
        try: self.wv_autosize.set(wv.get("autosize", False))
        except Exception: pass
        try: self._cd_enabled.set(wv.get("cd_enabled", False))
        except Exception: pass
        try: self._cd_radius.set(wv.get("cd_radius", 1.0))
        except Exception: pass
        try: self.wv_powerfn_enabled.set(wv.get("powerfn_enabled", False))
        except Exception: pass
        try: self.wv_powerfn_exp.set(wv.get("powerfn_exp", 1.0))
        except Exception: pass
        try: self.wv_bilateral_radius.set(wv.get("bilateral_radius", 2.0))
        except Exception: pass
        try: self.wv_zgauss_factor.set(wv.get("zgauss_factor", 1.0))
        except Exception: pass
        _s(self.wv_pre_smooth, wv.get("pre_smooth", 0))
        for i, (ws, wt, wsz, wsf, wbl) in enumerate([
            (self.ws1, self.wt1, self.wsz1, self.wsf1, self.wbl1),
            (self.ws2, self.wt2, self.wsz2, self.wsf2, self.wbl2),
            (self.ws3, self.wt3, self.wsz3, self.wsf3, self.wbl3),
        ]):
            if i < len(wv.get("layers", [])):
                L = wv["layers"][i]
                _s(ws, L.get("sharpen", 0)); _s(wt, L.get("denoise", 0))
                _s(wsz, L.get("sigma", 1)); _s(wsf, L.get("sharpenfilt", 0))
                try: wbl.set(L.get("enabled", True))
                except Exception: pass

        fft = data.get("fft", {})
        try: self.fft_enabled.set(fft.get("enabled", False))
        except Exception: pass
        try:
            stages = fft.get("stages")
            if stages:
                for s in self.fft_stage_ids:
                    st = stages.get(s, {})
                    self.fft_stage_on[s].set(bool(st.get("on", False)))
                    self.fft_params[s] = dict(
                        start=float(st.get("start", 60.0)),
                        end=float(st.get("end", 90.0)),
                        curve=float(st.get("curve", 25.0)))
                self.fft_active.set(fft.get("active", "POST"))
            else:
                # Pre-multi-stage preset: one layer with one band.
                layer = fft.get("layer", "POST")
                if layer not in self.fft_stage_ids:
                    layer = "POST"
                for s in self.fft_stage_ids:
                    self.fft_stage_on[s].set(s == layer)
                self.fft_params[layer] = dict(
                    start=float(fft.get("start", 60.0)),
                    end=float(fft.get("end", 90.0)),
                    curve=float(fft.get("curve", 25.0)))
                self.fft_active.set(layer)
        except Exception: pass
        try: self._fft_set_active()
        except Exception: pass

        rgb = data.get("rgb", {})
        try: self._align_rgb.set(bool(rgb.get("align_rgb", False)))
        except Exception: pass
        _s(self.rgb_r,   rgb.get("r_gain", 100));  _s(self.black_r, rgb.get("r_black", 0))
        _s(self.gamma_r, rgb.get("r_gamma", 100))
        _s(self.rgb_g,   rgb.get("g_gain", 100));  _s(self.black_g, rgb.get("g_black", 0))
        _s(self.gamma_g, rgb.get("g_gamma", 100))
        _s(self.rgb_b,   rgb.get("b_gain", 100));  _s(self.black_b, rgb.get("b_black", 0))
        _s(self.gamma_b, rgb.get("b_gamma", 100))
        _s(self.saturation, rgb.get("sat", 100))
        try: self._dc_enabled.set(rgb.get("dc_enabled", False))
        except Exception: pass
        try: self._dc_strength.set(rgb.get("dc_strength", 0))
        except Exception: pass
        try: self._dc_use_contrast.set(rgb.get("dc_use_contrast", False))
        except Exception: pass
        try: self._dc_contrast_str.set(rgb.get("dc_contrast_str", 0))
        except Exception: pass
        try: self._dh_enabled.set(rgb.get("dh_enabled", False))
        except Exception: pass
        try: self._dh_blocksize.set(rgb.get("dh_blocksize", 5))
        except Exception: pass
        try: self._dh_amount.set(rgb.get("dh_amount", 0.5))
        except Exception: pass
        try: self._lc_channel.set(rgb.get("lc_channel", "All"))
        except Exception: pass
        try: self._lc_black_all.set(rgb.get("lc_black_all", 0))
        except Exception: pass
        try: self._lc_white_all.set(rgb.get("lc_white_all", 1))
        except Exception: pass
        try: self._lc_gamma_all.set(rgb.get("lc_gamma_all", 1))
        except Exception: pass
        for c in "RGB":
            try: self._lc_black[c].set(rgb.get("lc_black", {}).get(c, 0))
            except Exception: pass
            try: self._lc_white[c].set(rgb.get("lc_white", {}).get(c, 1))
            except Exception: pass
            try: self._lc_gamma[c].set(rgb.get("lc_gamma", {}).get(c, 1))
            except Exception: pass
        try: self._clahe_enabled.set(rgb.get("clahe_enabled", False))
        except Exception: pass
        try: self._clahe_clip.set(rgb.get("clahe_clip", 0.5))
        except Exception: pass
        try: self._clahe_tile.set(rgb.get("clahe_tile", 4))
        except Exception: pass
        try: self._clahe_strength.set(rgb.get("clahe_strength", 0.05))
        except Exception: pass
        try: self._clahe_channel.set(rgb.get("clahe_channel", "luminance"))
        except Exception: pass

        dr = data.get("derind", {})
        try: self.dr_enabled.set(dr.get("enabled", False))
        except Exception: pass
        _s(self.dr_edge,   dr.get("edge", 20))
        _s(self.dr_smooth, dr.get("smooth", 2))
        _s(self.dr_inset,  dr.get("inset", 0))
        try: self.dr_gap_width.set(dr.get("gap_width", 0))
        except Exception: pass
        try: self.dr_gap_angle.set(dr.get("gap_angle", 0))
        except Exception: pass
        try: self.dr_saturn.set(dr.get("saturn", False))
        except Exception: pass
        try: self.dr_dark_edge.set(dr.get("dark_edge", True))
        except Exception: pass
        _s(self.dr_pre_blur, dr.get("pre_blur", 0))

        # ── Moon Recovery ──
        mr = data.get("moon_recovery", {})
        try: self.mr_enabled.set(mr.get("enabled", False))
        except Exception: pass
        _s(self.mr_boost,          mr.get("boost", 25))
        _s(self.mr_feather,        mr.get("feather", 5))
        if hasattr(self, "mr_saturation"): _s(self.mr_saturation, mr.get("saturation", 10))
        if hasattr(self, "mr_darken_edge"): _s(self.mr_darken_edge, mr.get("darken_edge", 0))
        _s(self.mr_preview_boost,  mr.get("preview_boost", 10))
        try:
            self._mr_circles = [tuple(c) for c in mr.get("circles", [])]
        except Exception:
            self._mr_circles = []
        self._mr_update_overlays()

        try: self._preview_rot_deg.set(float(data.get("rotate_deg", 0.0))); self._on_preview_rotate()
        except Exception: pass

        self._loading_project = False
        # Trigger full pipeline rerun
        self._schedule_live(self._current_tab_index)

    def _project_save(self):
        import json, tkinter.messagebox as mb
        obj  = self._proj_object.get().strip()
        date = self._proj_date.get().strip()
        if not obj:
            mb.showwarning("Save Project", "Please enter an object name first.")
            return
        filename = f"{obj}_{date}.kproj".replace(" ", "_")
        if self._project_dir and os.path.isdir(self._project_dir):
            path = os.path.join(self._project_dir, filename)
        else:
            import tkinter.filedialog as fd
            path = fd.asksaveasfilename(
                title="Save Project",
                initialfile=filename,
                defaultextension=".kproj",
                filetypes=[("Kepler Project", "*.kproj"), ("JSON", "*.json"), ("All", "*.*")],
            )
            if not path: return
            self._project_dir = os.path.dirname(path)
            self._proj_dir_lbl.configure(text=self._fmt_proj_dir())

        try:
            data = self._collect_project()
        except Exception as e:
            mb.showerror("Save Project", f"Error collecting settings:\n{e}")
            return
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            mb.showinfo("Save Project", f"Saved:\n{os.path.basename(path)}")
        except Exception as e:
            mb.showerror("Save Project", f"Could not write file:\n{e}")

    def _project_open(self):
        import json, tkinter.filedialog as fd, tkinter.messagebox as mb
        init = self._project_dir if self._project_dir and os.path.isdir(self._project_dir)                else os.path.expanduser("~")
        path = fd.askopenfilename(
            title="Open Project",
            initialdir=init,
            filetypes=[("Kepler Project", "*.kproj"), ("JSON", "*.json"), ("All", "*.*")],
        )
        if not path: return
        self._project_dir = os.path.dirname(path)
        self._proj_dir_lbl.configure(text=self._fmt_proj_dir())
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            mb.showerror("Open Project", f"Could not read file:\n{e}")
            return
        # Restore object/date fields from file
        self._proj_object.set(data.get("object", ""))
        self._proj_date.set(data.get("date", ""))
        self._apply_project(data)

    def open_image(self):
        _init = self._image_dir if self._image_dir and os.path.isdir(self._image_dir) \
                else os.path.expanduser("~")
        path = filedialog.askopenfilename(
            title="Open Planetary Image",
            initialdir=_init,
            filetypes=[("Image files","*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.fit *.fits"),
                       ("All files","*.*")])
        if not path: return
        self._image_dir = os.path.dirname(path)
        try:
            # Use tifffile for TIFs to preserve 16-bit depth; fall back to PIL for others
            ext = os.path.splitext(path)[1].lower()
            if ext in (".tif", ".tiff"):
                try:
                    import tifffile
                    raw = tifffile.imread(path)
                    if raw.ndim == 2:                     # grayscale → RGB
                        raw = np.stack([raw]*3, axis=-1)
                    elif raw.shape[-1] > 3:               # RGBA → RGB
                        raw = raw[..., :3]
                    if raw.dtype == np.uint16:
                        arr = raw.astype(np.float32) / 65535.0
                    elif raw.dtype == np.uint8:
                        arr = raw.astype(np.float32) / 255.0
                    else:
                        arr = raw.astype(np.float32)
                        mn, mx = arr.min(), arr.max()
                        if mx > mn: arr = (arr - mn) / (mx - mn)
                    self.original_arr = arr
                    # Build display PIL from float32 (8-bit for canvas rendering)
                    pil = proc.array_to_pil(arr)
                except ImportError:
                    pil = Image.open(path).convert("RGB")
                    self.original_arr = proc.pil_to_array(pil)
            elif ext == ".png":
                self.original_arr = proc.load_png(path)
                pil = proc.array_to_pil(self.original_arr)

            self.original_pil = pil; self.working_pil = pil.copy()
            self.working_arr  = self.original_arr.copy()
            self._source_path = path
            # Reset preview rotation when a new image is loaded
            try:
                self._preview_rot_deg.set(0.0)
            except AttributeError:
                pass
            # Refresh curve canvas histogram with new image data
            try:
                self._draw_curve_canvas()
            except AttributeError:
                pass
            self._update_stats(path)
            self._update_fft_spectrum(self.original_arr)
            self.status_var.set("IMAGE LOADED")
            self.img_size_var.set(f"{pil.width}×{pil.height}  ·  {os.path.basename(path)}")
            # Defer canvas draws until layout is complete so winfo_width/height
            # return real pixel sizes for correct Fit scaling.
            def _draw_after_layout():
                self.root.update_idletasks()
                self._draw_on_canvas(self.cnv_orig, pil)
                self._draw_on_canvas(self.cnv_proc, pil)
                self._update_histogram(self.original_arr)
            self.root.after(30, _draw_after_layout)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")

    def export_image(self):
        if self.working_pil is None:
            messagebox.showwarning("No Image","No processed image to export."); return

        # Determine defaults from source file
        if self._source_path:
            base, src_ext = os.path.splitext(os.path.basename(self._source_path))
            src_ext  = src_ext.lower()
            init_dir = os.path.dirname(self._source_path)
            default_ext = src_ext if src_ext in (".tif",".tiff",".png",".jpg",".jpeg",".bmp") else ".tif"
        else:
            base        = "export_KEP"
            default_ext = ".tif"
            init_dir    = os.path.expanduser("~")

        # ── Format picker ─────────────────────────────────────────────────────
        # The Linux GTK file dialog ignores the filetypes filter — the selected
        # filter has no effect on the returned filename. We show a small format
        # selector first so the user explicitly picks the output format, then
        # open the save dialog with that extension already in the filename.
        _fmt_opts = [
            ("TIFF 16-bit  (.tif)",  ".tif"),
            ("PNG 16-bit  (.png)",   ".png"),
            ("JPEG  (.jpg)",          ".jpg"),
            ("BMP  (.bmp)",           ".bmp"),
        ]
        picker = tk.Toplevel(self.root)
        picker.withdraw()  # hide until positioned to avoid flash at (0,0)
        picker.title("Export Format")
        picker.resizable(False, False)
        picker.configure(bg=BG_PANEL)
        tk.Label(picker, text="Save as format:",
                 bg=BG_PANEL, fg=FG_BRIGHT, font=F_SM).pack(padx=20, pady=(16,6))
        fmt_var = tk.StringVar(value=default_ext)
        for lbl, ext in _fmt_opts:
            tk.Radiobutton(picker, text=lbl, variable=fmt_var, value=ext,
                           bg=BG_PANEL, fg=FG_BRIGHT, selectcolor=BG_RAISED,
                           activebackground=BG_PANEL, font=F_SM,
                           anchor="w").pack(fill="x", padx=24, pady=2)
        btn_row = tk.Frame(picker, bg=BG_PANEL)
        btn_row.pack(pady=(10,16), padx=20, fill="x")
        chosen = [None]
        def _ok():
            chosen[0] = fmt_var.get()
            picker.destroy()
        def _cancel():
            picker.destroy()
        tk.Button(btn_row, text="Cancel", command=_cancel,
                  bg=BTN_BG, fg=FG_MID, font=F_SM, relief="groove", bd=2,
                  padx=10, pady=4, cursor="hand2",
                  activebackground=BTN_ACTIVE).pack(side="right", padx=(4,0))
        tk.Button(btn_row, text="  OK  ", command=_ok,
                  bg=ACCENT_G, fg="white", font=F_SM, relief="flat",
                  padx=10, pady=4, cursor="hand2",
                  activebackground=BTN_ACTIVE).pack(side="right")
        # Let tkinter size the window to its contents, then center it.
        # A fixed geometry clips the OK button on Windows due to larger DPI/font rendering.
        picker.update_idletasks()
        pw = picker.winfo_reqwidth()
        ph = picker.winfo_reqheight()
        px = self.root.winfo_x() + (self.root.winfo_width()  - pw) // 2
        py = self.root.winfo_y() + (self.root.winfo_height() - ph) // 2
        picker.geometry(f"+{px}+{py}")
        picker.grab_set()
        picker.deiconify()  # now show at correct position
        picker.wait_window()
        if chosen[0] is None:
            return
        default_ext  = chosen[0]

        # ── File save dialog ──────────────────────────────────────────────────
        # No filetypes filter — format is already decided by the picker above.
        default_name = base + "_KEP" + default_ext
        path = filedialog.asksaveasfilename(
            title="Export Processed Image",
            initialdir=init_dir,
            initialfile=default_name,
            defaultextension=default_ext)
        if not path: return
        # Resolve extension and PIL format explicitly — don't let PIL guess.
        # On Linux GTK, the dialog may return a path with no extension.
        _ext_to_fmt = {
            ".tif": "TIFF", ".tiff": "TIFF",
            ".png": "PNG",
            ".jpg": "JPEG", ".jpeg": "JPEG",
            ".bmp": "BMP",
        }
        _root, _ext = os.path.splitext(path)
        _ext = _ext.lower()
        if _ext not in _ext_to_fmt:
            # No extension or unrecognized — append the default
            _ext = default_ext.lower()
            path = (_root if _root else path) + _ext
        pil_fmt = _ext_to_fmt[_ext]
        try:
            try:
                rot_deg = float(self._preview_rot_deg.get())
            except (ValueError, AttributeError, tk.TclError):
                rot_deg = 0.0
            if _ext in (".tif", ".tiff", ".png") and self.working_arr is not None:
                # Export as 16-bit preserving full dynamic range (TIFF and PNG).
                import tifffile as _tifffile
                from scipy.ndimage import rotate as _ndrotate
                arr_out = self.working_arr
                if rot_deg != 0.0:
                    arr_out = _ndrotate(arr_out, -rot_deg,
                                        axes=(1, 0), reshape=False,
                                        order=3, cval=0.0)
                arr16 = np.clip(arr_out * 65535.0, 0, 65535).astype(np.uint16)
                _tifffile.imwrite(path, arr16)
            else:
                img = self.working_pil if self.working_arr is None                       else proc.array_to_pil(self.working_arr)
                if rot_deg != 0.0:
                    img = img.rotate(-rot_deg, expand=False,
                                     resample=Image.BICUBIC,
                                     fillcolor=(0, 0, 0))
                # Pass format= explicitly so PIL never has to guess from extension
                kw = {"quality": 95} if pil_fmt == "JPEG" else {}
                img.save(path, format=pil_fmt, **kw)
            rot_note = f"  (rotated {rot_deg:+.1f}°)" if rot_deg != 0.0 else ""
            self.status_var.set("EXPORTED")
            messagebox.showinfo("Exported", f"Image saved to:\n{path}{rot_note}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
        """Force a full pipeline re-run with current slider values.
        Cancels any pending debounce, resets processing state if stuck,
        then fires immediately."""
        if self.original_arr is None: return
        # Cancel any queued debounce timer
        if self._live_after is not None:
            self.root.after_cancel(self._live_after)
            self._live_after = None
        # If the processing flag is stuck (e.g. thread died silently), clear it
        if self.processing:
            self.processing = False
            self.progress_var.set(0)
        self._fire_live()

    def _fft_auto_denoise(self):
        """
        Estimate best-guess FFT denoise band from the image's power spectrum.

        Strategy:
          - Compute the smoothed radial power spectrum of the processed image
          - Find where the drop rate (first derivative) slows to <10% of the
            initial drop rate — this marks where signal energy gives way to
            noise/sharpening artifacts
          - Set start = that point (clamped to 15-45% Nyquist)
          - Set end = 70% (fixed — always roll off well past the noise floor)
          - Curve chosen based on the flatness of the noise plateau
        """
        src = self.working_arr if self.working_arr is not None else self.original_arr
        if src is None:
            self.fft_auto_lbl.configure(text="Load an image first")
            return

        import numpy as np
        from scipy.ndimage import gaussian_filter1d

        sp = proc.compute_fft_power(src)   # 64-bin normalized power spectrum
        n  = len(sp)

        sp_s = gaussian_filter1d(sp.astype(np.float64), sigma=2.0)
        d1 = np.diff(sp_s)

        # Initial drop rate: mean slope of the first 8 bins (signal-dominated)
        initial_rate = float(np.mean(d1[:8]))   # negative value

        # Find first bin (past bin 4) where slope slows to <10% of initial rate
        # i.e., the drop essentially stops — signal has given way to noise floor
        threshold = initial_rate * 0.10         # 10% of initial rate (less negative)
        slow_bins = np.where(d1[4:] > threshold)[0]
        if len(slow_bins) > 0:
            cross_bin = slow_bins[0] + 4
        else:
            cross_bin = int(n * 0.30)

        start_pct = float(np.clip(cross_bin / n * 100.0, 15.0, 45.0))
        end_pct   = 70.0   # always roll off to 70% Nyquist

        # Noise severity: how flat is the plateau from 30-70%?
        # A flat spectrum (ratio hi/lo near 1) = mild noise
        # A rising spectrum = heavy sharpening artifacts
        plateau = sp_s[int(n*0.30) : int(n*0.70)]
        if len(plateau) > 1:
            plateau_ratio = float(plateau.max() / (plateau.min() + 1e-8))
        else:
            plateau_ratio = 1.5

        if plateau_ratio > 1.3:
            curve = 85.0
            desc  = "heavy noise"
        elif plateau_ratio > 1.1:
            curve = 65.0
            desc  = "moderate noise"
        else:
            curve = 50.0
            desc  = "mild noise"

        # Apply to the stage currently being edited, and switch it on.
        act = self.fft_active.get()
        self.fft_params[act] = dict(start=round(start_pct, 1),
                                    end=round(end_pct, 1),
                                    curve=float(curve))
        self.fft_stage_on[act].set(True)
        self.fft_enabled.set(True)
        self.fft_disabled_lbl.pack_forget()
        self.fft_auto_lbl.configure(
            text=f"{self.fft_stage_names[act]}:  start={start_pct:.0f}%  "
                 f"end={end_pct:.0f}%  {desc}")
        self._fft_set_active()
        self._schedule_live(1)

    def _fft_reset(self):
        """Reset all FFT controls — every stage — to defaults."""
        for s in self.fft_stage_ids:
            self.fft_params[s] = dict(start=60.0, end=90.0, curve=25.0)
            self.fft_stage_on[s].set(s == "POST")
        self.fft_active.set("POST")
        self.fft_enabled.set(False)
        self.fft_disabled_lbl.pack(side="left", padx=8)
        try: self.fft_auto_lbl.configure(text="")
        except Exception: pass
        self._fft_set_active()
        self._schedule_live(1)

    def reset_tab_sliders(self):
        """Reset all values (sliders + spinboxes + checkboxes) in active tab."""
        idx = self._current_tab_index
        for s in self._tab_sliders.get(idx, []):
            s.set(s._default, fire_callback=False)
        if idx == 0:   self._reset_wavelet_extras()
        elif idx == 1: self._fft_reset()
        elif idx == 2: self._reset_rgb_extras()
        elif idx == 3: self._reset_tools_extras()
        elif idx == 4: self._derind_reset()
        elif idx == 5: self._orbital_reset()
        self.status_var.set("TAB VALUES RESET")
        self._pending_tab = idx
        self._fire_live()

    def _reset_wavelet_extras(self):
        """Reset Wavelet tab non-slider controls to defaults."""
        self._loading_project = True
        try:
            try: self.wv_convolve.set("LRGB  (L first, then RGB)")
            except Exception: pass
            try: self.wv_filter.set("gaussian")
            except Exception: pass
            try: self.wv_color_model.set("oklab (default)")
            except Exception: pass
            try: self.wv_autosize.set(False)
            except Exception: pass
            try: self._cd_enabled.set(False); self._cd_radius.set(1.0)
            except Exception: pass
            try: self.wv_powerfn_enabled.set(False); self.wv_powerfn_exp.set(1.0)
            except Exception: pass
            try: self.wv_bilateral_radius.set(2.0)
            except Exception: pass
            try: self.wv_zgauss_factor.set(1.0)
            except Exception: pass
            try: self.wbl1.set(True); self.wbl2.set(True); self.wbl3.set(True)
            except Exception: pass
        finally:
            self._loading_project = False

    def _reset_rgb_extras(self):
        """Reset RGB tab non-slider controls to defaults."""
        try: self._align_rgb.set(False)
        except Exception: pass

    def _reset_tools_extras(self):
        """Reset Tools tab non-slider controls to defaults."""
        # Use _loading_project flag to suppress trace callbacks while resetting
        self._loading_project = True
        try:
            try: self._dc_enabled.set(False); self._dc_strength.set(0.0)
            except Exception: pass
            try: self._dc_use_contrast.set(False); self._dc_contrast_str.set(0.0)
            except Exception: pass
            try: self._dh_enabled.set(False); self._dh_blocksize.set(5.0); self._dh_amount.set(0.5)
            except Exception: pass
            try:
                self._lc_channel.set("All")
                self._lc_black_all.set(0.0); self._lc_white_all.set(1.0); self._lc_gamma_all.set(1.0)
                for ch in "RGB":
                    self._lc_black[ch].set(0.0); self._lc_white[ch].set(1.0); self._lc_gamma[ch].set(1.0)
                for ch in ["All","R","G","B"]:
                    self._curve_pts[ch] = [[0.0,0.0],[1.0,1.0]]
                self._draw_curve_canvas()
            except Exception: pass
            try: self._clahe_enabled.set(False); self._clahe_channel.set("luminance")
            except Exception: pass
            try: self._clahe_clip.set(0.5); self._clahe_tile.set(4.0); self._clahe_strength.set(0.05)
            except Exception: pass
        finally:
            self._loading_project = False

    def _derind_reset(self):
        """Reset all De-rind controls to defaults."""
        for s in self._tab_sliders.get(4, []):
            s.set(s._default, fire_callback=False)
        self.dr_gap_width.set(0.0)
        self.dr_gap_angle.set(0.0)
        self.dr_saturn.set(False)
        self.dr_dark_edge.set(True)
        self.dr_show_map.set(False)
        self.dr_enabled.set(False)
        self.dr_auto_lbl.configure(text="")
        self._pending_tab = 3
        self._fire_live()

    def _orbital_reset(self):
        """Reset all Orbital (Moon Recovery) controls to defaults."""
        for s in self._tab_sliders.get(5, []):
            s.set(s._default, fire_callback=False)
        self.mr_enabled.set(False)
        if hasattr(self, "_mr_circles"):
            self._mr_circles.clear()
            self._mr_overlays_hidden = False
            self._mr_update_overlays()
        if hasattr(self, "mr_preview_boost"):
            self.mr_preview_boost.set(10, fire_callback=False)
        if getattr(self, "_mr_mode", False):
            self._mr_toggle_draw_mode()
        self._pending_tab = 4
        self._fire_live()

    def reset_all(self):
        """Reset ALL values (sliders + spinboxes + checkboxes) and reprocess."""
        for tab_idx, sliders in self._tab_sliders.items():
            for s in sliders:
                s.set(s._default, fire_callback=False)
        self._reset_wavelet_extras()
        self._fft_reset()
        self._reset_rgb_extras()
        self._reset_tools_extras()
        self._derind_reset()
        self._orbital_reset()
        # Rerun the full pipeline to reflect all reset values
        if self.original_arr is not None:
            self._pending_tab = 0
            self._fire_live()
        self.status_var.set("FULL RESET")

    def _update_stats(self, path=""):
        if self.original_arr is None: return
        arr = self.original_arr; h, w = arr.shape[:2]
        lum = proc.luminance(arr)
        self.stat_vars["width"].set(str(w)); self.stat_vars["height"].set(str(h))
        self.stat_vars["channels"].set("RGB")
        self.stat_vars["mean"].set(f"{lum.mean()*255:.1f}")
        self.stat_vars["std"].set(f"{lum.std()*255:.1f}")
        self.stat_vars["file"].set(os.path.basename(path)[:16] if path else "—")
        for i, ch in enumerate(["red","green","blue"]):
            d = arr[...,i]
            self.ch_stat_vars[ch]["min"].set(f"{d.min()*255:.1f}")
            self.ch_stat_vars[ch]["max"].set(f"{d.max()*255:.1f}")
            self.ch_stat_vars[ch]["mean"].set(f"{d.mean()*255:.1f}")
            self.ch_stat_vars[ch]["std"].set(f"{d.std()*255:.1f}")

    def _update_histogram(self, arr):
        for canvas in [self.hist_canvas, self.hist_canvas2]:
            h = canvas.winfo_height() or int(canvas.cget("height"))
            draw_line_histogram(canvas, arr, h)

    def _update_fft_spectrum(self, arr):
        def _go():
            spec = proc.compute_fft_power(arr)
            self._fft_spectrum = spec
            self.root.after(0, self._draw_fft_graph)
        threading.Thread(target=_go, daemon=True).start()

    # ── Processing ───────────────────────────────────────────
    def _get_wavelet_params(self):
        return [
            (int(self.ws1.get()), int(self.wt1.get()), float(self.wsz1.get())),
            (int(self.ws2.get()), int(self.wt2.get()), float(self.wsz2.get())),
            (int(self.ws3.get()), int(self.wt3.get()), float(self.wsz3.get())),
        ]

    def _do_wavelet(self, arr):
        fm = self.wv_filter.get().split()[0]
        br = getattr(self, "wv_bilateral_radius", None)
        bilateral_radius = float(br.get()) if br is not None else 2.0
        zf = getattr(self, "wv_zgauss_factor", None)
        zgauss_factor = float(zf.get()) if zf is not None else 1.0
        bilateral_layers = [
            self.wbl1.get(),
            self.wbl2.get(),
            self.wbl3.get(),
        ] if fm == "bilateral" else None
        return proc.wavelet_sharpen(
            arr, self._get_wavelet_params(), fm,
            bilateral_radius=bilateral_radius,
            zgauss_factor=zgauss_factor,
            bilateral_layers=bilateral_layers)

    def _do_fft(self, arr):
        """Apply the POST-All Layers FFT stage (PRE stages run in the wavelet)."""
        _, post = self._fft_collect()
        if post is None:
            return arr
        return proc.fft_denoise(arr, fft_start=post[0],
                                fft_width=post[1], fft_curve=post[2])

    def _do_color(self, arr):
        return proc.apply_color_adjustments(
            arr,
            r_gain=self.rgb_r.get()/100,    g_gain=self.rgb_g.get()/100,
            b_gain=self.rgb_b.get()/100,    r_gamma=self.gamma_r.get()/100,
            g_gamma=self.gamma_g.get()/100, b_gamma=self.gamma_b.get()/100,
            saturation=self.saturation.get()/100,
            vibrance=self.vibrance.get()/100,
            hue_rotation=self.hue_rot.get(),
            brightness=self.brightness.get()/255,
            contrast=self.contrast.get()/100)

    def _do_dering(self, arr):
        return proc.apply_derind(
            arr,
            edge=self.dr_edge.get(),
            smooth=self.dr_smooth.get(),
            gap_width=self.dr_gap_width.get(),
            gap_angle=self.dr_gap_angle.get(),
            saturn_mode=self.dr_saturn.get(),
            dark_edge=self.dr_dark_edge.get(),
            pre_blur=self.dr_pre_blur.get(),
            show_ring_map=self.dr_show_map.get())

    def _run_in_thread(self, fn, label="PROCESSING"):
        if self.original_arr is None: return
        if self.processing: return
        self.processing = True
        self.status_var.set(label + "…")
        self.progress_var.set(10)
        def worker():
            try:
                result = fn()
                self.working_arr = result
                self.root.after(0, self._finish_processing)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Processing Error",str(e)))
                self.root.after(0, self._reset_processing_state)
        threading.Thread(target=worker, daemon=True).start()

    def _finish_processing(self):
        if self.working_arr is not None and not getattr(self, "_is_panning", False):
            proc_pil = self._get_processed_pil()
            self._draw_on_canvas(self.cnv_proc, proc_pil)
            # Redraw circle overlays on top — _draw_on_canvas clears the canvas
            if hasattr(self, "_mr_circles") and not getattr(self, "_mr_suppress_redraw", False):
                self._mr_update_overlays()
            # Refresh magnifier if it has an active PROC view
            if self.magnifier._last_pil is not None and self.magnifier._last_name == "PROC":
                self.magnifier.click_magnify(
                    proc_pil,
                    self.magnifier._last_px,
                    self.magnifier._last_py,
                    "PROC")
        self._update_histogram(self.working_arr if self.working_arr is not None
                               else self.original_arr)
        self._update_fft_spectrum(self.working_arr if self.working_arr is not None
                                  else self.original_arr)
        self.progress_var.set(100); self.status_var.set("DONE")
        self.processing = False
        self.root.after(1500, lambda: self.progress_var.set(0))
        # If a slider moved while we were processing, fire immediately now
        if self._live_pending:
            self._live_pending = False
            self.root.after(0, self._fire_live)

    def _reset_processing_state(self):
        self.processing = False; self.status_var.set("ERROR"); self.progress_var.set(0)

    def run_estimate_sharpen(self):
        """Estimate optimal filter sigma per level from current sharpen values.
        Activates Autosize and updates sigma labels."""
        if self.original_arr is None:
            messagebox.showwarning("No Image", "Load an image first."); return
        self._run_autosize()
        self.wv_autosize.set(True)
        self._wv_sigmas = [float(self.wsz1.get()), float(self.wsz2.get()), float(self.wsz3.get())]
        self.estimate_lbl.configure(
            text=f"Decomp: σ1={self._wv_sigmas[0]:.1f}px  "
                 f"σ2={self._wv_sigmas[1]:.1f}px  "
                 f"σ3={self._wv_sigmas[2]:.1f}px")
        if self._live_after is not None:
            self.root.after_cancel(self._live_after)
            self._live_after = None
        self._pending_tab = 0
        self._fire_live()

    def auto_balance(self):
        """Full levels balance: sets R/G/B gain AND black point sliders so
        both shadow and highlight edges of the histogram align. Always computed
        from the original image so repeated clicks give the same stable result."""
        if self.original_arr is None:
            messagebox.showwarning("No Image","Load an image first."); return
        rg, gg, bg_, rbp, gbp, bbp = proc.auto_balance(self.original_arr)
        for slider, val in [
            (self.rgb_r,   rg  * 100),
            (self.rgb_g,   gg  * 100),
            (self.rgb_b,   bg_ * 100),
            (self.black_r, rbp * 100),
            (self.black_g, gbp * 100),
            (self.black_b, bbp * 100),
        ]:
            slider.set(val, fire_callback=False)
        self.status_var.set("AUTO RGB BALANCED")
        self._pending_tab = 2
        self._fire_live()

    def auto_white_balance(self):
        """Percentile stretch: sets R/G/B gain AND gamma sliders. Always
        computed from the original image so repeated clicks are idempotent."""
        if self.original_arr is None:
            messagebox.showwarning("No Image","Load an image first."); return
        gains, gammas = proc.auto_white_balance(self.original_arr)
        rg, gg, bg_  = gains
        rgm, ggm, bgm = gammas
        for slider, val in [(self.rgb_r,   rg*100),
                            (self.rgb_g,   gg*100),
                            (self.rgb_b,   bg_*100),
                            (self.gamma_r, rgm*100),
                            (self.gamma_g, ggm*100),
                            (self.gamma_b, bgm*100)]:
            slider.set(val, fire_callback=False)
        self.status_var.set("AUTO WHITE BALANCED")
        self._pending_tab = 2
        self._fire_live()
