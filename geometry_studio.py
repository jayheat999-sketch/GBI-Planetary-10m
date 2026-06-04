"""
geometry_studio.py

Analog-style monitoring rack for transformer geometry signals.

Treats your reaction / diffusion / laplacian / major_phase / minor_phase
signals as audio channels and gives you the instruments a mastering
engineer would have on the wall:

  - VECTORSCOPE   : X-Y Lissajous of any two channels, with phase shift
                    knob to rotate figure-8s into circles
  - GONIOMETER    : circular phase correlation meter (broadcast-style)
  - SPECTRUM      : FFT of each channel, peak detection
  - PHASE METER   : analog-needle-style instantaneous phase relationship
  - VU METERS     : amplitude monitoring per channel
  - COHERENCE     : frequency-resolved correlation (windowed)

Input: a geometry JSON from universal_reader.py (the kind your EE toolkit
already consumes).

Output: a tkinter+matplotlib window with all instruments live, sliders to
choose which channels feed the vectorscope and phase meter, a phase shift
knob, and a window-size knob for the coherence display.

Usage:
    python geometry_studio.py --geometry ./geometry_reader/GPT-2_geometry.json

    # Override which channels feed the X-Y display at startup:
    python geometry_studio.py --geometry geom.json --x diffusion --y reaction

Dependencies:
    numpy, scipy, matplotlib (tkinter ships with Python)

Jason Heater, 2026.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import coherence, hilbert
from scipy.fft import rfft, rfftfreq

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

import tkinter as tk
from tkinter import ttk


# Canonical channel names from the geometry framework.  Used only for
# display ordering and as defaults for --x / --y.  The loader itself
# accepts ANY numeric scalar field present in the per-token dicts,
# so GBI exports (angular_gradient, angular_laplacian, ...) and any
# future channels Just Work without code changes.
DEFAULT_CHANNELS = [
    "reaction", "diffusion", "laplacian", "major_phase", "minor_phase",
]


# -- token classification -----------------------------------------------------
# Color tokens by linguistic role so the vectorscope/goniometer show WHICH words
# form the structured core vs the scattered cloud. Works for the morpheme model
# (affix '#' markers) and the plain word model (no markers -> stem/content).

# Common English function words (closed-class). Lowercased, apostrophes kept.
_FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "nor", "so", "yet", "for", "of", "to",
    "in", "on", "at", "by", "with", "from", "as", "into", "onto", "upon", "over",
    "under", "out", "up", "down", "off", "about", "above", "below", "between",
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "have", "has", "had", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "not", "no", "yes",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours",
    "theirs", "this", "that", "these", "those", "who", "whom", "whose", "which",
    "what", "when", "where", "why", "how", "all", "any", "some", "if", "then",
    "than", "because", "while", "after", "before", "once", "there", "here",
    "very", "just", "too", "also", "more", "most", "such", "each", "every",
    "one", "two", "three", "his", "him", "'s", "n't", "'re", "'ve", "'ll", "'m", "'d",
}

# Colors per class (kept distinct on the dark background).
TOKEN_CLASS_COLORS = {
    "content":  "#00e5ff",   # cyan  -- nouns/verbs/adjs (the words that carry meaning)
    "function": "#ffb000",   # amber -- closed-class scaffold
    "affix":    "#ff4dd2",   # magenta -- morpheme pieces (#ing, un#)
    "stem":     "#7fff00",   # green -- a stem that was split off a word
    "punct":    "#888888",   # grey  -- punctuation
    "number":   "#ffffff",   # white -- digits
}
TOKEN_CLASS_ORDER = ["content", "function", "stem", "affix", "punct", "number"]


def classify_token(tok):
    """Map a token surface string to a linguistic class for coloring."""
    if tok is None:
        return "content"
    t = tok.strip()
    if t == "":
        return "punct"
    # morpheme markers from the morph tokenizer
    if t.endswith("#") or t.startswith("#"):
        return "affix"
    # punctuation: no alphanumeric content
    if not any(ch.isalnum() for ch in t):
        return "punct"
    # pure number
    core = t.strip("#")
    if core.isdigit():
        return "number"
    low = core.lower()
    if low in _FUNCTION_WORDS:
        return "function"
    return "content"


# -- data loading --------------------------------------------------------------

def _is_numeric_scalar(v):
    """True for ints, floats, and numpy/python bool-ish things that fit a float()."""
    if isinstance(v, bool):
        return False    # bools sneak through isinstance(.., int); skip them
    if isinstance(v, (int, float)):
        return True
    return False


def load_geometry(paths):
    """Load one or more geometry JSONs and return a dict of channel arrays plus labels.

    `paths` may be a single path or a list of paths.  When multiple paths
    are given they MUST have the same token labels (same length, same keys
    in the same order) -- typically one file is the standard geometry
    export and the others are GBI / probe exports for the same model and
    sequence.  All numeric scalar fields are collected.  If two files
    contain the same channel name, the LATER file wins and a warning is
    printed -- this lets you override a channel by feeding a fresh export
    after the canonical one.

    The loader does not hardcode channel names.  Any numeric scalar field
    in the per-token dicts is picked up automatically.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    merged_tokens = None      # dict: token_label -> {channel: value, ...}
    labels_order  = None
    sources = {}              # channel_name -> path it came from (for the warning)

    for path in paths:
        with open(path) as f:
            data = json.load(f)

        tokens = data["tokens"]
        these_labels = list(tokens.keys())

        if merged_tokens is None:
            merged_tokens = {lbl: dict(tokens[lbl]) for lbl in these_labels}
            labels_order  = these_labels
        else:
            if these_labels != labels_order:
                raise ValueError(
                    f"Token labels in {path} do not match the first file. "
                    f"All --geometry files must come from the same sequence "
                    f"(same length, same keys, same order)."
                )
            for lbl in these_labels:
                for k, v in tokens[lbl].items():
                    if k in merged_tokens[lbl] and k in sources and sources[k] != path:
                        # Same channel from two files -- later one wins.
                        pass
                    merged_tokens[lbl][k] = v

        # Track which file each channel came from, for the override warning.
        for k in tokens[these_labels[0]].keys():
            if k in sources and sources[k] != path:
                print(f"  [merge] '{k}' overridden by {path} "
                      f"(was from {sources[k]})")
            sources[k] = path

    # Discover all numeric scalar fields present in every token.  We require
    # presence in the FIRST token only -- ragged channels are tolerated and
    # any missing values are filled with NaN, which scipy/np handle fine for
    # display purposes.
    first = merged_tokens[labels_order[0]]
    candidate_names = [k for k, v in first.items() if _is_numeric_scalar(v)]

    # Put canonical channels first (in their conventional order), then
    # everything else alphabetically -- so the dropdowns and defaults pick
    # sensible names when both kinds are present.
    canonical_present = [c for c in DEFAULT_CHANNELS if c in candidate_names]
    extras            = sorted(c for c in candidate_names if c not in DEFAULT_CHANNELS)
    ordered_names     = canonical_present + extras

    channels = {}
    for name in ordered_names:
        arr = np.array(
            [merged_tokens[t].get(name, float("nan")) for t in labels_order],
            dtype=np.float64,
        )
        channels[name] = arr

    if not channels:
        raise ValueError(
            f"No numeric scalar channels found in {paths}. "
            f"Expected per-token dicts containing numeric fields; "
            f"got first-token keys: {list(first.keys())}"
        )

    # Pull the token surface strings (for token-aware coloring + boundaries).
    # Falls back to the token label if no 'token_str' field is present.
    token_strs = [
        str(merged_tokens[t].get("token_str", t)) for t in labels_order
    ]
    has_tokens = any("token_str" in merged_tokens[t] for t in labels_order)

    print(f"Loaded {len(labels_order)} tokens from {len(paths)} file(s).")
    print(f"  Channels: {list(channels.keys())}")
    if has_tokens:
        print(f"  token_str present -> token-aware coloring enabled.")
    else:
        print(f"  no token_str field -> coloring falls back to position only.")
    return channels, labels_order, token_strs, has_tokens


def normalize(x):
    """Map x to [-1, 1] for display. Handles dead signals."""
    rng = x.max() - x.min()
    if rng < 1e-12:
        return np.zeros_like(x)
    return 2.0 * (x - x.min()) / rng - 1.0


# -- instruments ---------------------------------------------------------------

class Studio:
    """The monitoring rack. Owns all instrument axes and updates them
    when the user changes a knob."""

    def __init__(self, channels, labels, token_strs=None, has_tokens=False,
                 init_x="diffusion", init_y="reaction"):
        self.channels = channels
        self.labels = labels
        self.channel_names = list(channels.keys())

        # Token strings + per-token linguistic class (for token-aware coloring).
        self.token_strs = token_strs if token_strs is not None else list(labels)
        self.has_tokens = has_tokens
        self.token_classes = [classify_token(t) for t in self.token_strs]
        # Sentence-boundary mask: a token that IS or ENDS with . ! ? marks a
        # boundary at its position (used for the overlay on the position line).
        self.boundary_idx = [
            i for i, t in enumerate(self.token_strs)
            if t.strip()[-1:] in {".", "!", "?"}
        ]
        # Coloring mode for vectorscope/goniometer: "position" or "class".
        self.color_mode = "class" if has_tokens else "position"

        # Pre-normalize copies for display. Keep originals for analysis.
        self.norm = {k: normalize(v) for k, v in channels.items()}

        # Pick safe defaults if requested channels aren't available.
        if init_x not in self.channel_names:
            init_x = self.channel_names[0]
        if init_y not in self.channel_names:
            init_y = self.channel_names[1] if len(self.channel_names) > 1 else self.channel_names[0]
        self.x_name = init_x
        self.y_name = init_y
        # Channel shown on the "signal vs token position" line.
        self.pos_name = init_x

        self.phase_shift_deg = 0.0
        self.coherence_window = 256

        self._build_window()

    def _point_colors(self, n):
        """Colors for the n scatter points under the current color mode."""
        if self.color_mode == "class":
            return [TOKEN_CLASS_COLORS[c] for c in self.token_classes[:n]]
        return plt.cm.viridis(np.linspace(0, 1, n))

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("Geometry Studio — Transformer Monitoring Rack")
        self.root.geometry("1400x950")
        self.root.configure(bg="#1a1a1a")

        # Notebook tabs: 2D monitoring rack | 3D inspection bench
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background="#1a1a1a", borderwidth=0)
        style.configure("TNotebook.Tab",
                        background="#2a2a2a", foreground="#e0e0e0",
                        padding=[12, 6], font=("Helvetica", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", "#3a3a3a")],
                  foreground=[("selected", "#00ff88")])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.tab_rack = tk.Frame(self.notebook, bg="#1a1a1a")
        self.tab_3d   = tk.Frame(self.notebook, bg="#1a1a1a")
        self.notebook.add(self.tab_rack, text="  MONITORING RACK  ")
        self.notebook.add(self.tab_3d,   text="  3D INSPECTION BENCH  ")

        self._build_rack_tab()
        self._build_3d_tab()

    def _build_rack_tab(self):
        # -- main figure with all the scopes -----------------------------------
        self.fig = plt.figure(figsize=(14, 8), facecolor="#1a1a1a")
        gs = GridSpec(
            3, 4, figure=self.fig,
            width_ratios=[1.4, 1.0, 1.0, 1.0],
            height_ratios=[1.0, 1.0, 1.0],
            hspace=0.45, wspace=0.35,
            left=0.05, right=0.97, top=0.94, bottom=0.07,
        )

        # Left column: big vectorscope (spans 2 rows) + goniometer
        self.ax_vector = self.fig.add_subplot(gs[0:2, 0])
        self.ax_gonio  = self.fig.add_subplot(gs[2, 0], projection="polar")

        # Middle columns: spectrum analyzers (one per channel, stacked)
        self.ax_spectra = [
            self.fig.add_subplot(gs[i, 1]) for i in range(3)
        ]

        # Right: phase meter, VU meters, coherence
        self.ax_phase    = self.fig.add_subplot(gs[0, 2], projection="polar")
        self.ax_vu       = self.fig.add_subplot(gs[0, 3])
        self.ax_coh      = self.fig.add_subplot(gs[1, 2:4])
        # Bottom-right split: signal-vs-token-position line | lissajous readout
        self.ax_posline  = self.fig.add_subplot(gs[2, 2])
        self.ax_lissinfo = self.fig.add_subplot(gs[2, 3])

        for ax in [self.ax_vector, self.ax_vu, self.ax_coh, self.ax_lissinfo,
                   self.ax_posline, *self.ax_spectra]:
            ax.set_facecolor("#0a0a0a")
            for spine in ax.spines.values():
                spine.set_color("#404040")
            ax.tick_params(colors="#a0a0a0", labelsize=8)
            ax.xaxis.label.set_color("#c0c0c0")
            ax.yaxis.label.set_color("#c0c0c0")
            ax.title.set_color("#e0e0e0")

        for ax in [self.ax_gonio, self.ax_phase]:
            ax.set_facecolor("#0a0a0a")
            ax.tick_params(colors="#a0a0a0", labelsize=7)
            ax.title.set_color("#e0e0e0")

        # Canvas (rack tab)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_rack)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Rack control panel
        panel = tk.Frame(self.tab_rack, bg="#2a2a2a", height=80)
        panel.pack(side=tk.BOTTOM, fill=tk.X)
        self._build_controls(panel)

        # Initial draw of all 2D instruments
        self.refresh()

    def _build_3d_tab(self):
        """The 3D inspection bench.

        Center stage: a single large 3D scatter of (X, Y, Z). Z defaults to
        token index — which is the diagnostic axis for distinguishing
        'helix in time' from 'static 2D figure with phase drift'. Other Z
        options: a third channel of the geometry, or 'unwrapped angle'
        (cumulative unwrapped phase of the X signal), which is the right
        axis if you want to flatten a helix into a straight ribbon.

        Camera sliders let you orbit. Slab slicer hides tokens outside a
        chosen Z range. Connect-the-dots option draws a polyline between
        consecutive tokens — that is what reveals whether the geometry is
        traversed as a connected path (it should be, if it's a real
        embedding) versus visited in scattered jumps.
        """
        # 3D figure
        self.fig_3d = plt.figure(figsize=(13, 8.5), facecolor="#1a1a1a")
        self.ax_3d  = self.fig_3d.add_subplot(111, projection="3d", facecolor="#0a0a0a")

        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.tab_3d)
        self.canvas_3d.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Native matplotlib navigation toolbar — gives you free orbit with
        # the mouse (drag = rotate, right-drag = zoom). Worth having.
        toolbar_frame = tk.Frame(self.tab_3d, bg="#1a1a1a")
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar_3d = NavigationToolbar2Tk(self.canvas_3d, toolbar_frame)
        toolbar_3d.config(background="#1a1a1a")
        toolbar_3d.update()

        # 3D control panel
        panel = tk.Frame(self.tab_3d, bg="#2a2a2a", height=120)
        panel.pack(side=tk.BOTTOM, fill=tk.X)

        style_lbl = {"bg": "#2a2a2a", "fg": "#e0e0e0", "font": ("Helvetica", 10)}
        style_btn = {"bg": "#3a3a3a", "fg": "#00ff88",
                     "font": ("Helvetica", 9, "bold"),
                     "activebackground": "#4a4a4a", "activeforeground": "#00ff88",
                     "relief": "flat", "borderwidth": 1, "padx": 8, "pady": 3}

        # Row 0: axis selectors
        tk.Label(panel, text="X:", **style_lbl).grid(row=0, column=0, padx=(10, 2), pady=6, sticky="e")
        self.x3_var = tk.StringVar(value=self.x_name)
        x3 = ttk.Combobox(panel, textvariable=self.x3_var, values=self.channel_names,
                          state="readonly", width=13)
        x3.grid(row=0, column=1, padx=2, pady=6)
        x3.bind("<<ComboboxSelected>>", lambda e: self._draw_3d())

        tk.Label(panel, text="Y:", **style_lbl).grid(row=0, column=2, padx=(15, 2), pady=6, sticky="e")
        self.y3_var = tk.StringVar(value=self.y_name)
        y3 = ttk.Combobox(panel, textvariable=self.y3_var, values=self.channel_names,
                          state="readonly", width=13)
        y3.grid(row=0, column=3, padx=2, pady=6)
        y3.bind("<<ComboboxSelected>>", lambda e: self._draw_3d())

        tk.Label(panel, text="Z:", **style_lbl).grid(row=0, column=4, padx=(15, 2), pady=6, sticky="e")
        z_options = ["token_index", "unwrapped_X_angle"] + self.channel_names
        self.z3_var = tk.StringVar(value="token_index")
        z3 = ttk.Combobox(panel, textvariable=self.z3_var, values=z_options,
                          state="readonly", width=18)
        z3.grid(row=0, column=5, padx=2, pady=6)
        z3.bind("<<ComboboxSelected>>", lambda e: self._draw_3d())

        # Connect-the-dots toggle
        self.connect_var = tk.BooleanVar(value=False)
        connect_cb = tk.Checkbutton(
            panel, text="connect tokens", variable=self.connect_var,
            command=self._draw_3d,
            bg="#2a2a2a", fg="#e0e0e0", selectcolor="#0a0a0a",
            activebackground="#2a2a2a", activeforeground="#00ff88",
            font=("Helvetica", 9),
        )
        connect_cb.grid(row=0, column=6, padx=(15, 6), pady=6)

        # Slab slicer: hide tokens outside [z_min, z_max] percentile range
        tk.Label(panel, text="SLAB (% of Z range):", **style_lbl).grid(row=1, column=0, columnspan=2, padx=10, pady=6, sticky="e")
        self.slab_lo_var = tk.DoubleVar(value=0.0)
        self.slab_hi_var = tk.DoubleVar(value=100.0)
        slab_lo = tk.Scale(panel, from_=0, to=100, resolution=1, orient=tk.HORIZONTAL,
                           variable=self.slab_lo_var, length=140,
                           command=lambda v: self._draw_3d(),
                           bg="#2a2a2a", fg="#e0e0e0", troughcolor="#0a0a0a",
                           highlightthickness=0, font=("Helvetica", 8), label="low")
        slab_lo.grid(row=1, column=2, columnspan=2, padx=4, pady=2)
        slab_hi = tk.Scale(panel, from_=0, to=100, resolution=1, orient=tk.HORIZONTAL,
                           variable=self.slab_hi_var, length=140,
                           command=lambda v: self._draw_3d(),
                           bg="#2a2a2a", fg="#e0e0e0", troughcolor="#0a0a0a",
                           highlightthickness=0, font=("Helvetica", 8), label="high")
        slab_hi.grid(row=1, column=4, columnspan=2, padx=4, pady=2)

        # View presets
        tk.Label(panel, text="VIEW:", **style_lbl).grid(row=1, column=6, padx=(20, 4), pady=6, sticky="e")
        preset_frame = tk.Frame(panel, bg="#2a2a2a")
        preset_frame.grid(row=1, column=7, columnspan=4, padx=2, pady=2, sticky="w")
        tk.Button(preset_frame, text="TOP (XY)",   command=lambda: self._preset_view(90, -90), **style_btn).pack(side=tk.LEFT, padx=2)
        tk.Button(preset_frame, text="SIDE (XZ)",  command=lambda: self._preset_view(0, -90),  **style_btn).pack(side=tk.LEFT, padx=2)
        tk.Button(preset_frame, text="SIDE (YZ)",  command=lambda: self._preset_view(0, 0),    **style_btn).pack(side=tk.LEFT, padx=2)
        tk.Button(preset_frame, text="CORNER",     command=lambda: self._preset_view(30, -60), **style_btn).pack(side=tk.LEFT, padx=2)

        # Initial draw
        self._draw_3d()

    def _preset_view(self, elev, azim):
        self.ax_3d.view_init(elev=elev, azim=azim)
        self.canvas_3d.draw_idle()

    def _draw_3d(self, *_):
        """Render the 3D scatter using the current axis/slab/connect settings."""
        ax = self.ax_3d
        ax.clear()
        ax.set_facecolor("#0a0a0a")

        x_name = self.x3_var.get()
        y_name = self.y3_var.get()
        z_name = self.z3_var.get()

        x = self.norm[x_name]
        y = self.norm[y_name]

        if z_name == "token_index":
            z = np.arange(len(x), dtype=np.float64)
            z_label = "token index"
        elif z_name == "unwrapped_X_angle":
            # If X is an angle-like signal, this turns the periodic wrap into
            # a monotonic axis — the helix becomes a straight ribbon.
            analytic = hilbert(self.norm[x_name])
            instantaneous_phase = np.angle(analytic)
            z = np.unwrap(instantaneous_phase)
            z_label = f"unwrapped angle of {x_name} (rad, cumulative)"
        else:
            z = self.norm[z_name]
            z_label = z_name

        # Slab slice on Z (percentile range)
        lo_pct = self.slab_lo_var.get()
        hi_pct = self.slab_hi_var.get()
        if lo_pct > hi_pct:
            lo_pct, hi_pct = hi_pct, lo_pct
        z_lo = np.percentile(z, lo_pct)
        z_hi = np.percentile(z, hi_pct)
        mask = (z >= z_lo) & (z <= z_hi)

        x_m = x[mask]; y_m = y[mask]; z_m = z[mask]
        n_shown = mask.sum()

        if n_shown == 0:
            ax.text2D(0.5, 0.5, "Slab is empty — widen the range",
                      transform=ax.transAxes, color="#a0a0a0",
                      ha="center", va="center")
        else:
            # Color by Z so the gradient reveals path direction
            colors = plt.cm.viridis((z_m - z_m.min()) / max(z_m.max() - z_m.min(), 1e-12))

            if self.connect_var.get():
                # Polyline through tokens in original sequence order (within
                # the visible slab). Thin, semi-transparent so dense regions
                # don't overwhelm. Then dots on top for individual tokens.
                ax.plot(x_m, y_m, z_m, color="#00ff88", linewidth=0.5, alpha=0.35)

            ax.scatter(x_m, y_m, z_m, c=colors, s=6, alpha=0.7, edgecolors="none")

        ax.set_xlabel(f"X = {x_name}", color="#c0c0c0", fontsize=10, labelpad=8)
        ax.set_ylabel(f"Y = {y_name}", color="#c0c0c0", fontsize=10, labelpad=8)
        ax.set_zlabel(z_label,         color="#c0c0c0", fontsize=10, labelpad=8)

        # Match the 2D vectorscope's normalized axes for XY
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

        # Cosmetic: dim grid, neon-on-black aesthetic
        ax.tick_params(colors="#a0a0a0", labelsize=8)
        ax.xaxis.pane.set_facecolor("#0a0a0a")
        ax.yaxis.pane.set_facecolor("#0a0a0a")
        ax.zaxis.pane.set_facecolor("#0a0a0a")
        ax.xaxis.pane.set_edgecolor("#303030")
        ax.yaxis.pane.set_edgecolor("#303030")
        ax.zaxis.pane.set_edgecolor("#303030")
        ax.grid(True, color="#252525", linewidth=0.5)

        ax.set_title(
            f"3D INSPECTION:  {x_name} × {y_name} × {z_name}    "
            f"showing {n_shown:,} / {len(x):,} tokens",
            fontsize=11, fontweight="bold", color="#00ff88", pad=14,
        )
        self.canvas_3d.draw_idle()

    def _build_controls(self, panel):
        style = {"bg": "#2a2a2a", "fg": "#e0e0e0", "font": ("Helvetica", 10)}

        # X channel selector
        tk.Label(panel, text="VECTORSCOPE X:", **style).grid(row=0, column=0, padx=10, pady=8, sticky="e")
        self.x_var = tk.StringVar(value=self.x_name)
        x_menu = ttk.Combobox(panel, textvariable=self.x_var,
                              values=self.channel_names, state="readonly", width=14)
        x_menu.grid(row=0, column=1, padx=4, pady=8)
        x_menu.bind("<<ComboboxSelected>>", lambda e: self._on_channel_change())

        # Y channel selector
        tk.Label(panel, text="Y:", **style).grid(row=0, column=2, padx=10, pady=8, sticky="e")
        self.y_var = tk.StringVar(value=self.y_name)
        y_menu = ttk.Combobox(panel, textvariable=self.y_var,
                              values=self.channel_names, state="readonly", width=14)
        y_menu.grid(row=0, column=3, padx=4, pady=8)
        y_menu.bind("<<ComboboxSelected>>", lambda e: self._on_channel_change())

        # Phase shift knob — the cassette deck azimuth control
        tk.Label(panel, text="PHASE SHIFT (deg):", **style).grid(row=0, column=4, padx=15, pady=8, sticky="e")
        self.phase_var = tk.DoubleVar(value=0.0)
        phase_slider = tk.Scale(
            panel, from_=-180, to=180, resolution=0.5,
            orient=tk.HORIZONTAL, variable=self.phase_var,
            length=200, command=lambda v: self._on_phase_change(),
            bg="#2a2a2a", fg="#e0e0e0", troughcolor="#0a0a0a",
            highlightthickness=0, font=("Helvetica", 9),
        )
        phase_slider.grid(row=0, column=5, padx=4, pady=4)

        # Coherence window size
        tk.Label(panel, text="COHERENCE WIN:", **style).grid(row=0, column=6, padx=15, pady=8, sticky="e")
        self.coh_var = tk.IntVar(value=256)
        coh_slider = tk.Scale(
            panel, from_=32, to=1024, resolution=32,
            orient=tk.HORIZONTAL, variable=self.coh_var,
            length=160, command=lambda v: self._on_coh_change(),
            bg="#2a2a2a", fg="#e0e0e0", troughcolor="#0a0a0a",
            highlightthickness=0, font=("Helvetica", 9),
        )
        coh_slider.grid(row=0, column=7, padx=4, pady=4)

        # -- second row: token-aware coloring + position-line channel ----------
        # COLOR BY: position (viridis) vs token class (content/function/...)
        tk.Label(panel, text="COLOR BY:", **style).grid(row=1, column=0, padx=10, pady=6, sticky="e")
        self.color_var = tk.StringVar(value=self.color_mode)
        color_menu = ttk.Combobox(panel, textvariable=self.color_var,
                                  values=["class", "position"], state="readonly", width=14)
        color_menu.grid(row=1, column=1, padx=4, pady=6)
        color_menu.bind("<<ComboboxSelected>>", lambda e: self._on_color_change())
        if not self.has_tokens:
            # no token strings -> class coloring is meaningless; lock to position
            color_menu.set("position")
            color_menu.configure(state="disabled")

        # POSITION LINE channel
        tk.Label(panel, text="POSITION LINE:", **style).grid(row=1, column=2, padx=10, pady=6, sticky="e")
        self.pos_var = tk.StringVar(value=self.pos_name)
        pos_menu = ttk.Combobox(panel, textvariable=self.pos_var,
                                values=self.channel_names, state="readonly", width=14)
        pos_menu.grid(row=1, column=3, padx=4, pady=6)
        pos_menu.bind("<<ComboboxSelected>>", lambda e: self._on_posline_change())

    # -- event handlers --------------------------------------------------------

    def _on_color_change(self):
        self.color_mode = self.color_var.get()
        self._draw_vectorscope()
        self._draw_goniometer()
        self._draw_position_line()
        self.canvas.draw_idle()

    def _on_posline_change(self):
        self.pos_name = self.pos_var.get()
        self._draw_position_line()
        self.canvas.draw_idle()

    def _on_channel_change(self):
        self.x_name = self.x_var.get()
        self.y_name = self.y_var.get()
        self.refresh()

    def _on_phase_change(self):
        self.phase_shift_deg = self.phase_var.get()
        self._draw_vectorscope()
        self._draw_lissajous_info()
        self.canvas.draw_idle()

    def _on_coh_change(self):
        self.coherence_window = int(self.coh_var.get())
        self._draw_coherence()
        self.canvas.draw_idle()

    # -- main refresh ----------------------------------------------------------

    def refresh(self):
        self._draw_vectorscope()
        self._draw_goniometer()
        self._draw_spectra()
        self._draw_phase_meter()
        self._draw_vu_meters()
        self._draw_coherence()
        self._draw_position_line()
        self._draw_lissajous_info()
        self.canvas.draw_idle()

    # -- vectorscope -----------------------------------------------------------

    def _apply_phase_shift(self, sig, deg):
        """
        Shift a real signal's phase by deg degrees using the analytic signal.
        This is the software equivalent of an analog phase shifter knob:
        builds the Hilbert transform, rotates the complex phasor by the
        requested angle, takes the real part.
        """
        if abs(deg) < 1e-6:
            return sig
        analytic = hilbert(sig)
        shifted = analytic * np.exp(1j * np.deg2rad(deg))
        return shifted.real

    def _draw_vectorscope(self):
        ax = self.ax_vector
        ax.clear()
        ax.set_facecolor("#0a0a0a")

        x = self.norm[self.x_name]
        y = self.norm[self.y_name]
        y_shifted = self._apply_phase_shift(y, self.phase_shift_deg)

        n = len(x)
        colors = self._point_colors(n)
        ax.scatter(x, y_shifted, c=colors, s=5, alpha=0.6, edgecolors="none")

        # Reference diagonals: in-phase (+45) and anti-phase (-45)
        ax.plot([-1, 1], [-1, 1], color="#404040", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.plot([-1, 1], [1, -1], color="#404040", linestyle="--", linewidth=0.8, alpha=0.7)
        # Unit circle (90° phase, equal amplitude reference)
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color="#404040", linewidth=0.6, alpha=0.6)

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.set_xlabel(f"X = {self.x_name}")
        ax.set_ylabel(f"Y = {self.y_name}  (shifted {self.phase_shift_deg:+.1f}°)")
        mode = "by token class" if self.color_mode == "class" else "by position"
        ax.set_title(f"VECTORSCOPE  ({mode})", fontsize=11, fontweight="bold",
                     color="#00ff88")
        ax.grid(True, alpha=0.15, color="#404040")
        self._draw_class_legend(ax)

    def _draw_class_legend(self, ax):
        """Small legend showing which color is which token class. Only drawn
        when coloring by class (and only for classes actually present)."""
        if self.color_mode != "class":
            return
        present = [c for c in TOKEN_CLASS_ORDER if c in set(self.token_classes)]
        handles = [
            Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor=TOKEN_CLASS_COLORS[c],
                   markeredgecolor="none", markersize=6, label=c)
            for c in present
        ]
        leg = ax.legend(handles=handles, loc="upper left", fontsize=7,
                        framealpha=0.25, facecolor="#101010", edgecolor="#404040",
                        labelcolor="#d0d0d0", ncol=2, handletextpad=0.3,
                        columnspacing=0.8, borderpad=0.4)
        leg.set_zorder(20)

    # -- goniometer ------------------------------------------------------------

    def _draw_goniometer(self):
        """Circular phase correlation. Standard broadcast goniometer:
        plot (M, S) = ((L+R)/sqrt(2), (L-R)/sqrt(2)) in polar.
        Tight vertical line = mono. Tight horizontal line = anti-phase.
        Blob = stereo. Tilted line = correlated stereo."""
        ax = self.ax_gonio
        ax.clear()
        ax.set_facecolor("#0a0a0a")

        L = self.norm[self.x_name]
        R = self.norm[self.y_name]
        M = (L + R) / np.sqrt(2)
        S = (L - R) / np.sqrt(2)
        # Convert (M, S) to polar: r = magnitude, theta = atan2(S, M).
        # Goniometer convention rotates so "mono" is vertical (up).
        r = np.sqrt(M*M + S*S)
        theta = np.arctan2(S, M) + np.pi/2  # rotate so M-axis points up

        n = len(L)
        colors = self._point_colors(n)
        ax.scatter(theta, r, c=colors, s=4, alpha=0.55, edgecolors="none")

        ax.set_ylim(0, max(r.max() * 1.1, 0.1))
        ax.set_title("GONIOMETER (M/S)", fontsize=10, fontweight="bold",
                     color="#00ff88", pad=15)
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.2, color="#404040")

    # -- spectrum analyzers ----------------------------------------------------

    def _draw_spectra(self):
        names = [self.x_name, self.y_name]
        # Third spectrum: third available channel, distinct from X and Y if possible
        third = next(
            (c for c in self.channel_names if c not in (self.x_name, self.y_name)),
            self.channel_names[0],
        )
        names.append(third)

        colors = ["#00ff88", "#ff6688", "#88aaff"]
        for ax, name, color in zip(self.ax_spectra, names, colors):
            ax.clear()
            ax.set_facecolor("#0a0a0a")
            sig = self.norm[name]
            spec = np.abs(rfft(sig))
            freqs = rfftfreq(len(sig), d=1.0)
            # log scale magnitude for dynamic range; clamp floor
            mag_db = 20 * np.log10(np.clip(spec / (spec.max() + 1e-12), 1e-5, 1.0))

            ax.plot(freqs, mag_db, color=color, linewidth=1.0)
            ax.fill_between(freqs, mag_db, -100, color=color, alpha=0.15)
            ax.set_xlim(0, 0.5)
            ax.set_ylim(-80, 5)
            ax.set_title(f"SPECTRUM: {name}", fontsize=9, color=color, fontweight="bold")
            ax.set_xlabel("cycles/token", fontsize=8)
            ax.set_ylabel("dB", fontsize=8)
            ax.grid(True, alpha=0.15, color="#404040")

    # -- phase meter (analog needle style) -------------------------------------

    def _draw_phase_meter(self):
        ax = self.ax_phase
        ax.clear()
        ax.set_facecolor("#0a0a0a")

        x = self.norm[self.x_name]
        y = self.norm[self.y_name]
        y_shifted = self._apply_phase_shift(y, self.phase_shift_deg)

        # Phase via dominant FFT component of each
        fx = rfft(x); fy = rfft(y_shifted)
        # Skip DC; find dominant bin in X
        mag = np.abs(fx[1:])
        if mag.size == 0 or mag.max() < 1e-12:
            phase_diff = 0.0
        else:
            k = np.argmax(mag) + 1
            phase_diff = np.angle(fy[k]) - np.angle(fx[k])
        # Wrap to (-pi, pi]
        phase_diff = ((phase_diff + np.pi) % (2*np.pi)) - np.pi

        # Draw the needle
        ax.plot([0, phase_diff], [0, 1], color="#00ff88", linewidth=3)
        ax.plot([phase_diff], [1], "o", color="#00ff88", markersize=8)

        # Reference marks at 0, ±90°, 180°
        for mark, label in [(0, "0° (mono)"), (np.pi/2, "+90°"),
                            (-np.pi/2, "-90°"), (np.pi, "180° (anti)")]:
            ax.plot([mark, mark], [0.85, 1.0], color="#606060", linewidth=1)

        ax.set_ylim(0, 1.1)
        ax.set_title(f"PHASE METER: {np.degrees(phase_diff):+.1f}°",
                     fontsize=10, fontweight="bold", color="#00ff88", pad=15)
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.2, color="#404040")

    # -- VU meters -------------------------------------------------------------

    def _draw_vu_meters(self):
        ax = self.ax_vu
        ax.clear()
        ax.set_facecolor("#0a0a0a")

        names = self.channel_names
        # RMS in dB of the original (un-normalized) channels
        rms_db = []
        for n in names:
            sig = self.channels[n]
            rms = np.sqrt(np.mean(sig * sig))
            # Normalize each channel's "0 dB" reference to its own peak,
            # so all meters land in a comparable range. This is what a
            # mastering VU does after gain staging.
            peak = np.max(np.abs(sig)) + 1e-12
            rms_db.append(20 * np.log10(rms / peak))

        y_pos = np.arange(len(names))
        # Color: green if below -6 dB, yellow if -6 to -3, red if hotter
        bar_colors = []
        for db in rms_db:
            if db < -6:
                bar_colors.append("#00ff88")
            elif db < -3:
                bar_colors.append("#ffcc00")
            else:
                bar_colors.append("#ff4444")
        ax.barh(y_pos, rms_db, color=bar_colors, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(-30, 0)
        ax.axvline(-6, color="#ffcc00", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.axvline(-3, color="#ff4444", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_title("VU METERS (dB rel. peak)", fontsize=10, color="#00ff88", fontweight="bold")
        ax.set_xlabel("dB", fontsize=8)
        ax.grid(True, alpha=0.15, color="#404040", axis="x")

    # -- coherence (frequency-resolved correlation) ----------------------------

    def _draw_coherence(self):
        ax = self.ax_coh
        ax.clear()
        ax.set_facecolor("#0a0a0a")

        x = self.norm[self.x_name]
        y = self.norm[self.y_name]

        nper = min(self.coherence_window, len(x) // 4)
        if nper < 16:
            ax.text(0.5, 0.5, "Not enough data for coherence",
                    ha="center", va="center", color="#a0a0a0",
                    transform=ax.transAxes)
        else:
            f, Cxy = coherence(x, y, fs=1.0, nperseg=nper)
            ax.plot(f, Cxy, color="#00ff88", linewidth=1.2)
            ax.fill_between(f, Cxy, 0, color="#00ff88", alpha=0.2)
            # Reference line at 0.5 — "half coupled"
            ax.axhline(0.5, color="#606060", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.set_xlim(0, 0.5)
            ax.set_ylim(0, 1.05)

        ax.set_title(f"COHERENCE: {self.x_name} ↔ {self.y_name}  (window={nper})",
                     fontsize=10, fontweight="bold", color="#00ff88")
        ax.set_xlabel("cycles/token", fontsize=8)
        ax.set_ylabel("|γ|²", fontsize=8)
        ax.grid(True, alpha=0.15, color="#404040")

    # -- Lissajous interpretation panel ----------------------------------------

    def _draw_position_line(self):
        """Signal vs token position — the most basic view for spotting WHERE in
        a sentence the geometry spikes. Vertical lines mark sentence boundaries
        (tokens ending in . ! ?), and points are tinted by token class so you can
        see the scaffold/content alternation along the sequence."""
        ax = self.ax_posline
        ax.clear()
        ax.set_facecolor("#0a0a0a")

        sig = self.norm[self.pos_name]
        n = len(sig)
        pos = np.arange(n)

        # faint connecting line for the trajectory shape
        ax.plot(pos, sig, color="#2f6f5a", linewidth=0.8, alpha=0.6, zorder=1)
        # points tinted by class (or position) so scaffold vs content is visible
        colors = self._point_colors(n)
        ax.scatter(pos, sig, c=colors, s=10, alpha=0.85, zorder=3, edgecolors="none")

        # sentence-boundary overlay
        for b in self.boundary_idx:
            ax.axvline(b, color="#ff4d4d", linewidth=0.7, alpha=0.45, zorder=2)

        ax.axhline(0, color="#404040", linewidth=0.5, alpha=0.5)
        ax.set_xlim(-1, n)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel("token position")
        ax.set_ylabel(self.pos_name, fontsize=8)
        nb = len(self.boundary_idx)
        ax.set_title(f"SIGNAL vs POSITION  (red = sentence end, {nb})",
                     fontsize=9, fontweight="bold", color="#00ff88")
        ax.grid(True, alpha=0.12, color="#404040")

    @staticmethod
    def _harmonic_stack(sig, top=3):
        """Top FFT peaks of a real signal (DC excluded), as (bin, relative_mag),
        plus the fraction of non-DC spectral energy lying OUTSIDE the dominant
        bin. That fraction is the 'helix' indicator: ~0 = a single clean
        frequency (simple Lissajous), high = energy spread across many harmonics
        (multi-lobe / screw-shaped figure)."""
        mag = np.abs(rfft(sig))
        if len(mag) <= 1:
            return [(0, 1.0)], 0.0
        ac = mag[1:]                       # drop DC
        total = float(np.sum(ac ** 2)) + 1e-12
        order = np.argsort(ac)[::-1]       # strongest first
        peak_mag = ac[order[0]] + 1e-12
        peaks = [(int(order[k] + 1), float(ac[order[k]] / peak_mag))
                 for k in range(min(top, len(ac)))]
        out_of_fundamental = 1.0 - float(ac[order[0]] ** 2) / total
        return peaks, out_of_fundamental

    def _draw_lissajous_info(self):
        ax = self.ax_lissinfo
        ax.clear()
        ax.set_facecolor("#0a0a0a")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#404040")

        x = self.norm[self.x_name]
        y = self.norm[self.y_name]
        y_shifted = self._apply_phase_shift(y, self.phase_shift_deg)

        corr = np.corrcoef(x, y_shifted)[0, 1]

        # Full harmonic stack per axis (not just the single dominant bin).
        xpeaks, x_oof = self._harmonic_stack(x)
        ypeaks, y_oof = self._harmonic_stack(y_shifted)
        kx, ky = xpeaks[0][0], ypeaks[0][0]

        # Raw ratio of dominant bins AND its lowest-terms reduction.
        g = math.gcd(kx, ky) or 1
        rx, ry = kx // g, ky // g
        raw_ratio = ky / kx if kx > 0 else float("nan")

        # The helix indicator: high out-of-fundamental energy on either axis
        # means the figure has many lobes (screw/helix), not a clean low-order
        # Lissajous, even if the reduced ratio is small.
        oof = max(x_oof, y_oof)
        rich = oof > 0.45

        # Verdict
        if abs(corr) > 0.85:
            shape = "LINE" + (" (in phase)" if corr > 0 else " (anti-phase)")
        elif rich and abs(corr) < 0.30:
            shape = f"HIGH-ORDER ({rx}:{ry} core, many lobes) — screw / helix"
        elif abs(corr) < 0.15 and (rx, ry) in [(2, 1), (1, 2)]:
            hi = "Y at 2× X" if (rx, ry) == (1, 2) else "X at 2× Y"
            shape = f"FIGURE-8 ({hi})"
        elif abs(corr) < 0.15 and (rx, ry) == (1, 1):
            shape = "CIRCLE / BLOB (1:1)"
        elif abs(corr) < 0.15:
            shape = f"LISSAJOUS {rx}:{ry}"
        else:
            shape = "ELLIPSE (partial correlation)"

        def fmt(peaks):
            return "  ".join(f"{b}:{m:.2f}" for b, m in peaks)

        info = [
            f"X = {self.x_name}",
            f"Y = {self.y_name}   shift {self.phase_shift_deg:+.1f}°",
            "",
            f"Pearson r = {corr:+.4f}",
            "",
            f"X harmonics (bin:rel)  {fmt(xpeaks)}",
            f"Y harmonics (bin:rel)  {fmt(ypeaks)}",
            f"dominant  X={kx} Y={ky}   raw {kx}:{ky}  reduced {rx}:{ry}",
            f"out-of-fundamental  X {x_oof*100:.0f}%  Y {y_oof*100:.0f}%"
            + ("  HIGH->multi-lobe" if rich else ""),
            "",
            f"READING: {shape}",
            "",
            "Sweep PHASE SHIFT: a clean low-order figure collapses",
            "to a line/circle; a true helix stays multi-lobed.",
        ]
        ax.text(0.02, 0.98, "\n".join(info), transform=ax.transAxes,
                color="#e0e0e0", fontsize=8.5, family="monospace",
                verticalalignment="top")
        ax.set_title("LISSAJOUS READOUT", fontsize=10, fontweight="bold", color="#00ff88")

    def run(self):
        self.root.mainloop()


# -- main ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Geometry Studio -- monitoring rack for transformer geometry signals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Standard usage:\n"
            "  python geometry_studio.py --geometry geom.json\n"
            "\n"
            "  # Merge a GBI export onto the same vectorscope as a standard geom:\n"
            "  python geometry_studio.py --geometry geom.json gbi_geometry_final.json \\\n"
            "      --x reaction --y angular_gradient\n"
        ),
    )
    p.add_argument("--geometry", required=True, nargs="+",
                   help="One or more geometry JSON files.  Multiple files must "
                        "share token labels; their channels are merged.")
    p.add_argument("--x", default="diffusion", help="Initial X channel for vectorscope")
    p.add_argument("--y", default="reaction",  help="Initial Y channel for vectorscope")
    args = p.parse_args()

    paths = []
    for g in args.geometry:
        path = Path(g)
        if not path.exists():
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)
        paths.append(path)

    channels, labels, token_strs, has_tokens = load_geometry(paths)
    studio = Studio(channels, labels, token_strs=token_strs, has_tokens=has_tokens,
                    init_x=args.x, init_y=args.y)
    studio.run()


if __name__ == "__main__":
    main()
