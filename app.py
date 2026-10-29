"""Tkinter application: menus, quote panel, resizable chart / strategy panes, live loop."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from chart import OVERLAYS, PANELS, ChartCanvas
from config import BG, BLUE, FG, GREEN, LONG_TERM, MUTED, PANEL, RED, SHORT_TERM, YELLOW
from data import Snapshot, evaluate_snapshot, fetch_snapshot
from strategies import CATEGORIES, PRESETS, STRATEGIES, Signal

class StockMonitorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Real-Time Stock Monitor & Strategy")
        root.geometry("1380x900")
        root.configure(bg=BG)

        self.queue: queue.Queue = queue.Queue()
        self.running = False
        self.snapshot: Snapshot | None = None
        self.view = tk.StringVar(value="short")
        self.refresh_secs = tk.IntVar(value=30)
        self.ticker_var = tk.StringVar(value="AAPL")
        self.countdown = 0
        self._fetching = False
        self.short_sig: Signal | None = None
        self.long_sig: Signal | None = None
        self.strategy_vars: dict[str, tk.BooleanVar] = {
            st.key: tk.BooleanVar(value=st.default_on) for st in STRATEGIES
        }
        self.chart_style = tk.StringVar(value="candle")
        self.panel_vars: dict[str, tk.BooleanVar] = {key: tk.BooleanVar(value=d) for key, _n, _w, d in PANELS}
        self.overlay_vars: dict[str, tk.BooleanVar] = {key: tk.BooleanVar(value=d) for key, _n, d in OVERLAYS}
        self.sr_count = tk.IntVar(value=3)

        self._style()
        self._build_menu()
        self._build()
        self.root.after(200, self._poll_queue)
        self.root.after(1000, self._tick)
        self.fetch_now()

    # ---- styling ---------------------------------------------------------- #
    def _style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, fieldbackground=PANEL, font=("Helvetica", 11))
        s.configure("TFrame", background=BG)
        s.configure("Panel.TFrame", background=PANEL)
        s.configure("TLabel", background=BG, foreground=FG)
        s.configure("Panel.TLabel", background=PANEL, foreground=FG)
        s.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=("Helvetica", 10))
        s.configure("Title.TLabel", background=PANEL, foreground=FG, font=("Helvetica", 18, "bold"))
        s.configure("Price.TLabel", background=PANEL, foreground=FG, font=("Helvetica", 34, "bold"))
        s.configure("TButton", background="#3b3b55", foreground=FG, padding=6, borderwidth=0)
        s.map("TButton", background=[("active", "#4a4a6a")])
        s.configure("TEntry", foreground=FG, insertcolor=FG)
        s.configure("TCombobox", foreground=FG, arrowcolor=FG)
        s.configure("TRadiobutton", background=BG, foreground=FG)
        s.map("TRadiobutton", background=[("active", BG)])
        s.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=FG, rowheight=24, borderwidth=0)
        s.configure("Treeview.Heading", background="#3b3b55", foreground=FG, relief="flat")

    # ---- strategy menu ---------------------------------------------------- #
    def enabled_strategies(self) -> list[str]:
        return [st.key for st in STRATEGIES if self.strategy_vars[st.key].get()]

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        m = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Strategies", menu=m)

        presets = tk.Menu(m, tearoff=0)
        for name, keys in PRESETS.items():
            presets.add_command(label=name, command=lambda k=keys: self.apply_preset(k))
        m.add_cascade(label="Presets", menu=presets)
        m.add_command(label="Select all", command=lambda: self.apply_preset(PRESETS["Everything"]))
        m.add_command(label="Clear all", command=lambda: self.apply_preset([]))
        m.add_separator()

        for cat in CATEGORIES:
            sub = tk.Menu(m, tearoff=0)
            for st in STRATEGIES:
                if st.category != cat:
                    continue
                sub.add_checkbutton(label=st.name, variable=self.strategy_vars[st.key],
                                    command=self._on_strategy_change)
            m.add_cascade(label=cat, menu=sub)

        m.add_separator()
        m.add_command(label="Describe strategies…", command=self.show_strategy_help)

        v = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=v)
        v.add_radiobutton(label="Candlestick chart", variable=self.chart_style, value="candle", command=self._redraw)
        v.add_radiobutton(label="Line chart", variable=self.chart_style, value="line", command=self._redraw)
        v.add_separator()
        pm = tk.Menu(v, tearoff=0)
        v.add_cascade(label="Panels", menu=pm)
        for key, name, _w, _d in PANELS:
            if key == "price":
                pm.add_checkbutton(label=name + " (always shown)", variable=self.panel_vars[key], state="disabled")
            else:
                pm.add_checkbutton(label=name, variable=self.panel_vars[key], command=self._redraw)
        pm.add_separator()
        pm.add_command(label="Only price", command=lambda: self.set_panels([]))
        pm.add_command(label="All panels", command=lambda: self.set_panels([k for k, *_ in PANELS]))
        om = tk.Menu(v, tearoff=0)
        v.add_cascade(label="Price overlays", menu=om)
        for key, name, _d in OVERLAYS:
            om.add_checkbutton(label=name, variable=self.overlay_vars[key], command=self._redraw)
        sm_ = tk.Menu(v, tearoff=0)
        v.add_cascade(label="Support / resistance levels per side", menu=sm_)
        for k in (1, 2, 3, 5, 8):
            sm_.add_radiobutton(label=str(k), variable=self.sr_count, value=k, command=self._redraw)
        v.add_separator()
        v.add_radiobutton(label="Short-term timeframe", variable=self.view, value="short", command=self._redraw)
        v.add_radiobutton(label="Long-term timeframe", variable=self.view, value="long", command=self._redraw)
        self.root.config(menu=menubar)

    def set_panels(self, keys: list[str]):
        for key, var in self.panel_vars.items():
            var.set(key == "price" or key in keys)
        self._redraw()

    def apply_preset(self, keys: list[str]):
        wanted = set(keys)
        for key, var in self.strategy_vars.items():
            var.set(key in wanted)
        self._on_strategy_change()

    def _on_strategy_change(self):
        self.lbl_enabled.configure(text=f"{len(self.enabled_strategies())} / {len(STRATEGIES)} strategies enabled")
        if self.snapshot is not None:
            self._render()

    def show_strategy_help(self):
        win = tk.Toplevel(self.root)
        win.title("Strategy reference")
        win.configure(bg=BG)
        win.geometry("760x620")
        txt = tk.Text(win, bg=PANEL, fg=FG, wrap="word", font=("Helvetica", 11), padx=14, pady=12, relief="flat")
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        txt.tag_configure("cat", foreground=YELLOW, font=("Helvetica", 13, "bold"), spacing1=10)
        txt.tag_configure("name", foreground=BLUE, font=("Helvetica", 11, "bold"))
        txt.tag_configure("on", foreground=GREEN)
        txt.tag_configure("off", foreground=MUTED)
        txt.insert("end", "Each enabled rule votes +1 (bullish), -1 (bearish) or 0. The votes are summed; "
                          "LONG needs at least ~1/3 of the enabled rules to agree (minimum 2), SHORT the same "
                          "on the downside. Otherwise the call is NEUTRAL. Stops and targets are 2x / 3x ATR.\n")
        for cat in CATEGORIES:
            txt.insert("end", f"\n{cat}\n", "cat")
            for st in STRATEGIES:
                if st.category != cat:
                    continue
                on = self.strategy_vars[st.key].get()
                txt.insert("end", ("● " if on else "○ "), "on" if on else "off")
                txt.insert("end", f"{st.name}\n", "name")
                txt.insert("end", f"    {st.description}\n")
        txt.configure(state="disabled")

    # ---- layout ----------------------------------------------------------- #
    def _build(self):
        top = ttk.Frame(self.root, padding=(12, 10))
        top.pack(fill="x")

        ttk.Label(top, text="Ticker").pack(side="left")
        ent = ttk.Entry(top, textvariable=self.ticker_var, width=10, font=("Helvetica", 13))
        ent.pack(side="left", padx=(6, 12))
        ent.bind("<Return>", lambda e: self.fetch_now())

        ttk.Button(top, text="Load / Refresh", command=self.fetch_now).pack(side="left")
        self.btn_live = ttk.Button(top, text="▶ Start live", command=self.toggle_live)
        self.btn_live.pack(side="left", padx=6)

        ttk.Label(top, text="Every").pack(side="left", padx=(16, 4))
        cb = ttk.Combobox(top, textvariable=self.refresh_secs, values=[10, 15, 30, 60, 120, 300], width=5, state="readonly")
        cb.pack(side="left")
        ttk.Label(top, text="sec").pack(side="left", padx=(4, 16))

        ttk.Label(top, text="Chart:").pack(side="left")
        ttk.Radiobutton(top, text="Short-term", variable=self.view, value="short", command=self._redraw).pack(side="left", padx=4)
        ttk.Radiobutton(top, text="Long-term", variable=self.view, value="long", command=self._redraw).pack(side="left", padx=4)

        self.status = ttk.Label(top, text="Idle", foreground=MUTED)
        self.status.pack(side="right")
        self.lbl_enabled = ttk.Label(top, text=f"{sum(st.default_on for st in STRATEGIES)} / {len(STRATEGIES)} strategies enabled",
                                     foreground=MUTED)
        self.lbl_enabled.pack(side="right", padx=(0, 18))

        # Left / right panes separated by a draggable sash.  Hovering the sash shows a
        # horizontal double arrow; dragging it trades width between the two panes.
        self.paned = tk.PanedWindow(self.root, orient="horizontal", bg=BG, sashwidth=8, sashrelief="flat",
                                    sashpad=0, sashcursor="sb_h_double_arrow", bd=0, opaqueresize=True)
        self.paned.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        left = ttk.Frame(self.paned)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        self.paned.add(left, minsize=420, stretch="always")

        # Quote panel (top-left)
        quote = ttk.Frame(left, style="Panel.TFrame", padding=14)
        quote.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        self.lbl_name = ttk.Label(quote, text="—", style="Title.TLabel")
        self.lbl_name.grid(row=0, column=0, columnspan=4, sticky="w")
        self.lbl_price = ttk.Label(quote, text="—", style="Price.TLabel")
        self.lbl_price.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.lbl_change = ttk.Label(quote, text="", style="Panel.TLabel", font=("Helvetica", 15, "bold"))
        self.lbl_change.grid(row=1, column=1, sticky="sw", padx=(14, 0), pady=(0, 8))
        self.quote_fields = {}
        keys = ["Prev close", "Day high", "Day low", "Volume", "Market", "Updated"]
        for i, key in enumerate(keys):
            r, c = 2 + (i // 3) * 2, i % 3
            ttk.Label(quote, text=key, style="Muted.TLabel").grid(row=r, column=c, sticky="w", padx=(0, 24), pady=(10, 0))
            v = ttk.Label(quote, text="—", style="Panel.TLabel", font=("Helvetica", 12, "bold"))
            v.grid(row=r + 1, column=c, sticky="w", padx=(0, 24))
            self.quote_fields[key] = v

        # Chart (bottom-left): native Tk canvas, vector drawn
        chart_frame = ttk.Frame(left, style="Panel.TFrame")
        chart_frame.grid(row=1, column=0, sticky="nsew")
        self.chart = ChartCanvas(chart_frame, self)
        self.chart.pack(fill="both", expand=True)

        # Strategy panel (right, spanning both rows)
        right = ttk.Frame(self.paned, style="Panel.TFrame", padding=14)
        self.paned.add(right, minsize=360, stretch="never")
        # initial split ~60 / 40 once the window has a size
        self.root.after(50, lambda: self.paned.sash_place(0, int(self.root.winfo_width() * 0.58), 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(5, weight=1)

        ttk.Label(right, text="Strategy signals", style="Title.TLabel").grid(row=0, column=0, sticky="w")

        self.sig_cards = {}
        for i, (key, cfg) in enumerate([("short", SHORT_TERM), ("long", LONG_TERM)]):
            card = tk.Frame(right, bg="#343450", padx=12, pady=10, highlightthickness=0)
            card.grid(row=1 + i, column=0, sticky="ew", pady=(10, 0))
            card.columnconfigure(1, weight=1)
            tk.Label(card, text=cfg["label"], bg="#343450", fg=MUTED, font=("Helvetica", 10)).grid(row=0, column=0, columnspan=2, sticky="w")
            lbl = tk.Label(card, text="—", bg="#343450", fg=FG, font=("Helvetica", 22, "bold"))
            lbl.grid(row=1, column=0, sticky="w")
            score = tk.Label(card, text="", bg="#343450", fg=MUTED, font=("Helvetica", 11))
            score.grid(row=1, column=1, sticky="e")
            levels = tk.Label(card, text="", bg="#343450", fg=FG, font=("Helvetica", 11), justify="left")
            levels.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
            self.sig_cards[key] = (lbl, score, levels)

        ttk.Label(right, text="Indicators (last bar)", style="Muted.TLabel").grid(row=3, column=0, sticky="w", pady=(16, 4))
        self.tree = ttk.Treeview(right, columns=("ind", "short", "long"), show="headings", height=11)
        for col, txt, w in [("ind", "Indicator", 130), ("short", "Short-term", 120), ("long", "Long-term", 120)]:
            self.tree.heading(col, text=txt)
            self.tree.column(col, width=w, anchor="w" if col == "ind" else "e")
        self.tree.grid(row=4, column=0, sticky="ew")

        ttk.Label(right, text="Why", style="Muted.TLabel").grid(row=5, column=0, sticky="nw", pady=(16, 4))
        self.reasons = tk.Text(right, bg="#343450", fg=FG, relief="flat", wrap="word", font=("Menlo", 10),
                               padx=10, pady=8, height=14, highlightthickness=0)
        self.reasons.grid(row=6, column=0, sticky="nsew")
        right.rowconfigure(6, weight=1)
        self.reasons.tag_configure("pos", foreground=GREEN)
        self.reasons.tag_configure("neg", foreground=RED)
        self.reasons.tag_configure("neu", foreground=MUTED)
        self.reasons.tag_configure("hdr", foreground=YELLOW, font=("Menlo", 10, "bold"))

        ttk.Label(right, text="Educational tool. Not financial advice. Data ~15 min delayed for some exchanges.",
                  style="Muted.TLabel", wraplength=420).grid(row=7, column=0, sticky="w", pady=(10, 0))

    # ---- live loop -------------------------------------------------------- #
    def toggle_live(self):
        self.running = not self.running
        if self.running:
            self.btn_live.configure(text="■ Stop live")
            self.countdown = 0
        else:
            self.btn_live.configure(text="▶ Start live")
            self._set_status("Live updates stopped.")

    def _tick(self):
        if self.running and not self._fetching:
            if self.countdown <= 0:
                self.fetch_now()
                self.countdown = int(self.refresh_secs.get())
            else:
                self.countdown -= 1
                self._set_status(f"Live · next refresh in {self.countdown}s")
        self.root.after(1000, self._tick)

    def fetch_now(self):
        ticker = self.ticker_var.get().strip().upper()
        if not ticker:
            return
        if self._fetching:
            return
        self._fetching = True
        self._set_status(f"Fetching {ticker} …")
        threading.Thread(target=self._worker, args=(ticker,), daemon=True).start()

    def _worker(self, ticker: str):
        try:
            snap = fetch_snapshot(ticker)
            self.queue.put(("ok", snap))
        except Exception as exc:  # noqa: BLE001
            self.queue.put(("err", str(exc)))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                self._fetching = False
                if kind == "ok":
                    self.snapshot = payload
                    self._render()
                else:
                    self._set_status(f"Error: {payload}", RED)
                    if not self.running:
                        messagebox.showerror("Fetch failed", payload)
        except queue.Empty:
            pass
        self.root.after(200, self._poll_queue)

    def _set_status(self, text: str, color: str = MUTED):
        self.status.configure(text=text, foreground=color)

    # ---- rendering -------------------------------------------------------- #
    def _render(self):
        s = self.snapshot
        if s is None:
            return
        chg = s.price - s.prev_close
        pct = chg / s.prev_close * 100 if s.prev_close else 0.0
        color = GREEN if chg >= 0 else RED

        self.lbl_name.configure(text=f"{s.name}  ({s.ticker})")
        self.lbl_price.configure(text=f"{s.price:,.2f} {s.currency}", foreground=color)
        self.lbl_change.configure(text=f"{chg:+,.2f}  ({pct:+.2f}%)", foreground=color)
        self.quote_fields["Prev close"].configure(text=f"{s.prev_close:,.2f}")
        self.quote_fields["Day high"].configure(text=f"{s.day_high:,.2f}")
        self.quote_fields["Day low"].configure(text=f"{s.day_low:,.2f}")
        self.quote_fields["Volume"].configure(text=f"{s.volume:,.0f}")
        self.quote_fields["Market"].configure(text=s.market_state)
        self.quote_fields["Updated"].configure(text=s.fetched_at.strftime("%H:%M:%S"))
        self._set_status(f"Updated {s.fetched_at.strftime('%H:%M:%S')}" + ("  · live" if self.running else ""), GREEN)

        self.short_sig, self.long_sig = evaluate_snapshot(s, self.enabled_strategies())
        for key, sig in (("short", self.short_sig), ("long", self.long_sig)):
            lbl, score, levels = self.sig_cards[key]
            col = GREEN if sig.label == "LONG" else RED if sig.label == "SHORT" else YELLOW
            lbl.configure(text=sig.label, fg=col)
            score.configure(text=f"score {sig.score:+d} of {sig.n_rules} rules  (need ±{sig.threshold})")
            if sig.label == "NEUTRAL":
                levels.configure(text="No edge: stay flat or hold existing position.")
            else:
                rr = abs(sig.target - sig.entry) / max(abs(sig.entry - sig.stop), 1e-9)
                levels.configure(
                    text=f"Entry ≈ {sig.entry:,.2f}   Stop {sig.stop:,.2f}   Target {sig.target:,.2f}   (R:R 1:{rr:.1f}, ATR-based)"
                )

        # indicator table
        self.tree.delete(*self.tree.get_children())
        sd, ld = s.short_df.iloc[-1], s.long_df.iloc[-1]
        rows = [
            ("SMA fast", f"SMA{SHORT_TERM['fast']} {sd['SMA_fast']:,.2f}", f"SMA{LONG_TERM['fast']} {ld['SMA_fast']:,.2f}"),
            ("SMA slow", f"SMA{SHORT_TERM['slow']} {sd['SMA_slow']:,.2f}", f"SMA{LONG_TERM['slow']} {ld['SMA_slow']:,.2f}"),
            ("RSI(14)", f"{sd['RSI']:.1f}", f"{ld['RSI']:.1f}"),
            ("MACD", f"{sd['MACD']:+.3f}", f"{ld['MACD']:+.3f}"),
            ("MACD hist", f"{sd['MACD_hist']:+.3f}", f"{ld['MACD_hist']:+.3f}"),
            ("ATR(14)", f"{sd['ATR']:.2f}", f"{ld['ATR']:.2f}"),
            ("BB upper", f"{sd['BB_up']:,.2f}", f"{ld['BB_up']:,.2f}"),
            ("BB lower", f"{sd['BB_lo']:,.2f}", f"{ld['BB_lo']:,.2f}"),
            ("Stoch %K / %D", f"{sd['STOCH_K']:.0f} / {sd['STOCH_D']:.0f}", f"{ld['STOCH_K']:.0f} / {ld['STOCH_D']:.0f}"),
            ("ADX (+DI/-DI)", f"{sd['ADX']:.0f} ({sd['DI_plus']:.0f}/{sd['DI_minus']:.0f})", f"{ld['ADX']:.0f} ({ld['DI_plus']:.0f}/{ld['DI_minus']:.0f})"),
            ("CCI", f"{sd['CCI']:.0f}", f"{ld['CCI']:.0f}"),
            ("VWAP", f"{sd['VWAP']:,.2f}", f"{ld['VWAP']:,.2f}"),
            ("Supertrend", f"{sd['ST']:,.2f} {'▲' if sd['ST_dir'] > 0 else '▼'}", f"{ld['ST']:,.2f} {'▲' if ld['ST_dir'] > 0 else '▼'}"),
            ("PSAR", f"{sd['PSAR']:,.2f}", f"{ld['PSAR']:,.2f}"),
        ]
        hi52, lo52 = s.long_df["High"].max(), s.long_df["Low"].min()
        rows.append(("52w high / low", "", f"{hi52:,.2f} / {lo52:,.2f}"))
        for r in rows:
            self.tree.insert("", "end", values=r)

        # reasons
        self.reasons.configure(state="normal")
        self.reasons.delete("1.0", "end")
        for title, sig in ((SHORT_TERM["label"], self.short_sig), (LONG_TERM["label"], self.long_sig)):
            self.reasons.insert("end", f"{title} → {sig.label}\n", "hdr")
            for line in sig.reasons:
                tag = "pos" if line.startswith("[+]") else "neg" if line.startswith("[-]") else "neu"
                self.reasons.insert("end", f"  {line}\n", tag)
            self.reasons.insert("end", "\n")
        self.reasons.configure(state="disabled")

        self._redraw()

    def _redraw(self):
        self.chart.redraw()
