#!/usr/bin/env python3
"""Live preview window for board capture -- `capture_board_frames.py --interactive`.

WHY A NATIVE WINDOW RATHER THAN AN HTML PAGE. The hand-curation tools in
this project (make_gallery.py, make_review_gallery.py) write an HTML page
you open in a browser, and reusing that pattern here is tempting. It is
wrong for two reasons, the second decisive:

 1. The board serves ONE viewer. A browser tab showing the stream holds
    port 81's single httpd task inside stream_handler(), so a capture tool
    cannot reach the board at the same time. Workable via /claim, but two
    processes fighting over one slot is a bad foundation.

 2. Saving from a browser means drawing the frame to a canvas and
    exporting it, which RE-ENCODES the image. The exported JPEG carries
    the browser's quantization tables instead of the OV2640's -- and the
    sensor's own tables are the entire reason a captured frame is worth
    more to the DCT model than an Open Images photograph. Re-encoding
    here would silently destroy the thing being collected.

So: open the MJPEG stream once, in-process; decode each frame for DISPLAY
ONLY; and on a keypress write the bytes exactly as they came off the
socket. PIL never touches what lands on disk -- Image.open(...).save(...)
would be the same re-encode as the browser, just harder to notice.

What this buys over the headless path:
  - one connection, no per-frame request, preview at the stream's real rate
  - you see what you are capturing, so scenes can be composed deliberately
  - the board's live prediction sits beside the preview, so the frames the
    model gets WRONG -- the highest-value data -- can be captured on sight

Class is chosen from a dropdown (or the number keys, or by typing a new
name into it and pressing Enter). The list fills itself from the class
names the flashed model reports in /status.

Keys: SPACE save . A auto-capture . U undo last . 1-9 pick class
      C re-claim the stream . Q quit
"""

from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Imported rather than passed in. Running `capture_board_frames.py` as a
# script makes it __main__, so this creates a second module object for the
# same file -- harmless, because everything imported here is a constant or
# a pure function and the module has no import-time side effects.
from capture_board_frames import (
    MultipartStreamReader,
    claim_stream,
    next_index,
    top_scores,
)

# Colours. Dark, because the preview is a small bright rectangle and a dark
# surround stops the eye adapting to the chrome instead of the image.
BG = "#16181c"
PANEL = "#1e2127"
FG = "#e6e8eb"
DIM = "#8b929c"
OK = "#5fd08a"
WARN = "#e8a33d"
BAD = "#e8615a"


class _StreamWorker(threading.Thread):
    """Holds the one connection to :81/stream and keeps the newest frame.

    Latest-wins, not a queue: a queue would build a backlog and the preview
    would drift behind reality, which is exactly wrong for a tool whose job
    is deciding whether the scene in front of the camera is worth saving.
    """

    daemon = True

    def __init__(self, host: str, port: int, timeout: float):
        super().__init__(name="stream")
        self._host, self._port, self._timeout = host, port, timeout
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame = None          # bytes, exactly as received
        self._seq = 0               # per-frame identity, so one frame cannot be saved twice
        self._times = deque(maxlen=30)
        self.state = "connecting"
        self.stream_status = None   # /stream carries a status part ~1/sec

    # -- reader side -------------------------------------------------
    def run(self) -> None:
        url = f"http://{self._host}:{self._port}/stream"
        while not self._stop.is_set():
            resp = None
            try:
                # Take the slot before connecting, and again on every
                # reconnect: a browser tab reopened mid-session would
                # otherwise hold the stream and this would never recover.
                claim_stream(self._host, self._timeout)
                time.sleep(0.3)
                resp = urllib.request.urlopen(url, timeout=self._timeout)
                reader = MultipartStreamReader(resp)
                self.state = "streaming"
                while not self._stop.is_set():
                    ctype, payload = reader.next_part()
                    if "image/jpeg" in ctype:
                        # Same integrity check as the headless path: a
                        # truncated frame that reaches disk quietly poisons
                        # the dataset months later.
                        if not (payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")):
                            continue
                        with self._lock:
                            self._frame = payload
                            self._seq += 1
                            self._times.append(time.time())
                    elif "application/json" in ctype:
                        try:
                            self.stream_status = json.loads(payload)
                        except json.JSONDecodeError:
                            pass
            except Exception as e:            # noqa: BLE001 -- any transport fault reconnects
                if self._stop.is_set():
                    break
                self.state = f"reconnecting ({type(e).__name__})"
                time.sleep(1.5)
            finally:
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:         # noqa: BLE001
                        pass

    # -- consumer side -----------------------------------------------
    def latest(self) -> tuple:
        with self._lock:
            return self._frame, self._seq

    def fps(self) -> float:
        with self._lock:
            t = list(self._times)
        if len(t) < 2:
            return 0.0
        span = t[-1] - t[0]
        return (len(t) - 1) / span if span > 0 else 0.0

    def stop(self) -> None:
        self._stop.set()

    def reclaim(self) -> None:
        threading.Thread(target=claim_stream, args=(self._host, self._timeout), daemon=True).start()


class _StatusWorker(threading.Thread):
    """Polls /status on port 80 for the live prediction.

    Port 80 is a separate httpd instance with its own task, so it keeps
    answering while port 81 sits inside stream_handler(). The stream itself
    carries a status part, but only about once a second; polling here is
    fresher, and is what makes 'capture the frames it gets wrong' practical.
    """

    daemon = True

    def __init__(self, host: str, interval: float, timeout: float):
        super().__init__(name="status")
        self._host, self._interval, self._timeout = host, interval, timeout
        self._stop = threading.Event()
        self.status = None
        self.error = None

    def run(self) -> None:
        url = f"http://{self._host}/status"
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(url, timeout=self._timeout) as r:
                    self.status = json.loads(r.read())
                self.error = None
            except Exception as e:            # noqa: BLE001
                self.error = f"{type(e).__name__}"
            self._stop.wait(self._interval)

    def stop(self) -> None:
        self._stop.set()


class CaptureWindow:
    def __init__(self, args):
        self.args = args
        self.out_root = Path(args.out_dir)
        self.scale = max(1, int(args.scale))
        self.interval = args.interval

        self.stream = _StreamWorker(args.host, args.port, args.timeout)
        self.status = _StatusWorker(args.host, args.status_interval, args.timeout)

        self.labels = [args.label] + [s.strip() for s in args.labels.split(",") if s.strip()]
        self.labels = list(dict.fromkeys(self.labels))     # de-dup, keep order
        self._labels_from_board = not args.labels          # fill from /status once, later
        self.label = args.label

        self.saved = {}          # label -> count saved this session
        self.history = []        # paths, newest last, for undo
        self.next_idx = {}       # label -> next free frame number
        self.auto = False
        self.last_auto_save = 0.0
        self.last_saved_seq = -1
        self.message = ("ready", DIM)

        self._build_ui()

    # -- UI ----------------------------------------------------------
    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title(f"board capture - {self.args.host}")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        # ttk widgets ignore the plain bg/fg options the rest of this window
        # uses, and the default theme's colours are unreadable on a dark
        # panel. "clam" is the one stock theme that honours these settings.
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Capture.TCombobox", fieldbackground=PANEL, background=PANEL,
                        foreground=FG, arrowcolor=FG, bordercolor=DIM,
                        lightcolor=PANEL, darkcolor=PANEL, insertcolor=FG,
                        selectbackground=PANEL, selectforeground=FG, padding=4)
        style.map("Capture.TCombobox",
                  fieldbackground=[("readonly", PANEL), ("focus", PANEL)],
                  foreground=[("disabled", DIM)])
        # The dropdown list is a classic Tk listbox inside a toplevel, not a
        # ttk widget, so it is styled through the option database instead.
        self.root.option_add("*TCombobox*Listbox.background", PANEL)
        self.root.option_add("*TCombobox*Listbox.foreground", FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", OK)
        self.root.option_add("*TCombobox*Listbox.selectForeground", BG)
        self.root.option_add("*TCombobox*Listbox.font", "TkDefaultFont 12")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=12)

        # A placeholder image, not width=/height= options: on a Label with
        # no image those are CHARACTERS and LINES, so 160*scale by
        # 120*scale asks for a window about 1850x1010 and the window
        # manager keeps that size even after the first real frame shrinks
        # the widget. Sizing it with an actual image gets pixels.
        self._placeholder = ImageTk.PhotoImage(
            Image.new("RGB", (160 * self.scale, 120 * self.scale), "#0b0d10"))
        self.canvas = tk.Label(body, bg="#000000", bd=0, image=self._placeholder)
        self.canvas.grid(row=0, column=0, sticky="nw")
        self._photo = None

        side = tk.Frame(body, bg=PANEL, padx=14, pady=12)
        side.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        def head(text):
            return tk.Label(side, text=text, bg=PANEL, fg=DIM,
                            font=("TkDefaultFont", 9), anchor="w")

        head("SAVING AS").pack(fill="x")

        # A dropdown rather than a label, because the class changes far more
        # often than anything else in this window -- you point the camera at
        # a different thing and keep going. It is EDITABLE: typing a name
        # that is not in the list and pressing Enter adds it, so a class the
        # flashed model does not know (which is most of the interesting
        # ones, when collecting data for a class you are about to add) does
        # not mean restarting with --labels.
        self.label_var = tk.StringVar(value=self.label)
        self.combo = ttk.Combobox(side, textvariable=self.label_var,
                                  values=self.labels, style="Capture.TCombobox",
                                  font=("TkDefaultFont", 16, "bold"), width=14)
        self.combo.pack(fill="x", pady=(2, 2))
        self.combo.bind("<<ComboboxSelected>>", self._on_combo_selected)
        self.combo.bind("<Return>", self._on_combo_entered)

        self.lbl_count = tk.Label(side, text="", bg=PANEL, fg=DIM,
                                  font=("TkFixedFont", 9), anchor="w", justify="left")
        self.lbl_count.pack(fill="x", pady=(0, 14))

        head("BOARD SAYS").pack(fill="x")
        self.lbl_pred = tk.Label(side, text="-", bg=PANEL, fg=FG,
                                 font=("TkDefaultFont", 16, "bold"), anchor="w")
        self.lbl_pred.pack(fill="x")
        self.lbl_scores = tk.Label(side, text="", bg=PANEL, fg=DIM,
                                   font=("TkFixedFont", 10), anchor="w", justify="left")
        self.lbl_scores.pack(fill="x", pady=(0, 14))

        head("CLASSES").pack(fill="x")
        self.lbl_classes = tk.Label(side, text="", bg=PANEL, fg=FG,
                                    font=("TkFixedFont", 10), anchor="w", justify="left")
        self.lbl_classes.pack(fill="x", pady=(0, 14))

        self.lbl_link = tk.Label(side, text="", bg=PANEL, fg=DIM,
                                 font=("TkFixedFont", 9), anchor="w", justify="left")
        self.lbl_link.pack(fill="x", side="bottom")

        foot = tk.Frame(self.root, bg=BG)
        foot.pack(fill="x", padx=12, pady=(0, 10))
        self.lbl_msg = tk.Label(foot, text="", bg=BG, fg=DIM,
                                font=("TkFixedFont", 10), anchor="w")
        self.lbl_msg.pack(fill="x")
        tk.Label(foot, bg=BG, fg=DIM, font=("TkFixedFont", 9), anchor="w",
                 text="SPACE save   A auto-capture   U undo last   1-9 class "
                      "(or the dropdown)   C re-claim stream   Q quit").pack(fill="x")

        self.combo.bind("<FocusOut>", self._on_combo_focus_out)
        # bind_all, not bind: the combobox would otherwise swallow every
        # keypress once it has been clicked, and the shortcuts would look
        # broken for the rest of the session.
        self.root.bind_all("<KeyPress>", self._on_key)

    # -- actions -----------------------------------------------------
    def _dir_for(self, label: str) -> Path:
        d = self.out_root / label
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_current(self, auto: bool = False) -> None:
        frame, seq = self.stream.latest()
        if frame is None:
            self.message = ("no frame yet -- waiting for the stream", WARN)
            return
        if seq == self.last_saved_seq:
            # The same frame twice is a duplicate file, not a second sample.
            if not auto:
                self.message = ("that frame is already saved -- wait for the next one", WARN)
            return

        label = self.label
        d = self._dir_for(label)
        if label not in self.next_idx:
            self.next_idx[label] = next_index(d, label)
        idx = self.next_idx[label]
        path = d / f"{label}_{idx:06d}.jpg"

        # Straight from the socket to the file. Not via PIL: a decode/encode
        # round-trip would replace the OV2640's quantization tables with
        # libjpeg's and quietly destroy the reason for capturing at all.
        path.write_bytes(frame)

        self.next_idx[label] = idx + 1
        self.saved[label] = self.saved.get(label, 0) + 1
        self.history.append(path)
        self.last_saved_seq = seq
        self.message = (f"saved {path.name}  ({len(frame)} bytes)", OK)

    def undo_last(self) -> None:
        if not self.history:
            self.message = ("nothing to undo this session", WARN)
            return
        path = self.history.pop()
        label = path.parent.name
        try:
            path.unlink()
        except OSError as e:
            self.message = (f"could not remove {path.name}: {e}", BAD)
            return
        self.saved[label] = max(0, self.saved.get(label, 1) - 1)
        # Hand the number back so the next save reuses it rather than
        # leaving a gap; next_index() would find it free again anyway.
        if self.next_idx.get(label):
            self.next_idx[label] -= 1
        self.last_saved_seq = -1
        self.message = (f"removed {path.name}", WARN)

    def set_label(self, name: str) -> None:
        """The one place self.label changes, so the dropdown, the number
        keys and the tally can never disagree about the current class."""
        name = name.strip()
        if not name or name == self.label:
            self.label_var.set(self.label)
            return
        if name not in self.labels:
            self.labels.append(name)
            self.combo.configure(values=self.labels)
        self.label = name
        self.label_var.set(name)
        self.message = (f"saving as {name}", DIM)

    def select_label(self, i: int) -> None:
        if 0 <= i < len(self.labels):
            self.set_label(self.labels[i])

    def _on_combo_selected(self, _event) -> None:
        self.set_label(self.label_var.get())
        # Hand focus back, or SPACE would type a space into the box
        # instead of saving a frame.
        self.root.focus_set()

    def _on_combo_entered(self, _event) -> None:
        """Enter commits a typed name -- including one the board has never
        heard of, which is the case when collecting data for a class the
        model does not have yet."""
        name = self.label_var.get().strip()
        if name and name not in self.labels:
            self.message = (f"added class {name}", OK)
        self.set_label(name)
        self.root.focus_set()

    def _on_combo_focus_out(self, _event) -> None:
        # Half-typed text left in the box would otherwise look like the
        # current class while saves went somewhere else.
        if self.label_var.get() != self.label:
            self.label_var.set(self.label)

    def _on_key(self, event) -> None:
        # While the dropdown has focus it is a text field: SPACE, A, U and Q
        # are characters being typed, not commands. Escape gives focus back.
        if event.widget is self.combo:
            if event.keysym.lower() == "escape":
                self.label_var.set(self.label)
                self.root.focus_set()
            return
        k = event.keysym.lower()
        if k == "space":
            self.save_current()
        elif k == "a":
            self.auto = not self.auto
            self.last_auto_save = 0.0
            self.message = (f"auto-capture {'ON' if self.auto else 'off'} "
                            f"(every {self.interval}s)", OK if self.auto else DIM)
        elif k == "u":
            self.undo_last()
        elif k == "c":
            self.stream.reclaim()
            self.message = ("re-claimed the stream slot", DIM)
        elif k in ("q", "escape"):
            self.quit()
        elif len(k) == 1 and k.isdigit() and k != "0":
            self.select_label(int(k) - 1)

    # -- render loop -------------------------------------------------
    def _tick(self) -> None:
        frame, seq = self.stream.latest()
        if frame is not None and seq != getattr(self, "_shown_seq", None):
            try:
                img = Image.open(io.BytesIO(frame))
                img.load()
                if self.scale != 1:
                    img = img.resize((img.width * self.scale, img.height * self.scale),
                                     Image.NEAREST)
                self._photo = ImageTk.PhotoImage(img)
                self.canvas.configure(image=self._photo)
                self._shown_seq = seq
            except Exception as e:            # noqa: BLE001 -- a bad frame must not kill the UI
                self.message = (f"decode failed: {type(e).__name__}", BAD)

        status = self.status.status or self.stream.stream_status
        scores = top_scores(status, 3)

        # Fill the class selector from the board on the first status that
        # carries scores, so the number keys work without --labels.
        if self._labels_from_board and status and status.get("scores"):
            board = sorted(str(s.get("class")) for s in status["scores"] if isinstance(s, dict))
            added = [n for n in board if n not in self.labels]
            if added:
                self.labels.extend(added)
                self.combo.configure(values=self.labels)
            self._labels_from_board = False

        if scores:
            top, pct = scores[0]
            self.lbl_pred.configure(text=f"{top}  {pct:.0f}%",
                                    fg=OK if top == self.label else WARN)
            self.lbl_scores.configure(
                text="\n".join(f"{n:<12} {p:5.1f}%" for n, p in scores))
        else:
            self.lbl_pred.configure(text="-", fg=DIM)
            self.lbl_scores.configure(text="no classification yet")

        n_here = self.saved.get(self.label, 0)
        on_disk = self.next_idx.get(self.label)
        self.lbl_count.configure(
            text=f"{n_here} saved this session"
                 + (f", {on_disk} in the folder" if on_disk is not None else ""))

        self.lbl_classes.configure(text="\n".join(
            f"{'>' if n == self.label else ' '} {i + 1}  {n:<14}{self.saved.get(n, 0):>4}"
            for i, n in enumerate(self.labels[:9])))

        board_fps = (status or {}).get("fps")
        self.lbl_link.configure(
            text=f"{self.stream.state}\n"
                 f"preview {self.stream.fps():4.1f} fps"
                 + (f"   board {board_fps:.1f} fps" if isinstance(board_fps, (int, float)) else "")
                 + (f"\nstatus: {self.status.error}" if self.status.error else "")
                 + (f"\nAUTO every {self.interval}s" if self.auto else ""))

        if self.auto and time.time() - self.last_auto_save >= self.interval:
            self.save_current(auto=True)
            self.last_auto_save = time.time()

        text, colour = self.message
        self.lbl_msg.configure(text=text, fg=colour)

        self.root.after(33, self._tick)      # ~30 Hz; the board sends fewer

    def quit(self) -> None:
        self.stream.stop()
        self.status.stop()
        self.root.destroy()

    def run(self) -> None:
        self.stream.start()
        self.status.start()
        self.root.after(50, self._tick)
        self.root.mainloop()
        total = sum(self.saved.values())
        print(f"\nsaved {total} frame(s) this session")
        for label, n in sorted(self.saved.items()):
            if n:
                print(f"  {label:<16} {n:>4}   -> {self.out_root / label}")
        if total:
            print("\nCopy into data/train/<class>/ with a distinct prefix -- capture "
                  "filenames collide\nwith Open Images ones. Copy, never move: "
                  "board_captures/ is the only copy.")


def run_interactive(args) -> None:
    CaptureWindow(args).run()
