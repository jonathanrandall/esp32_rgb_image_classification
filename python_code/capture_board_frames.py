#!/usr/bin/env python3
"""Capture real JPEG frames from the ESP32 board into a labelled dataset.

WHY THIS EXISTS. Every model in this project is trained on Open Images
photographs: padded crops around one annotated object, letterboxed to
160x120 and re-encoded. The camera sees something quite different -- a
whole room, the subject small and off-centre, indoor lighting, the
OV2640's own exposure and white balance, and the sensor's own JPEG
quantization tables. A model can score well on the test split and behave
badly on the board without either number being wrong, because the two
distributions are not the same.

The `people` class shows this most sharply. The training pipeline drops
person-containing images from every class except `people`
(filter_intersections.py), so the model has never seen a person and a
computer in one frame -- which is exactly what a camera pointed at a desk
produces. Frames captured here carry the real co-occurrence, the real
framing and the real encoder, so they are the only data that closes that
gap directly.

The board serves `/frame` on the stream port: ONE multipart response
containing a JPEG part followed by a status JSON part, then it ends. That
is deliberately bounded, unlike `/stream` which never returns -- so
polling it does not hold the board's single stream slot open.

Images land at the board's native capture size (160x120, 4:2:2), already
the training resolution, so build_data.py's resize step is a no-op on
them and no re-encode is needed.

Usage:
    # capture 200 frames of you sitting at your desk
    python capture_board_frames.py --label computer_people --count 200

    # slower, for moving the camera around between shots
    python capture_board_frames.py --label people --count 100 --interval 1.0

    # a different board
    python capture_board_frames.py --host 192.168.1.50 --label car --count 50

Frames are written to ../board_captures/<label>/<label>_<n>.jpg, numbered
so a later run appends rather than overwriting. Point build_data.py's
--source at a directory of these, or merge them into an existing dataset.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "board_captures"

# Must match main.cpp's PART_BOUNDARY. If the firmware's boundary ever
# changes, this is the one thing here that breaks.
PART_BOUNDARY = b"123456789000000000000987654321"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="esp32cam_dct.local",
                   help="board hostname or IP (default: esp32cam_dct.local, the classifier's mDNS name)")
    p.add_argument("--port", type=int, default=81,
                   help="stream port, where /frame lives (default: 81)")
    p.add_argument("--label", required=True,
                   help="class name -- becomes the output subdirectory and the filename prefix")
    p.add_argument("--count", type=int, default=100, help="frames to capture (default: 100)")
    p.add_argument("--interval", type=float, default=0.5,
                   help="seconds between captures (default: 0.5). Raise it if you are moving the "
                        "camera or the subject between frames -- consecutive frames of a static "
                        "scene are near-duplicates and add little.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT),
                   help=f"output root (default: {DEFAULT_OUT})")
    p.add_argument("--timeout", type=float, default=10.0, help="per-request timeout in seconds (default: 10)")
    p.add_argument("--show-prediction", action="store_true",
                   help="print the board's own top class for each frame, read from the status part. "
                        "Useful for spotting exactly which live scenes the current model gets wrong.")
    return p.parse_args()


def split_multipart(body: bytes) -> list:
    """Return [(headers_bytes, payload_bytes), ...] for each part.

    Hand-rolled rather than via `email` or `requests_toolbelt`: the
    response is tiny and rigidly formatted by frame_handler(), and this
    script is meant to run with nothing installed beyond the stdlib."""
    parts = []
    sep = b"--" + PART_BOUNDARY
    for chunk in body.split(sep):
        head, _, payload = chunk.partition(b"\r\n\r\n")
        if not payload:
            continue
        # The trailing CRLF before the next boundary is framing, not data.
        parts.append((head, payload.rstrip(b"\r\n")))
    return parts


def claim_stream(host: str, timeout: float) -> bool:
    """Ask the incumbent stream client to let go, via GET /claim on port 80.

    The board serves ONE viewer: port 81's single httpd task sits inside
    stream_handler() for as long as a client is connected, so /frame -- on
    that same port -- is never reached while a browser tab (or a stale,
    still-open socket from a closed one) holds it. The symptom is a plain
    timeout, which looks identical to a dead board.

    /claim lives on port 80, whose handlers all return promptly, so it stays
    answerable exactly when port 81 cannot hear anything. Firmware older
    than 2026-09-04 has no /claim; a 404 there is not an error, it just
    means the tab has to be closed by hand."""
    try:
        with urllib.request.urlopen(f"http://{host}/claim", timeout=timeout) as r:
            r.read()
        return True
    except urllib.error.HTTPError:
        return False        # old firmware, no /claim
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def fetch_frame(url: str, timeout: float) -> tuple:
    """(jpeg_bytes, status_dict_or_None). Raises on transport failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise SystemExit(
                f"{url} returned 404 -- no /frame endpoint.\n\n"
                "This tool targets the DCT firmware (esp32_cam/esp32_classifier), which is the\n"
                "only one that serves /frame AND the only one whose JPEGs come from the OV2640's\n"
                "HARDWARE encoder. esp32_rgb_cnn captures RGB565 and software-encodes a preview\n"
                "with frame2jpg(), so its frames carry the software encoder's quantization\n"
                "tables rather than the sensor's -- wrong data for DCT training.\n\n"
                "You do not need to capture from both: a hardware JPEG serves BOTH models.\n"
                "train_cnn.py reads its coefficients directly; train_rgb_cnn.py decodes it to\n"
                "pixels. The reverse does not work.\n\n"
                "Flash esp32_classifier and use --host esp32cam_dct.local (the default)."
            ) from None
        raise
    jpeg, status = None, None
    for head, payload in split_multipart(body):
        if b"image/jpeg" in head and jpeg is None:
            jpeg = payload
        elif b"application/json" in head and status is None:
            try:
                status = json.loads(payload)
            except json.JSONDecodeError:
                pass
    if jpeg is None:
        raise ValueError(f"no image/jpeg part in {len(body)} byte response")
    # Cheap integrity check: a truncated capture is worse than a failed one,
    # because it lands on disk and quietly poisons the dataset later.
    if not (jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")):
        raise ValueError(f"JPEG missing SOI/EOI marker ({len(jpeg)} bytes) -- truncated capture")
    return jpeg, status


def top_class(status: dict) -> str:
    """Board's own current prediction, if the status part carries one."""
    if not isinstance(status, dict):
        return "?"
    for key in ("scores", "classes", "predictions"):
        v = status.get(key)
        if isinstance(v, list) and v:
            first = v[0]
            if isinstance(first, dict):
                return str(first.get("name", first.get("class", "?")))
            return str(first)
    return str(status.get("class", status.get("top", "?")))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Continue the numbering rather than overwriting, so several sessions
    # (different rooms, different lighting) accumulate into one class.
    existing = sorted(out_dir.glob(f"{args.label}_*.jpg"))
    start = 0
    if existing:
        nums = [int(p.stem.rsplit("_", 1)[1]) for p in existing if p.stem.rsplit("_", 1)[1].isdigit()]
        start = max(nums) + 1 if nums else len(existing)

    # Take the stream slot before the first request, and again after any
    # failure -- a browser reconnecting mid-run would otherwise take it back
    # and stall every remaining capture.
    if claim_stream(args.host, args.timeout):
        print("claimed the stream slot (any open viewer will have been dropped)")
        time.sleep(0.3)
    else:
        print("note: /claim unavailable (old firmware?) -- close any open camera tab "
              "if captures time out")

    url = f"http://{args.host}:{args.port}/frame"
    print(f"capturing {args.count} frames from {url}")
    print(f"  -> {out_dir}  (starting at index {start}, {len(existing)} already there)")
    print(f"  interval {args.interval}s\n")

    saved = failed = 0
    consecutive_failures = 0
    for i in range(args.count):
        idx = start + i
        try:
            jpeg, status = fetch_frame(url, args.timeout)
        except (urllib.error.URLError, ValueError, OSError, TimeoutError) as e:
            failed += 1
            consecutive_failures += 1
            print(f"  [{i+1}/{args.count}] FAILED: {type(e).__name__}: {e}")
            # Something is wrong with the board or the link, not this frame.
            if consecutive_failures >= 10:
                print("\n10 consecutive failures -- stopping. Check the board is up "
                      "(curl http://%s/status) and that nothing else holds the stream." % args.host)
                break
            # Most likely cause is a viewer having taken the slot back.
            claim_stream(args.host, args.timeout)
            time.sleep(max(args.interval, 1.0))
            continue
        consecutive_failures = 0
        path = out_dir / f"{args.label}_{idx:06d}.jpg"
        path.write_bytes(jpeg)
        saved += 1
        note = f"  board says: {top_class(status)}" if args.show_prediction else ""
        print(f"  [{i+1}/{args.count}] {path.name}  {len(jpeg):6} bytes{note}")
        if i + 1 < args.count:
            time.sleep(args.interval)

    print(f"\nsaved {saved}, failed {failed} -> {out_dir}")
    if saved:
        sizes = [p.stat().st_size for p in out_dir.glob("*.jpg")]
        print(f"{len(sizes)} total in this class, mean {sum(sizes)//len(sizes)} bytes")
        print("\nNext: capture the other classes, then vary the scene -- different rooms,")
        print("lighting and distances. Near-duplicate frames of one static scene inflate")
        print("the count without adding information.")
    if failed and not saved:
        sys.exit(1)


if __name__ == "__main__":
    main()
