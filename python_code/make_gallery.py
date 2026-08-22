#!/usr/bin/env python3

"""
Generate a local, click-to-reject HTML contact-sheet for eyeballing a
directory of dataset images -- built because browsing thousands of
small crops one at a time in a standard image viewer doesn't scale,
and Ubuntu's stock file managers don't do fast thumbnail-grid review
nearly as well as Windows Explorer does.

Click a thumbnail to mark it rejected (red border, dimmed). "Export
rejected list" downloads a .txt of filenames (one per line, sorted) --
hand that file back and the listed files can be pulled out of the
dataset. Marks are also kept in the browser's localStorage as a
convenience so a reload doesn't lose them, but that's best-effort only
-- some browsers (Chrome in particular) give each file:// page its own
opaque origin, which can break localStorage persistence across
reloads. Export before closing the tab if the marks matter.

No server needed -- opens directly as a local file.

Usage:
    python make_gallery.py <image_dir> [<image_dir> ...]

Writes <image_dir>/_gallery.html for each directory given, referencing
the images already in that directory by filename.
"""

import sys
from pathlib import Path

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}

TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0; background: #1b1b1f; color: #e8e8ec; }}
  header {{ position: sticky; top: 0; z-index: 10; background: #232329; padding: 10px 16px; display: flex; gap: 16px; align-items: center; border-bottom: 1px solid #34343c; flex-wrap: wrap; }}
  header h1 {{ font-size: 15px; margin: 0; font-weight: 600; color: #f2f2f5; }}
  header .stat {{ font-size: 13px; color: #a8a8b3; }}
  header .stat b {{ color: #f2f2f5; }}
  button {{ background: #3a3a45; color: #f2f2f5; border: 1px solid #4a4a56; border-radius: 6px; padding: 6px 12px; font-size: 13px; cursor: pointer; }}
  button:hover {{ background: #464652; }}
  button.danger {{ background: #5a2a2a; border-color: #7a3a3a; }}
  button.danger:hover {{ background: #6a3232; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 6px; padding: 12px; }}
  .cell {{ cursor: pointer; border: 3px solid transparent; border-radius: 6px; overflow: hidden; background: #26262c; transition: border-color .1s; }}
  .cell img {{ display: block; width: 100%; aspect-ratio: {aspect}; object-fit: cover; }}
  .cell .fname {{ font-size: 10px; color: #8a8a96; padding: 2px 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .cell.rejected {{ border-color: #e05252; }}
  .cell.rejected img {{ opacity: 0.35; }}
  .cell.rejected .fname {{ color: #e05252; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <span class="stat"><b id="total">{count}</b> images</span>
  <span class="stat"><b id="rejcount">0</b> rejected</span>
  <button id="exportBtn">Export rejected list</button>
  <button id="clearBtn" class="danger">Clear marks</button>
</header>
<div class="grid" id="grid">
{cells}
</div>
<script>
const STORAGE_KEY = "gallery_rejects::{key}";
let rejected = new Set();
try {{
  rejected = new Set(JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"));
}} catch (e) {{}}

function persist() {{
  try {{ localStorage.setItem(STORAGE_KEY, JSON.stringify([...rejected])); }} catch (e) {{}}
  document.getElementById("rejcount").textContent = rejected.size;
}}

document.querySelectorAll(".cell").forEach(cell => {{
  const fname = cell.dataset.fname;
  if (rejected.has(fname)) cell.classList.add("rejected");
  cell.addEventListener("click", () => {{
    if (rejected.has(fname)) {{ rejected.delete(fname); cell.classList.remove("rejected"); }}
    else {{ rejected.add(fname); cell.classList.add("rejected"); }}
    persist();
  }});
}});

document.getElementById("exportBtn").addEventListener("click", () => {{
  const text = [...rejected].sort().join("\\n") + (rejected.size ? "\\n" : "");
  const blob = new Blob([text], {{type: "text/plain"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "rejected_{key}.txt";
  a.click();
}});

document.getElementById("clearBtn").addEventListener("click", () => {{
  if (!confirm("Clear all rejection marks on this page?")) return;
  rejected.clear();
  document.querySelectorAll(".cell.rejected").forEach(c => c.classList.remove("rejected"));
  persist();
}});

persist();
</script>
</body>
</html>
"""

CELL_TEMPLATE = '<div class="cell" data-fname="{fname}"><img src="{fname}" loading="lazy"><div class="fname">{fname}</div></div>'


def make_gallery(image_dir: Path) -> Path | None:
    files = sorted(
        p.name for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )
    if not files:
        print(f"  no images found in {image_dir}, skipping")
        return None

    cells = "\n".join(CELL_TEMPLATE.format(fname=f) for f in files)
    resolved = image_dir.resolve()
    key_parts = resolved.parts[-3:] if len(resolved.parts) >= 3 else resolved.parts
    key = "_".join(key_parts)
    title = "/".join(key_parts) + f" -- {len(files)} images"

    html = TEMPLATE.format(title=title, count=len(files), cells=cells, key=key, aspect="4/3")
    out_path = image_dir / "_gallery.html"
    out_path.write_text(html)
    print(f"  wrote {out_path} ({len(files)} images)")
    return out_path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        image_dir = Path(arg)
        if not image_dir.is_dir():
            print(f"  not a directory, skipping: {image_dir}")
            continue
        make_gallery(image_dir)


if __name__ == "__main__":
    main()
