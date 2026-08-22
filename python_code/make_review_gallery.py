#!/usr/bin/env python3
"""One gallery for a whole review batch, with a folder dropdown.

make_gallery.py writes one _gallery.html per directory, which means one
tab (and one export click) per class per split -- 18 of each for a
9-class batch. This writes a SINGLE page at curation_review/_review.html
that switches between folders from a dropdown.

The folder listing is baked in at generation time. A file:// page cannot
enumerate directories, so re-run this script after every
curation_pull.py.

WHAT IT EXPORTS. Exactly the same per-folder files
`rejected_curation_review_<split>_<class>.txt` that make_gallery.py
produces, so curation_resolve.py reads them unchanged. "Export all
reviewed folders" writes one file per folder marked done -- INCLUDING
empty ones for folders with nothing wrong, which is what tells
curation_resolve.py "this was reviewed and everything passed" rather
than "not reviewed yet". Creating those empty files by hand is otherwise
the most tedious part of the loop.

Marking a folder done is explicit (the "Mark reviewed" button), not
implied by visiting it -- clicking through the dropdown to see what is
there should not silently accept a batch.

Browsers block multi-file downloads by default. The first "Export all"
raises a permission prompt; allow it once per origin. If downloads still
fail, use "Export current folder" per folder, or "Copy all as text" and
save the sections by hand.

Usage:
    python make_review_gallery.py                      # curation_review/
    python make_review_gallery.py some/other/root      # any <split>/<class> tree
"""

import json
import sys
from pathlib import Path

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0;
         background: #1b1b1f; color: #e8e8ec; }
  header { position: sticky; top: 0; z-index: 10; background: #232329; padding: 10px 16px;
           display: flex; gap: 12px; align-items: center; border-bottom: 1px solid #34343c;
           flex-wrap: wrap; }
  select { background: #2c2c34; color: #f2f2f5; border: 1px solid #4a4a56; border-radius: 6px;
           padding: 6px 10px; font-size: 13px; max-width: 340px; }
  .stat { font-size: 13px; color: #a8a8b3; }
  .stat b { color: #f2f2f5; font-variant-numeric: tabular-nums; }
  button { background: #3a3a45; color: #f2f2f5; border: 1px solid #4a4a56; border-radius: 6px;
           padding: 6px 12px; font-size: 13px; cursor: pointer; }
  button:hover { background: #464652; }
  button.primary { background: #2f5d3a; border-color: #3f7a4d; }
  button.primary:hover { background: #376d44; }
  button.danger { background: #5a2a2a; border-color: #7a3a3a; }
  button.danger:hover { background: #6a3232; }
  button:disabled { opacity: .45; cursor: default; }
  .spacer { flex: 1; }
  .done-badge { font-size: 12px; padding: 3px 8px; border-radius: 999px; background: #2f5d3a;
                color: #cdebd5; }
  .done-badge.pending { background: #4a4a56; color: #c8c8d2; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 6px;
          padding: 12px; }
  .cell { cursor: pointer; border: 3px solid transparent; border-radius: 6px; overflow: hidden;
          background: #26262c; transition: border-color .1s; }
  .cell img { display: block; width: 100%; aspect-ratio: 4/3; object-fit: cover; }
  .cell .fname { font-size: 10px; color: #8a8a96; padding: 2px 4px; white-space: nowrap;
                 overflow: hidden; text-overflow: ellipsis; }
  .cell.rejected { border-color: #e05252; }
  .cell.rejected img { opacity: 0.35; }
  .cell.rejected .fname { color: #e05252; }
  #dump { display: none; width: 100%; height: 60vh; background: #16161a; color: #d8d8e0;
          border: 1px solid #34343c; font-family: ui-monospace, monospace; font-size: 12px;
          padding: 10px; }
</style>
</head>
<body>
<header>
  <select id="folderSel"></select>
  <button id="prevBtn" title="Previous folder">&#8592;</button>
  <button id="nextBtn" title="Next folder">&#8594;</button>
  <span class="stat"><b id="rejcount">0</b> / <b id="total">0</b> rejected</span>
  <span id="doneBadge" class="done-badge pending">not reviewed</span>
  <button id="doneBtn" class="primary">Mark reviewed</button>
  <span class="spacer"></span>
  <span class="stat"><b id="progress">0</b> of <b id="nfolders">0</b> folders reviewed</span>
  <button id="exportOneBtn">Export current folder</button>
  <button id="exportAllBtn">Export all reviewed</button>
  <button id="dumpBtn">Copy all as text</button>
  <button id="clearBtn" class="danger">Clear this folder</button>
</header>
<textarea id="dump" spellcheck="false"></textarea>
<div class="grid" id="grid"></div>
<script>
const FOLDERS = __FOLDERS__;     // [{key, split, cls, files:[...]}]
const ROOT_KEY = "__ROOTKEY__";
const SKEY = "review_gallery::" + ROOT_KEY;

let state = {};   // key -> {rejected:[...], done:bool}
try { state = JSON.parse(localStorage.getItem(SKEY) || "{}"); } catch (e) {}
for (const f of FOLDERS) if (!state[f.key]) state[f.key] = {rejected: [], done: false};

let idx = 0;
const sel = document.getElementById("folderSel");
const grid = document.getElementById("grid");

function persist() {
  try { localStorage.setItem(SKEY, JSON.stringify(state)); } catch (e) {}
}

function labelFor(f) {
  const st = state[f.key];
  const mark = st.done ? "\\u2713 " : "  ";
  const rej = st.rejected.length ? "  (" + st.rejected.length + " rejected)" : "";
  return mark + f.split + "/" + f.cls + "  [" + f.files.length + "]" + rej;
}

function refreshOptions() {
  const keep = sel.selectedIndex;
  sel.innerHTML = "";
  FOLDERS.forEach((f, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = labelFor(f);
    sel.appendChild(o);
  });
  sel.selectedIndex = keep < 0 ? 0 : keep;
  const nDone = FOLDERS.filter(f => state[f.key].done).length;
  document.getElementById("progress").textContent = nDone;
  document.getElementById("nfolders").textContent = FOLDERS.length;
}

function render() {
  const f = FOLDERS[idx];
  const st = state[f.key];
  const rejected = new Set(st.rejected);
  grid.innerHTML = "";
  const frag = document.createDocumentFragment();
  for (const name of f.files) {
    const cell = document.createElement("div");
    cell.className = "cell" + (rejected.has(name) ? " rejected" : "");
    const img = document.createElement("img");
    img.src = f.split + "/" + f.cls + "/" + name;
    img.loading = "lazy";
    const cap = document.createElement("div");
    cap.className = "fname"; cap.textContent = name;
    cell.appendChild(img); cell.appendChild(cap);
    cell.addEventListener("click", () => {
      const s = new Set(state[f.key].rejected);
      if (s.has(name)) { s.delete(name); cell.classList.remove("rejected"); }
      else { s.add(name); cell.classList.add("rejected"); }
      state[f.key].rejected = [...s];
      persist(); updateCounts(); refreshOptions();
    });
    frag.appendChild(cell);
  }
  grid.appendChild(frag);
  sel.selectedIndex = idx;
  updateCounts();
  window.scrollTo(0, 0);
}

function updateCounts() {
  const f = FOLDERS[idx], st = state[f.key];
  document.getElementById("total").textContent = f.files.length;
  document.getElementById("rejcount").textContent = st.rejected.length;
  const badge = document.getElementById("doneBadge");
  badge.textContent = st.done ? "reviewed" : "not reviewed";
  badge.className = "done-badge" + (st.done ? "" : " pending");
  document.getElementById("doneBtn").textContent = st.done ? "Unmark reviewed" : "Mark reviewed";
}

function go(i) { idx = (i + FOLDERS.length) % FOLDERS.length; render(); }

function download(name, text) {
  const blob = new Blob([text], {type: "text/plain"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

function listText(key) {
  const r = [...state[key].rejected].sort();
  return r.join("\\n") + (r.length ? "\\n" : "");
}

sel.addEventListener("change", () => go(parseInt(sel.value, 10)));
document.getElementById("prevBtn").addEventListener("click", () => go(idx - 1));
document.getElementById("nextBtn").addEventListener("click", () => go(idx + 1));
document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowLeft") go(idx - 1);
  if (e.key === "ArrowRight") go(idx + 1);
});

document.getElementById("doneBtn").addEventListener("click", () => {
  const f = FOLDERS[idx];
  state[f.key].done = !state[f.key].done;
  persist(); updateCounts(); refreshOptions();
});

document.getElementById("exportOneBtn").addEventListener("click", () => {
  const f = FOLDERS[idx];
  download("rejected_" + f.key + ".txt", listText(f.key));
});

document.getElementById("exportAllBtn").addEventListener("click", () => {
  const done = FOLDERS.filter(f => state[f.key].done);
  if (!done.length) { alert("No folders marked reviewed yet."); return; }
  if (!confirm(done.length + " file(s) will download, one per reviewed folder "
      + "(empty ones included -- that is what marks a clean batch as reviewed).\\n\\n"
      + "Your browser may ask permission to download multiple files.")) return;
  done.forEach((f, i) => setTimeout(
    () => download("rejected_" + f.key + ".txt", listText(f.key)), i * 250));
});

document.getElementById("dumpBtn").addEventListener("click", () => {
  const ta = document.getElementById("dump");
  if (ta.style.display === "block") { ta.style.display = "none"; return; }
  let out = "";
  for (const f of FOLDERS) {
    if (!state[f.key].done) continue;
    out += "=== rejected_" + f.key + ".txt ===\\n" + listText(f.key) + "\\n";
  }
  ta.value = out || "(no folders marked reviewed yet)";
  ta.style.display = "block";
  ta.focus(); ta.select();
});

document.getElementById("clearBtn").addEventListener("click", () => {
  const f = FOLDERS[idx];
  if (!confirm("Clear all rejection marks in " + f.split + "/" + f.cls + "?")) return;
  state[f.key].rejected = [];
  persist(); render(); refreshOptions();
});

refreshOptions();
render();
</script>
</body>
</html>
"""


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "curation_review"
    if not root.is_dir():
        raise SystemExit(f"{root} not found -- run curation_pull.py first")

    folders = []
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            files = sorted(p.name for p in class_dir.iterdir()
                           if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS)
            if not files:
                continue
            # Key must match make_gallery.py's: the directory's last three
            # path components joined by "_", which is what curation_resolve.py
            # looks for in ~/Downloads.
            key = f"{root.name}_{split_dir.name}_{class_dir.name}"
            folders.append({"key": key, "split": split_dir.name,
                            "cls": class_dir.name, "files": files})

    if not folders:
        raise SystemExit(f"no images found under {root}")

    n_img = sum(len(f["files"]) for f in folders)
    html = (TEMPLATE
            .replace("__FOLDERS__", json.dumps(folders))
            .replace("__ROOTKEY__", root.name)
            .replace("{title}", f"{root.name} -- {len(folders)} folders, {n_img} images"))

    out = root / "_review.html"
    out.write_text(html)
    print(f"wrote {out}")
    print(f"  {len(folders)} folders, {n_img} images")
    print()
    print("Open it, then for each folder: click the bad images, press 'Mark reviewed',")
    print("and move on with the dropdown or the arrow keys. When every folder is marked,")
    print("press 'Export all reviewed' and save the files to ~/Downloads.")
    print()
    print("Then: python python_code/curation_resolve.py")


if __name__ == "__main__":
    main()
