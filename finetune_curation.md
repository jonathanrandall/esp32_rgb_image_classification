# Hand curation for fine-tuning

Train the base model on all of `data/` untouched, then hand-curate a
subset and fine-tune on that.

Not to be confused with [`data_curation/README.md`](data_curation/README.md),
which is the *older* workflow: it curates the raw pull in place, before any
training. Same gallery tool and the same pull/review/resolve shape; different
dataset and different purpose. This one operates on `data/` and produces a
fine-tuning set.

Every script referenced here lives in [`python_code/`](python_code/); see its
[CLI reference](python_code/README.md#cli-reference) for every flag and
default.

## Directories

```
data/                built by build_data.py. NEVER modified by this workflow.
                     Supplies the test split.
data_tmp/            working pool. Created on first pull as a copy of
                     data/{train,val}. Images move OUT, so what remains is
                     exactly "not yet reviewed".
curation_review/     the batch currently under review. Empty between batches.
data_hand_curated/   accepted images, accumulating. This is the fine-tuning set.
data_rejected/       rejected images. Nothing is deleted.
```

`test/` is never copied into the pool and never reviewed — see
[Why the test set stays untouched](#why-the-test-set-stays-untouched).

## The loop

### 1. Pull a batch

```bash
cd <repo root>
python python_code/curation_pull.py --train-count 100 --val-count 20
```

First run also copies `data/{train,val}` into `data_tmp/` (~12,000 images,
~90 MB, about a minute). After that it just moves a batch.

Defaults are 100 train + 20 val per class, across every class found in the
pool. With the 9-class taxonomy that is **1,080 images per batch**, a long
sitting. Either shrink it:

```bash
python python_code/curation_pull.py --train-count 30 --val-count 10   # 360
```

or take a few classes at a time:

```bash
python python_code/curation_pull.py --classes people,furniture
```

The loop is built to be re-run, so small batches cost nothing.

Classes are discovered from the pool, not hard-coded. Curation is
independent of whatever `--classes` subset you later train on.

### 2. Review

One page for the whole batch, with a folder dropdown:

```bash
python python_code/make_review_gallery.py
```

Open `curation_review/_review.html`. For each folder: click the bad
thumbnails, press **Mark reviewed**, then move on with the dropdown or the
left/right arrow keys. The dropdown shows a tick against reviewed folders
and a running reject count, and the header tracks how many folders you
have finished.

When every folder is marked, press **Export all reviewed** and save the
files to `~/Downloads`.

Two things this does that matter:

- It writes **one file per reviewed folder, including empty ones**. An
  empty `rejected_*.txt` is what tells `curation_resolve.py` "reviewed,
  nothing wrong" as opposed to "not reviewed yet". Creating those by hand
  is otherwise the most tedious part of the loop.
- Marking a folder reviewed is **explicit**. Clicking through the dropdown
  to see what is in a folder does not silently accept it.

The folder list is baked in when the page is generated, because a
`file://` page cannot enumerate directories. **Re-run
`make_review_gallery.py` after every `curation_pull.py`.**

Browsers block multi-file downloads by default; the first "Export all"
raises a permission prompt, which you allow once. If downloads still fail,
use **Export current folder** one at a time, or **Copy all as text** and
save the sections by hand.

**Leave the downloaded filenames alone.** `curation_resolve.py` reads the
split and class out of them (`rejected_curation_review_train_furniture.txt`).

**Export before closing the tab.** Marks are kept in localStorage as a
convenience, but Chrome gives each `file://` page an opaque origin and
often loses them on reload.

`make_gallery.py` still works if you want a single directory on its own —
it writes a `_gallery.html` per folder, and produces identical export
files. Full instructions: `python_code/documents/gallery_instructions.md`.

### 3. Resolve

```bash
python python_code/curation_resolve.py --dry-run   # look first
python python_code/curation_resolve.py
```

Accepted images move to `data_hand_curated/`, rejected ones to
`data_rejected/`, and the rejected *filenames* are written into
`everyday_openimages160x120/meta/{split}_curation_rejects.json`.

A folder with **no reject list is skipped, not accepted** — it is reported
as unreviewed and left exactly as it was. So you can review half the
classes, resolve, and pick up the rest later. If a batch genuinely has
nothing wrong with it, save an empty `rejected_*.txt` in that folder so it
resolves.

Repeat 1–3 until `data_hand_curated/` is big enough.

### 4. Fine-tune

```bash
python python_code/train_rgb_cnn.py \
    --data-dir data_hand_curated \
    --fine-tune-from rgb_cnn \
    --rgb-block-width 5 --rgb-block-height 5 \
    --classes people,computer,doors,fruit,car \
    --epochs 15 \
    --artifacts-name rgb_5x5_finetuned
```

`--fine-tune-from` names an `output/` subdirectory. It loads that run's
`float_model.pt`, checks its manifest against this run's configuration,
and trains at `--fine-tune-lr` (default `1e-4`, roughly 10× below the
from-scratch rate) instead of starting from random weights. QAT and the
int8 conversion then run exactly as normal, so the result is still
deployable.

The log prints the starting validation accuracy before any fine-tuning
steps, so "did this help?" is answerable from one run's output.

**The class list and its order must match the base run.** The check
refuses to proceed otherwise, because the final `Linear`'s rows are
positional — a reordered list loads cleanly, trains happily, and produces
a model whose exported labels do not match its logits.

Since the class list changes from time to time, use `--artifacts-name`
values that encode the configuration (`rgb_5x5_9cls`, `rgb_5x5_5cls`)
rather than the default `rgb_cnn`. Two runs sharing a name silently
overwrite each other's weights.

## The reject list is the part worth protecting

`data_hand_curated/` is a working directory — disposable, rebuildable.
The durable artifact is:

```
everyday_openimages160x120/meta/{train,val}_curation_rejects.json
```

`build_data.py` and `filter_intersections.py` consult it as a third drop
rule, alongside the Open-Images intersection check and the YOLO person
check. Without it, the next `build_data.py` run puts every image you
rejected straight back into `data/`.

It is keyed by **original source class** — the 23 raw ones (`bookshelves`,
`chair`, `cups`), not the merged 9. So it survives changes to
`CLASS_MERGE_MAP` as well as changes to `--classes`. Curate once; it keeps
applying.

It is small and diffable. **Commit it.** Right now your review effort
would exist in exactly one copy on one disk.

To disable it for one run: `build_data.py --no-curation-filter`.

There is also `apply_curation_rejects.py`, which folds gallery exports
into the same file directly — useful if you review `data/` or
`<source>_processed/` outside this loop.

## Why the test set stays untouched

`data/test` is never pulled into curation, and
`train_rgb_cnn.py` falls back to reading it from `data/` whenever
`--data-dir` has no `test/` of its own. Override with `--test-data-dir`.

Comparing a fine-tuned model against its base only means something if both
are scored on the same images. Two consequences:

**The test set stays noisy.** You are cleaning train and val but scoring
against uncurated data, so whatever fraction of `data/test` is mislabelled
sets a ceiling neither model can cross. Fine-tuning may therefore show a
smaller gain than it actually delivers on camera. That is the deliberate
trade: a stable yardstick that under-measures, rather than a moving one
that flatters. To size the real effect later, curate a copy of the test
set separately and report both numbers, clearly labelled.

**Do not re-run `build_data.py` mid-comparison.** Curation rejects are
applied in step 1 of that script, *before* step 2 carves the test split out
of train. A rebuild after curating therefore changes `data/test` twice
over: rejected images vanish from the pool, and the stratified split
reshuffles what remains. Any accuracy recorded against the old test set
stops being comparable.

So either curate → fine-tune → compare → *then* rebuild, or rebuild
whenever you like but re-score the base model before quoting a comparison.
The rejects themselves are never lost either way — only the test set's
identity changes.

## Recovering from mistakes

Nothing is deleted, so everything is reversible by moving files back.

| situation | fix |
|---|---|
| rejected an image by mistake | move it from `data_rejected/<split>/<class>/` back to `data_hand_curated/`, and delete its name from the meta JSON |
| want to redo a whole batch | move images from `data_hand_curated/` and `data_rejected/` back into `data_tmp/<split>/<class>/` |
| want to start curation over | delete `data_tmp/`, `curation_review/`, `data_hand_curated/`, `data_rejected/` and the two `*_curation_rejects.json` files |
| pulled a batch too early | resolve it, or move the images back to `data_tmp/` by hand |

`data/` is never touched by any of this, so a full reset always gets you
back to the untouched dataset.

Resolved reject lists are archived to
`curation_review/_resolved_rejected_lists/`, so a re-run cannot
re-process an already-resolved folder.
