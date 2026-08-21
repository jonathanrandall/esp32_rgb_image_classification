# Dataset curation workflow

Manual visual curation pass over `openimages_160x120/`, in small batches
per class, using the existing gallery tool (`python_code/make_gallery.py`)
to mark bad images and two scripts here to move batches around. Nothing
is ever deleted -- everything is moved between folders, so any step is
reversible by hand if needed.

## Layout

```
data_curation/
  open_images_original/{train,val}/<class>/   the full original pull, untouched -- backup, don't edit
  open_images_tmp/{train,val}/<class>/        working copy select_batch.py draws batches from
  select_images/
    select_batch.py                            step 1: pull a batch
    resolve_selection.py                        step 3: resolve a reviewed batch
    <split>/<class>/                            a batch currently under review (created as needed)

../openimages_160x120/{train,val}/<class>/     accepted images land back here (starts empty)
../openimages_160x120_rejected/{train,val}/<class>/   rejected images land here (created as needed)
```

`openimages_160x120/` is the dataset the training scripts actually read
(`--dataset-source openimages_160x120`, the default). It starts out
empty here -- it only fills back up as batches get reviewed and resolved.
`open_images_original/` is never touched by any script; if something
goes wrong, everything can be re-copied from there.

## Step 1 -- pull a batch to review

```bash
cd data_curation/select_images
python select_batch.py                          # all classes, both splits, 100 images each
python select_batch.py --classes birds,car       # just these classes
python select_batch.py --count 50 --splits val   # smaller batch, val only
python select_batch.py --seed 42                 # reproducible sample
```

Moves up to `--count` (default 100) random images per class/split from
`open_images_tmp/` into `select_images/<split>/<class>/`. Safe to re-run:
a class/split that already has a full batch sitting in `select_images/`
is skipped, not topped up -- resolve it first (step 3) before pulling
another one. If a class runs low in `open_images_tmp/`, it moves whatever
remains and prints a warning rather than failing.

## Step 2 -- review the batch

Use the existing gallery tool, pointed at whichever `select_images/<split>/<class>/`
folders you just pulled:

```bash
cd ../..    # project root
python python_code/make_gallery.py data_curation/select_images/train/birds data_curation/select_images/val/birds
```

Open the `_gallery.html` it writes into each folder, click thumbnails to
mark rejects, then **"Export rejected list"**. That downloads a
`rejected_<key>.txt` file (usually to `~/Downloads/`) -- full details in
`python_code/documents/gallery_instructions.md`. You don't need to move the
downloaded file anywhere: `resolve_selection.py` (step 3) looks in
`~/Downloads/` automatically, matched by the exact folder it came from.
If you'd rather keep it explicit, you can also drop/rename the file
directly into the same `select_images/<split>/<class>/` folder it's for.

A class/split with no exported reject list yet is left alone by step 3 --
it's fine to review classes in any order, or leave some for later.

## Step 3 -- resolve a reviewed batch

```bash
cd data_curation/select_images
python resolve_selection.py
```

For every `select_images/<split>/<class>/` folder that has a reject list
(from step 2, found either locally or in `~/Downloads/`): images **not**
on the list move to `../../openimages_160x120/<split>/<class>/`
(accepted), images on the list move to
`../../openimages_160x120_rejected/<split>/<class>/` (rejected). The
reject list itself gets archived into
`select_images/_resolved_rejected_lists/` so a re-run doesn't reprocess
an already-resolved folder, and the stale `_gallery.html` is deleted from
the now-emptied folder.

**This also deletes the project's `../../data/` directory** (built from
whatever was there before this curation pass) the first time anything
gets resolved -- it's stale the moment `openimages_160x120/` changes, and
`data/` doesn't auto-rebuild on content changes alone (only on capture
size/chroma/class-set changes). After resolving, rebuild it by running
any training script with the flags you want, e.g. the "Current best
config" command in `python_code/README.md`.

## Repeat

Loop steps 1-3 per class (or all classes at once) until you're happy with
the coverage. `openimages_160x120/` accumulates the accepted images as
you go, so training can start on whatever's been resolved so far -- it
doesn't have to wait for every class to be fully reviewed, though results
won't be comparable to a fully-curated run until it is.

## If something goes wrong

Nothing here deletes source images -- worst case, re-copy from
`open_images_original/` into `open_images_tmp/` (or straight into
`openimages_160x120/`) and start over. `openimages_160x120_rejected/`
keeps every rejected image too, so a rejection can always be undone by
moving a file back.
