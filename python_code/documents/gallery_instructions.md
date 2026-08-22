# Dataset gallery review

A local, no-server-needed HTML contact-sheet for eyeballing a directory of
dataset images and marking bad ones -- built because reviewing thousands of
small crops one at a time in a standard image viewer doesn't scale, and
Ubuntu's stock file managers don't do fast thumbnail-grid browsing nearly as
well as Windows Explorer does.

## Generating a gallery

```bash
cd python_code
python make_gallery.py <image_dir> [<image_dir> ...]
```

Each `<image_dir>` should be a folder of images (e.g. one class of one
split, like `openimages_160x120/train/car`). This writes an
`_gallery.html` file directly into that folder, referencing the images
already there by filename -- nothing is copied or resized.

Galleries already generated for the current Open Images pull:

```
openimages_160x120/train/{appliances,birds,car,furniture,people}/_gallery.html
openimages_160x120/val/{appliances,birds,car,furniture,people}/_gallery.html
```

`val/` is much smaller (500 or fewer per class) than `train/` (up to 5,000
per class) -- a good place to start if you want a quick read on the
mislabel/quality rate before deciding whether `train/` needs the same pass.

## Using a gallery

Open the `_gallery.html` file directly -- double-click it in your file
manager, or:

```bash
xdg-open openimages_160x120/val/car/_gallery.html
```

- **Click a thumbnail** to mark it rejected (red border, dimmed). Click
  again to unmark it.
- The header shows a running count of total images and how many are
  currently marked rejected.
- **"Export rejected list"** downloads a `rejected_<key>.txt` file --
  one filename per line, sorted. This is the file to hand back so the
  listed images can actually be pulled out of the dataset.
- **"Clear marks"** unmarks everything on that page (asks for
  confirmation first).

### A note on persistence

Marks are also saved to the browser's `localStorage` as a convenience, so
reloading the page shouldn't lose your progress. This is best-effort only,
though -- some browsers (Chrome in particular) give each local `file://`
page its own opaque origin, which can break `localStorage` persistence
across reloads. Firefox handles this more reliably. Either way: **export
before closing the tab** if the marks in a session matter, rather than
relying on reload-and-resume.

## After exporting a rejected list

Hand the `.txt` file back (or just say where it landed). Rejected files
get **moved, not deleted** -- each one is relocated from its class
directory into a `_rejected/` subfolder alongside it, e.g.:

```
openimages_160x120/train/car/car_001234.jpg
  -> openimages_160x120/train/car/_rejected/car_001234.jpg
```

The dataset builder (`dct_common/dataset.py`) only globs files directly
inside each class directory (`*.jpg`, non-recursive), so anything sitting in
`_rejected/` is automatically excluded from `data/` without needing to be
re-downloaded later if you change your mind about a particular image --
just move it back out.
