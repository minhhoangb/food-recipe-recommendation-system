# Hưng's annotation log — dev judging pool

Annotator: **Hung** | File: `pool_dev_hung.csv` | Rows labelled: **161** (of the 161 assigned)
Scale: the one in `annotation_guide.md` (0 / 1 / 2), unchanged.

## Exposure disclosure (read this before using these labels)

**On 2026-10-30, before labelling, I opened `reports/ai_annotations/dev/ai_annotation_report.md`.**

That file contains an AI annotation pass over the same dev pool, including a per-query
label-count table and four worked examples with their AI labels and reasons. I did **not**
open `pool_dev_key.csv`, `pool_dev_systems.jsonl`, or `reports/ai_annotations/dev/pool_dev_ai.csv`,
so no per-system key and no per-item AI label was seen — but the per-query count table and
the four examples were.

Consequences, stated plainly:

- These labels **must not be reported as blind independent human labels.**
- They must not be used as one of the two independent labels of a double-labelled item
  without this disclosure attached.
- They may be used as a single human judgement with the exposure recorded, and as one
  side of an adjudication, provided the exposure travels with them.
- The AI pass had already been made before my labelling, so the exposure cannot be undone.
  Relocating the file changes nothing.

I judged every row myself from the query text and the recipe (title, ingredient lines,
directions) against the guide's rules, in one pass over the assignment file. The AI
counts were already in my context when I started, so I cannot rule out that they anchored
my sense of the distribution for a few queries. Where I was unsure I recorded `unsure` in
the `notes` column rather than silently guessing.

**Recommendation for the report:** a second annotator (Hoang or Hieu) re-labels a random
subset of my 161 rows *without* reading my labels, and the agreement between that pass and
mine is reported with this disclosure attached. The double-labelled items already planned
in `pool_dev_metadata.json` should keep both original labels, as the guide requires.

## Label distribution

| Label | Items |
| --- | ---: |
| 0 not relevant | 41 |
| 1 partly relevant | 90 |
| 2 relevant | 30 |

Two rows carry `unsure` in `notes`: `m17-07` and `m17-12`. Both are one-line recipe
fragments with no usable cooking method (a prepare-a-chutta-stick instruction and a
"put the karaage into cups" serving note). The guide's rule 0 for an unusable recipe
applies; I kept the lower label and flagged it.

## Per-query counts

| Query | Items | 0 | 1 | 2 |
| --- | ---: | ---: | ---: | ---: |
| m01 | 5 | 1 | 3 | 1 |
| m02 | 16 | 0 | 11 | 5 |
| m04 | 4 | 0 | 3 | 1 |
| m05 | 16 | 1 | 13 | 2 |
| m07 | 6 | 0 | 6 | 0 |
| m08 | 15 | 2 | 12 | 1 |
| m10 | 4 | 2 | 2 | 0 |
| m11 | 21 | 8 | 13 | 0 |
| m13 | 4 | 0 | 2 | 2 |
| m14 | 20 | 8 | 6 | 6 |
| m16 | 5 | 5 | 0 | 0 |
| m17 | 20 | 3 | 5 | 12 |
| m19 | 4 | 1 | 3 | 0 |
| m20 | 21 | 10 | 11 | 0 |

## Notes on the borderline groups

- **m16 (gochujang, pork belly, kimchi)** — all five pooled recipes use pork belly but none
  use gochujang or kimchi. Kimchi is the hard part: pickled mustard greens and Chinese
  preserved vegetables are not kimchi, so those rows are 0 rather than 1.
- **m20 (saffron, arborio rice, white wine, parmesan)** — many risotto recipes match three of
  the four keys but omit saffron. Long-grain rice is not arborio rice, so the two plain
  rice dishes are 0. The white-wine-only rows are drinks.
- **m17 ("I only have chicken")** — a single generic ingredient, so most dishes that use
  chicken qualify. Rows with three or more extra non-staples are 1.
- **m08 (vegetarian constraint)** — one pooled recipe contains chicken and is 0 because the
  guide makes an explicit constraint absolute. Red kidney beans are not black beans.

## How these labels were produced

`scripts/apply_labels.py` holds the label table (item id -> relevance and the reason) and
fills the `relevance` and `notes` columns of the assignment file. The other columns and the
row order are untouched, so the annotated file stays comparable with the pool it came from.
Re-running it reproduces the same file:

```bash
python scripts/apply_labels.py reports/manual_eval/pool_dev_hung.csv
```
