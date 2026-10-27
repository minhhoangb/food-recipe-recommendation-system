# Annotation guide: manual relevance labels (0 / 1 / 2)

Written on 27/10/2026, **before any system output for these queries was looked at**. Do not edit the
scale or the rules after pooling starts; if a rule turns out unclear, add a dated note at the end
and apply it to every label, already given or not.

## Why manual labels

The synthetic qrels only know one relevant recipe per query (the recipe the query was cut from), so
Recall/MRR on them are an origin-recipe proxy. Manual labels judge every pooled recipe, so they
can tell whether a "wrong" result was actually a good recipe. They are used for pooled nDCG@10.

## Files

| File | Content |
| --- | --- |
| `queries.csv` | 40 natural English queries: 20 `dev` (labelled in khung 3), 20 `final` (blind, labelled once after the configuration is frozen in khung 4) |
| `labels_template.csv` | Columns of the pooled file each annotator fills in |

Query columns: `query_id`, `split`, `difficulty` (easy / medium / hard), `text` (what the user
says), `ingredients` (the list sent to every system, `;`-separated), `probes` (what the query tests).

## What you see

For each query, a pool of recipes: the top 5 of every frozen system, merged, duplicates removed,
**shuffled, with no system name**. Each row shows the title, the ingredient lines and the
directions of one recipe. Judge the recipe against the query `text` and its `ingredients`.

## Key ingredients and staples

- **Staples**: salt, black pepper, water, cooking oil (vegetable, canola, olive), granulated sugar,
  all-purpose flour. A staple is assumed to be in the kitchen.
- **Key ingredients**: the query ingredients that are not staples. If a query lists only staples
  (or staples plus butter, eggs, milk), every listed ingredient is key.
- An ingredient counts as *used* when it is a real component of the dish, not a garnish or an
  optional serving suggestion. Same food under another name counts (scallion = green onion,
  chickpeas = garbanzo beans); a different food does not (chicken stock is not chicken breast,
  baking soda is not baking powder).

## The scale

| Label | Meaning | Rule of thumb |
| --- | --- | --- |
| **2** relevant | The user can cook this mainly from what they listed | uses **every** key ingredient (or all but one when the query has 5 or more) **and** needs at most 2 extra non-staple ingredients **and** fits any intent in the text (meal, dish type, "no oven", "vegetarian") |
| **1** partly relevant | A reasonable suggestion that needs more shopping or bends the request | uses **at least half** of the key ingredients, **or** uses them all but needs 3 or more extra non-staples, **or** fits the ingredients but not the stated intent |
| **0** not relevant | Not a useful answer to this query | uses fewer than half of the key ingredients, **or** breaks an explicit constraint (meat for a vegetarian query, nuts for "no nuts"), **or** the recipe is unusable (no real directions, a sub-recipe fragment, a beverage when a meal was asked) |

Tie-breakers, in this order:

1. An explicit constraint in the text wins over ingredient overlap (breaking it is always 0).
2. Count extra ingredients from the recipe's ingredient lines, ignoring staples and anything
   marked optional / to taste / for serving.
3. When torn between two labels, take the lower one and write `unsure` in `notes`.
4. Two recipes with the same text get the same label, whatever their ids.

## Procedure

1. Write your initials in `annotator` on every row you label; one person per row.
2. Read the query text first, then the recipe. About one minute per recipe; do not open the link
   unless the directions are cut off.
3. Fill `relevance` with 0, 1 or 2 only. Use `notes` for `unsure`, missing directions, or the
   rule you used for a hard case.
4. Do not look up which system produced a recipe, and do not change a label after anyone has
   seen per-system scores. Labels suggested by an AI tool are not human labels and must not be
   entered as such.
5. Double labelling: at least 20% of each pool is labelled independently by a second person (a
   copy of the row with the second annotator's initials). Disagreements are discussed after both
   finished; the agreed label goes to `relevance`, the two original labels stay in the log, and
   agreement is reported with `sklearn.metrics.cohen_kappa_score`.
6. Unjudged recipes (outside the pool) are not assumed irrelevant in the report: pooled nDCG@10
   is reported together with the pool coverage of each system's top 10.

## Example (illustration only, not a real pool row)

Query "I have chicken thighs, rice and soy sauce, what can I make for dinner?"
(key: chicken thigh, rice, soy sauce):

- Teriyaki chicken thighs over rice, extra: ginger, honey -> **2**
- Chicken fried rice made with chicken breast, soy sauce, peas, carrot, egg -> **1** (one key
  ingredient replaced, 3 extra non-staples)
- Soy-glazed salmon -> **0** (only one key ingredient)

## Change log

(empty)
