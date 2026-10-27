# Data audit: RecipeNLG

Numbers come from `data/processed/main/dataset_manifest.json`, the notebook `notebooks/01_eda.ipynb`, and a
one-pass scan of the raw CSV with the pandas parser (list columns are parsed with `json.loads`, then
`ast.literal_eval`; nothing is evaluated as code). The licence of the dataset has not been checked or assigned here.

## 1. Raw file (structure)

| Check | Result |
| --- | --- |
| Rows | 2,231,142 (ids 0 to 2,231,141, contiguous) |
| Columns | index, `title`, `ingredients`, `directions`, `link`, `source`, `NER` |
| Null values | 1 in `title`; 0 in every other column |
| Malformed list (non-empty text that is not a list of strings) | 0 in `ingredients`, `directions`, `NER` |
| Empty list | 573 in `NER`; 0 in `ingredients` and `directions` |
| Links without a scheme | all links lack `http(s)://` (the pipeline adds `https://`); 0 unusable |
| Source | `Gathered` 1,643,098 (73.6%), `Recipes1M` 588,044 (26.4%) |
| Exact duplicates on (title, ingredients text) | 1,245 (0.06%) |

The file is sorted by source, so the first rows are all `Gathered`. Taking the head of the file would be biased; the
sample is a seeded random draw over all ids.

## 2. Sample used by the project

50,000 recipes (`main`), of which the first 5,000 form `pilot`. From the 50,360 recipes examined: 180 dropped as exact
duplicates (same title and same normalised ingredient set), 180 dropped because they had fewer than 2 or more than 40
normalised ingredients, 0 dropped for null or malformed fields. The sample keeps the raw source mix
(`Gathered` 73.7%, `Recipes1M` 26.3%). Recipes are grouped when they share an ingredient set or a title: 39,869 groups,
92.7% singletons, largest group 141. Split by group: 35,896 train, 3,970 validation, 10,134 test.

## 3. Manual review of 100 recipes

Method: 100 recipes drawn from the corpus with `random.Random(42).sample` (the notebook reproduces the sample), each
read as raw ingredient lines, then the NER column, then the normalised list. A recipe counts as having an issue when
its normalised ingredient list contains a junk or truncated item, or lacks an ingredient present in the raw lines.

**36 of 100 recipes have at least one issue.** Each recipe is counted once under its main category:

| Category | Recipes | What goes wrong |
| --- | ---: | --- |
| A. Junk word replaces the real ingredient | 9 | NER keeps a modifier and loses the noun |
| B. Name truncated to a fragment or brand | 13 | NER cuts the name short |
| C. Head noun missing | 6 | only the modifier survives |
| D. Reference to another recipe | 2 | "Pie Meringue", "Fresh Pasta Dough" become ingredients |
| E. Introduced by our normaliser | 3 | repeated word, alternatives, odd plural |
| F. Ingredient absent from NER | 3 | NER simply omits it |

### Examples (raw line, then NER, then normalised)

| recipe_id | Category | Raw | NER | Normalised | Problem |
| --- | --- | --- | --- | --- | --- |
| recipenlg:1867895 | A | 1 tablespoon virgin or extra-virgin coconut oil | virgin | virgin | coconut oil lost |
| recipenlg:2165415 | A | 7 ounces skinless, boneless duck breast | skinless | skinless | duck breast lost |
| recipenlg:18013 | A | 5 c. warm, cooked squash, slightly drained | warm | warm | squash lost |
| recipenlg:629471 | A | 6 or 8 center cut pork chops | center | center | pork chops lost |
| recipenlg:365311 | A | 1 box white or yellow cake mix | white | white | cake mix lost |
| recipenlg:843977 | A | 2 Tbsp. white or red wine vinegar | white | white | vinegar lost |
| recipenlg:611275 | A | generous dash of crushed thyme | generous | generous | thyme lost |
| recipenlg:294082 | A | 4 level tsp. baking powder | level tsp | level tsp | baking powder lost |
| recipenlg:801458 | B | Kraft Miracle Whip | Miracle | miracle | brand fragment |
| recipenlg:811009 | B | 1 can Mountain Dew | Dew | dew | brand fragment |
| recipenlg:2048944 | B | 1 tablespoon Shao Hsing rice wine | Hsing rice wine | hsing rice wine | name cut |
| recipenlg:1726553 | B | 1/2 cup carnation evaporated milk | carnation | carnation | evaporated milk lost |
| recipenlg:1723189 | B | 2 Tablespoons Canola Or Veg Oil | Canola Or | canola | oil lost |
| recipenlg:1232662 | B | 1 tablespoon orange flower water | orange flower | orange flower | "water" lost |
| recipenlg:2131777 | B | BAKER'S GERMAN'S Sweet Chocolate | 'S Sweet Chocolate | s sweet chocolate | apostrophe fragment |
| recipenlg:1780454 | B | confectioners' sugar | confectioners | confectioner | alias "confectioners sugar" cannot match |
| recipenlg:78727 | C | 1/4 c. all-purpose, sifted flour | all-purpose | all purpose | flour lost |
| recipenlg:1958965 | C | 3 1/2 to 4 cups all-purpose flour | all-purpose | all purpose | flour lost |
| recipenlg:980306 | C | 1 jalapeno pepper, diced | pepper | pepper | jalapeno lost |
| recipenlg:580820 | C | 1 (16 oz.) cucumber Ranch dressing | cucumber | cucumber | dressing lost |
| recipenlg:1103380 | C | 1 (300 g) package soft silken tofu | silken | silken | tofu lost |
| recipenlg:2152423 | D | Fresh Pasta Dough (page 365) | Fresh Pasta Dough | pasta dough | another recipe |
| recipenlg:809883 | D | Pie Meringue | Pie | pie | another recipe |
| recipenlg:1592921 | E | 4 tablespoons agar agar strands | agar agar | agar | repeated word collapsed |
| recipenlg:1566961 | E | 1 can (10oz) Consomme Soup Or Beef Broth | Consomme Soup Or Beef Broth | consomme soup, beef broth | alternatives become two required ingredients |
| recipenlg:272469 | E | 1 (10 oz.) pkg. frozen broccoli cuts | frozen broccoli cuts | broccoli cut | "cuts" singularised |
| recipenlg:1343748 | F | 1 t salt 1/2 t pepper | (pepper missing) | | pepper omitted |
| recipenlg:233658 | F | 1/4 c. + 2 Tbsp. unsweetened cocoa | + | | cocoa omitted |

What worked: quantities, units and preparation words are removed correctly in nearly every line, plurals are folded
(`eggs` to `egg`), and the aliases fire (`confectioners sugar` to `powdered sugar` in recipenlg:2092737, `scallions`
to `green onion` in recipenlg:269222).

Other observations from the same 100 recipes: sub-recipes inside one recipe repeat ingredient lines (recipenlg:2152423 has
32 raw lines for 17 normalised ingredients), and brand names often stand in for the product (Cool Whip, Carnation).

## 4. Normaliser is not a fixed point

Normalising an already-normalised name changes 198 of the 14,450 distinct names (1.4%), in 387 recipes (0.77%) of the
50,000. Examples:

| recipe_id | Name in corpus | After a second pass |
| --- | --- | --- |
| recipenlg:63234 | five spice powder | spice powder |
| recipenlg:454320 | pot roast | roast |
| recipenlg:119213 | one step angel food cake mix | step angel food cake mix |
| recipenlg:200581 | chunk style peanut butter | style peanut butter |
| recipenlg:316633 | ready to spread | ready |
| recipenlg:326167 | section | (empty) |

Retrieval stays consistent: every model's pipeline applies `normalize_all` to documents and to queries, so both receive
the same number of passes (`src/pipeline.py`, covered by the "a train recipe finds itself" test). The cost is that a few
names are mangled twice.

## 5. Other measurements

- NER has fewer entries than the raw ingredient lines in 10,095 recipes (20.2%); the NER column is the primary source
  because raw lines are noisier, but it silently drops ingredients.
- 133 distinct normalised names have 3 characters or fewer; most are real (`egg`, `oil`, `yam`), a few are fragments.
- 79 names appear in more than 2% of the recipes; salt appears in 43.5%.
- 8,704 of the 14,450 names (60%) occur in one recipe only; many are the fragments listed above.

## 6. Recommendations (not applied in this change)

Changing the normaliser or the NER handling alters the processed data, so it needs the shared-file review and a rebuilt,
re-shared `processed-*.zip`.

1. Drop single-word NER entries made of modifiers or units (virgin, skinless, warm, center, white, level, generous, ...)
   and recover the ingredient from the raw line.
2. When NER has fewer entries than raw lines, parse the missing lines with the raw-line parser instead of omitting them.
3. Treat `all-purpose` as `all-purpose flour`; keep a repeated word (`agar agar`); keep only the first alternative of
   "X or Y"; handle the apostrophe in `confectioners' sugar`.
4. Make normalisation idempotent (apply until stable), so a second pass is a no-op.
5. Ignore lines that name another recipe (a parenthesised page reference, "recipe" in the line).
