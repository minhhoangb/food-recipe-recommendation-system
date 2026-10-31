"""Write the human relevance labels for one assignment file of the dev judging pool.

    python scripts/apply_labels.py reports/manual_eval/pool_dev_hung.csv

The labels live in LABELS below (item_id -> (relevance, notes)). The script only fills the
`relevance` and `notes` columns of the given file; the other columns and the file order are
left untouched, so the annotated file stays byte-comparable with the pool it came from.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

LABELS: dict[str, tuple[int, str]] = {
    # --- m01: chicken breasts, broccoli, garlic ---
    "m01-01": (0, "broccoli only; chicken comes from a soup mix, not chicken breast"),
    "m01-04": (2, "chicken breasts, broccoli and garlic all used; 2 extras"),
    "m01-08": (1, "chicken and broccoli; no garlic; stuffing mix and soup extra"),
    "m01-09": (1, "broccoli and garlic; chicken stock is not chicken breast"),
    "m01-21": (1, "chicken breasts and garlic; potatoes, green beans and sauce starter extra"),
    # --- m02: ground beef, onion, tomato sauce ---
    "m02-01": (1, "all three used but pinto beans, corn bread mix and cheese are 3 extras"),
    "m02-02": (1, "all three used but shells, garlic, cheese and marinara are 4 extras"),
    "m02-03": (2, "all three used; only barbecue sauce extra"),
    "m02-04": (1, "all three used but sausage, tomatoes and sauce mix are 3 extras"),
    "m02-05": (1, "all three used but green pepper, cheese and macaroni are 3 extras"),
    "m02-06": (2, "all three used; rigatoni and parsley are 2 extras"),
    "m02-07": (2, "all three used; potatoes and carrots are 2 extras"),
    "m02-08": (1, "beef and onion; fresh tomatoes are not tomato sauce"),
    "m02-09": (1, "beef and onion; canned tomatoes are not tomato sauce"),
    "m02-10": (1, "all three used but garlic, mushroom soup, sauce mix and parmesan are 4 extras"),
    "m02-11": (1, "beef and onion; canned tomatoes are not tomato sauce"),
    "m02-12": (1, "beef, onion and tomatoes; rice and potatoes extra, not tomato sauce"),
    "m02-13": (2, "all three used; only cabbage extra"),
    "m02-14": (2, "all three used; bread crumbs and egg are 2 extras"),
    "m02-15": (1, "all three used but mustard, chili powder and Worcestershire are 3 extras"),
    "m02-16": (1, "beef and tomato sauce; no onion"),
    # --- m04: eggs, spinach, cheese for breakfast ---
    "m04-01": (1, "eggs, spinach and cheese used; cod is extra and it is not a breakfast dish"),
    "m04-03": (1, "eggs, spinach and cheese used; ham and pie crust extra"),
    "m04-07": (1, "spinach and eggs; no cheese"),
    "m04-08": (2, "eggs, spinach and cheese; sour cream and onion soup mix are 2 extras"),
    # --- m05: pasta with shrimp, garlic, butter ---
    "m05-01": (1, "pasta, garlic and butter; imitation crab is not shrimp; 4 extras"),
    "m05-02": (1, "shrimp, garlic and pasta; butter only a substitute; 4 extras"),
    "m05-03": (2, "shrimp, garlic, butter and pasta all used; only parmesan extra"),
    "m05-04": (0, "pasta only; no shrimp, garlic or butter"),
    "m05-05": (1, "shrimp, garlic and pasta; no butter; 5 extras"),
    "m05-06": (1, "shrimp, garlic and pasta; no butter; 5 extras"),
    "m05-07": (1, "all four used but soy sauce, snow peas and carrot are 3 extras"),
    "m05-08": (1, "all four used but wine, chicken base, starch, parsley and cheese are 5 extras"),
    "m05-09": (1, "pasta and butter; no shrimp or garlic"),
    "m05-10": (1, "all four used but onion, celery, tomatoes, soup, mushrooms and cayenne are 6 extras"),
    "m05-11": (1, "pasta and butter; no shrimp or garlic"),
    "m05-12": (2, "shrimp, garlic, butter and pasta all used; basil and mushrooms are 2 extras"),
    "m05-13": (1, "shrimp and pasta; no garlic or butter"),
    "m05-14": (1, "shrimp and butter; grits are not pasta, no garlic"),
    "m05-15": (1, "shrimp and pasta; no garlic or butter"),
    "m05-16": (1, "pasta, garlic and butter; no shrimp"),
    # --- m07: leftover turkey, celery, mayonnaise, dried cranberries ---
    "m07-01": (1, "turkey, celery and mayonnaise; no dried cranberries"),
    "m07-08": (1, "chicken is not turkey; celery and mayonnaise only"),
    "m07-09": (1, "turkey, celery and mayonnaise; no dried cranberries"),
    "m07-14": (1, "turkey and mayonnaise; celery only as soup, no cranberries"),
    "m07-15": (1, "turkey and celery; no mayonnaise, no cranberries"),
    "m07-19": (1, "turkey, celery and mayonnaise; no dried cranberries"),
    # --- m08: black beans, corn, red onion, cilantro (vegetarian) ---
    "m08-01": (1, "black beans and corn; no red onion, no cilantro"),
    "m08-02": (1, "black beans, red onion and cilantro; no corn"),
    "m08-03": (1, "all four used but red pepper, cumin, tabasco, lime and vinegar are 5 extras"),
    "m08-04": (1, "all four used but tomatoes, vinegar and avocado are 3 extras"),
    "m08-05": (0, "red kidney beans are not black beans; scallions are not red onion"),
    "m08-06": (1, "black beans, corn and red onion; no cilantro"),
    "m08-07": (1, "all four used but sugar, lime, jalapeno and garlic are 4 extras"),
    "m08-08": (1, "black beans and corn; no red onion, no cilantro"),
    "m08-09": (1, "black beans, red onion and cilantro; no corn"),
    "m08-10": (1, "all four used but tomatoes, picante sauce, lime and cumin are 4 extras"),
    "m08-11": (1, "black beans, corn and red onion; no cilantro"),
    "m08-12": (0, "contains chicken; breaks the explicit vegetarian constraint"),
    "m08-13": (1, "black beans and onion; no corn, no cilantro"),
    "m08-14": (2, "all four used; only lemon juice extra"),
    "m08-15": (1, "black beans, corn and red onion; no cilantro"),
    # --- m10: ground turkey, zucchini, bell peppers, taco seasoning ---
    "m10-03": (0, "ground beef is not ground turkey; only taco seasoning of the four"),
    "m10-04": (1, "taco seasoning and bell pepper; beef is not turkey, no zucchini"),
    "m10-11": (1, "zucchini and bell pepper; no ground turkey, no taco seasoning"),
    "m10-13": (0, "crawfish is not ground turkey; only bell pepper of the four"),
    # --- m11: chickpeas, coconut milk, curry powder, spinach ---
    "m11-01": (0, "curry powder only; a dressing"),
    "m11-02": (0, "curry powder only; lamb and kohlrabi, no chickpeas or spinach"),
    "m11-03": (1, "coconut milk and curry powder; no chickpeas or spinach"),
    "m11-04": (0, "curry powder only; popcorn"),
    "m11-05": (1, "coconut milk and spinach; no chickpeas"),
    "m11-06": (1, "chickpeas and curry powder; no coconut milk, no spinach"),
    "m11-07": (1, "chickpeas, curry powder and spinach; no coconut milk"),
    "m11-08": (1, "spinach and curry powder; black-eyed peas are not chickpeas"),
    "m11-09": (1, "coconut milk, curry powder and spinach; no chickpeas"),
    "m11-10": (1, "coconut milk and curry powder; no chickpeas or spinach"),
    "m11-11": (1, "chickpeas, coconut milk and curry powder; no spinach"),
    "m11-12": (1, "coconut milk and curry powder; no chickpeas or spinach"),
    "m11-13": (1, "chickpeas and spinach; no coconut milk or curry powder"),
    "m11-14": (0, "coconut milk only; no chickpeas, curry powder or spinach"),
    "m11-15": (0, "curry powder only; canned soups"),
    "m11-16": (0, "curry powder only; popcorn seasoning"),
    "m11-17": (1, "chickpeas and spinach; no coconut milk or curry powder"),
    "m11-18": (0, "spinach and bacon only; no chickpeas, coconut milk or curry"),
    "m11-19": (1, "chickpeas and spinach; no coconut milk or curry powder"),
    "m11-20": (0, "spinach only; a dip"),
    "m11-21": (1, "chickpeas and spinach; no coconut milk or curry powder"),
    # --- m13: tomatoes, basil, mozzarella, balsamic vinegar, olive oil ---
    "m13-03": (1, "tomatoes, basil and olive oil; no mozzarella, no balsamic"),
    "m13-10": (1, "tomatoes, basil, balsamic and olive oil; no mozzarella; pork chops extra"),
    "m13-12": (2, "all but olive oil used; bread and pepper flakes are 2 extras"),
    "m13-18": (2, "all five used; only bread extra"),
    # --- m14: tahini, lemon, garlic ---
    "m14-01": (1, "tahini, lemon and garlic used but carrots, harissa, cumin and za'atar are 4 extras"),
    "m14-02": (2, "tahini, lemon and garlic all used; eggplant and spice are 2 extras"),
    "m14-03": (0, "tahini only; no lemon or garlic"),
    "m14-04": (0, "tahini only; no lemon or garlic"),
    "m14-05": (0, "a cocktail; lemon only, no tahini or garlic"),
    "m14-06": (2, "tahini, lemon and garlic all used; chickpeas and red pepper are 2 extras"),
    "m14-07": (0, "tahini only; no lemon or garlic"),
    "m14-08": (1, "tahini, lemon and garlic used but cauliflower, cumin, cayenne and pine nuts are 4 extras"),
    "m14-09": (1, "tahini, lemon and garlic used but chickpeas, peppers and artichokes are 4 extras"),
    "m14-10": (1, "tahini, lemon and garlic used but watercress, mint and parsley are 3 extras"),
    "m14-11": (1, "tahini, lemon and garlic used but chickpeas, cumin and parsley are 3 extras"),
    "m14-12": (2, "tahini, lemon and garlic all used; only garbanzo beans extra"),
    "m14-13": (0, "a drink; lemon only"),
    "m14-14": (2, "tahini, lemon and garlic all used; only chickpeas extra"),
    "m14-15": (0, "a cheese plate; no tahini or garlic"),
    "m14-16": (0, "a drink; lemon only"),
    "m14-17": (0, "a cocktail; lemon only"),
    "m14-18": (1, "tahini, lemon and garlic used but chickpeas, cumin and onion powder are 3 extras"),
    "m14-19": (2, "tahini, lemon and garlic all used; split peas and mint are 2 extras"),
    "m14-20": (2, "tahini, lemon and garlic all used; spinach and sesame seeds are 2 extras"),
    # --- m16: gochujang, pork belly, kimchi ---
    "m16-04": (0, "pork belly only; no gochujang, no kimchi"),
    "m16-08": (0, "pork belly only; no gochujang, no kimchi"),
    "m16-09": (0, "pork belly only; no gochujang, no kimchi"),
    "m16-13": (0, "pork belly only; pickled greens are not kimchi, no gochujang"),
    "m16-15": (0, "pork belly only; no gochujang, no kimchi"),
    # --- m17: I only have chicken ---
    "m17-01": (2, "chicken only; flour and oil are staples"),
    "m17-02": (2, "chicken only; salt and water are staples"),
    "m17-03": (1, "chicken used; biscuit dough, egg, milk and soup extra"),
    "m17-04": (2, "chicken only; liquid smoke is one extra"),
    "m17-05": (2, "chicken used; maple syrup and yogurt are 2 extras"),
    "m17-06": (2, "chicken used; taco seasoning and corn chips are 2 extras"),
    "m17-07": (0, "one-line fragment with no usable cooking method; unsure"),
    "m17-08": (2, "chicken only; beer and garlic powder are 2 extras"),
    "m17-09": (2, "chicken only; butter, salt and pepper are staples"),
    "m17-10": (2, "chicken only; hoisin sauce is one extra"),
    "m17-11": (2, "chicken only; salt and pepper are staples"),
    "m17-12": (0, "a serving suggestion with no cooking method; unsure"),
    "m17-13": (2, "chicken used; biscuits are one extra"),
    "m17-14": (2, "chicken used; barbecue sauce is one extra"),
    "m17-15": (1, "chicken used; dried beef, bacon, soup and sour cream are 4 extras"),
    "m17-16": (1, "chicken used; two seasonings and barbecue sauce are 3 extras"),
    "m17-17": (2, "chicken used; rub and butter are 2 extras"),
    "m17-18": (0, "bacon with cracker crumbs; no chicken"),
    "m17-19": (1, "chicken used but the directions are very thin; crumbs and sauce extra"),
    "m17-20": (1, "chicken used; gravy, vegetables and stuffing are 3 extras"),
    # --- m19: kale, quinoa, feta, cucumber, red onion, lemon, olive oil, mint ---
    "m19-06": (1, "quinoa, cucumber, lemon, olive oil and mint; no kale, no feta"),
    "m19-08": (1, "cucumber, mint, lemon and olive oil; no kale, quinoa or feta"),
    "m19-18": (0, "kale, quinoa and red onion only; fewer than half of the 8 ingredients"),
    "m19-20": (1, "cucumber, red onion, lemon, olive oil, mint and feta; no kale, no quinoa"),
    # --- m20: saffron, arborio rice, white wine, parmesan ---
    "m20-01": (0, "a drink; white wine only"),
    "m20-02": (0, "a cocktail; white wine only"),
    "m20-03": (0, "a drink; white wine only"),
    "m20-04": (1, "arborio rice, white wine and parmesan; no saffron"),
    "m20-05": (0, "saffron rice is served alongside, not cooked; no arborio, wine or parmesan"),
    "m20-06": (1, "arborio rice, white wine and parmesan; no saffron"),
    "m20-07": (1, "arborio rice and white wine; parmesan is only to serve, no saffron"),
    "m20-08": (1, "arborio rice and white wine; no parmesan, no saffron"),
    "m20-09": (0, "red wine and plain rice; no arborio, saffron or parmesan"),
    "m20-10": (0, "a punch; white wine only"),
    "m20-11": (0, "brown and white rice, not arborio; no saffron, wine or parmesan"),
    "m20-12": (1, "arborio rice, white wine and parmesan; no saffron"),
    "m20-13": (1, "all four used but onion, pepper, asparagus, eggs and stock are 5 extras"),
    "m20-14": (0, "a dessert; white wine only"),
    "m20-15": (1, "arborio rice, white wine and parmesan; no saffron"),
    "m20-16": (0, "saffron and long grain rice, not arborio; no wine or parmesan"),
    "m20-17": (1, "arborio rice, white wine and parmesan; no saffron"),
    "m20-18": (1, "saffron, arborio rice and white wine; no parmesan"),
    "m20-19": (1, "all four used but peas, fennel, onion, broth and parsley are 5 extras"),
    "m20-20": (1, "arborio rice, white wine and parmesan; no saffron"),
    "m20-21": (0, "saffron and long grain rice, not arborio; no wine or parmesan"),
}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 1
    path = Path(argv[1])
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())

    missing = [row["item_id"] for row in rows if row["item_id"] not in LABELS]
    extra = sorted(set(LABELS) - {row["item_id"] for row in rows})
    if missing or extra:
        print(
            f"labels do not match the file: {len(missing)} missing {missing[:5]}, "
            f"{len(extra)} extra {extra[:5]}"
        )
        return 1

    for row in rows:
        relevance, notes = LABELS[row["item_id"]]
        row["relevance"] = str(relevance)
        row["notes"] = notes

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} labels to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
