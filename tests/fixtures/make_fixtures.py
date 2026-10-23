"""Regenerate tests/fixtures/raw_sample.csv (hand-written recipes, NOT RecipeNLG data).

Run:  python tests/fixtures/make_fixtures.py
Contains 28 valid recipes plus deliberately broken rows (malformed lists, NaN, duplicates,
too few ingredients, missing link) so the parser and data_collection can be tested offline.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path(__file__).with_name("raw_sample.csv")

RECIPES = [
    ("Garlic Chicken and Rice", ["chicken breasts", "rice", "garlic", "soy sauce", "green onions"]),
    ("Lemon Chicken", ["chicken breasts", "lemon juice", "garlic", "olive oil", "thyme"]),
    ("Egg Fried Rice", ["rice", "eggs", "green onions", "soy sauce", "peas"]),
    ("Tomato Basil Pasta", ["spaghetti", "tomatoes", "basil", "garlic", "olive oil"]),
    ("Creamy Mushroom Pasta", ["penne", "mushrooms", "heavy cream", "garlic", "parmesan cheese"]),
    ("Beef Tacos", ["ground beef", "taco shells", "cheddar cheese", "lettuce", "tomatoes"]),
    ("Black Bean Burrito", ["black beans", "tortillas", "rice", "salsa", "cheddar cheese"]),
    ("Pancakes", ["flour", "milk", "eggs", "baking powder", "butter", "sugar"]),
    ("Banana Bread", ["bananas", "flour", "sugar", "eggs", "baking soda", "butter"]),
    ("Chocolate Chip Cookies", ["flour", "butter", "brown sugar", "eggs", "chocolate chips", "vanilla"]),
    ("Oatmeal Cookies", ["oats", "flour", "butter", "brown sugar", "eggs", "cinnamon"]),
    ("Apple Pie", ["apples", "flour", "butter", "sugar", "cinnamon", "lemon juice"]),
    ("Greek Salad", ["cucumber", "tomatoes", "feta cheese", "olives", "red onion", "olive oil"]),
    ("Caesar Salad", ["romaine lettuce", "parmesan cheese", "croutons", "garlic", "lemon juice"]),
    ("Tomato Soup", ["tomatoes", "onion", "garlic", "vegetable stock", "cream"]),
    ("Chicken Noodle Soup", ["chicken stock", "egg noodles", "carrots", "celery", "chicken breasts"]),
    ("Vegetable Stir Fry", ["broccoli", "carrots", "soy sauce", "garlic", "ginger", "sesame oil"]),
    ("Salmon with Dill", ["salmon fillets", "dill", "lemon juice", "butter", "garlic"]),
    ("Shrimp Scampi", ["shrimp", "linguine", "garlic", "butter", "white wine", "parsley"]),
    ("Guacamole", ["avocados", "lime juice", "red onion", "cilantro", "tomatoes"]),
    ("Hummus", ["chickpeas", "tahini", "lemon juice", "garlic", "olive oil"]),
    ("French Toast", ["bread", "eggs", "milk", "cinnamon", "butter", "vanilla"]),
    ("Scrambled Eggs", ["eggs", "butter", "milk", "chives"]),
    ("Potato Soup", ["potatoes", "onion", "chicken stock", "cream", "bacon"]),
    ("Mashed Potatoes", ["potatoes", "butter", "milk", "garlic", "chives"]),
    ("Pesto Pasta", ["spaghetti", "basil", "pine nuts", "parmesan cheese", "garlic", "olive oil"]),
    ("Fruit Smoothie", ["bananas", "strawberries", "yogurt", "honey", "milk"]),
    ("Lentil Curry", ["lentils", "onion", "garlic", "ginger", "curry powder", "coconut milk"]),
]
UNITS = ["1 cup", "2 tablespoons", "1/2 teaspoon", "3", "1 (8 oz.) package", "¼ cup", "2 lbs."]


def raw_lines(names: list[str]) -> list[str]:
    return [f"{UNITS[i % len(UNITS)]} {name}" for i, name in enumerate(names)]


def row(index: int, title, ingredients, directions, link, source, ner) -> list:
    return [index, title, ingredients, directions, link, source, ner]


def main() -> None:
    rows: list[list] = []
    for i, (title, names) in enumerate(RECIPES):
        ner = json.dumps([] if i == 5 else names)  # row 5: empty NER -> raw-line fallback
        rows.append(
            row(
                i,
                title,
                json.dumps(raw_lines(names)),
                json.dumps([f"Step 1 of {title}.", "Step 2: cook until done."]),
                f"www.example.com/recipes/{i}",
                "Gathered",
                ner,
            )
        )
    nxt = len(RECIPES)
    names = RECIPES[0][1]
    # Bad rows (each expected to be dropped for the stated reason):
    rows += [
        row(nxt, "Exact Duplicate", json.dumps(raw_lines(names)), json.dumps(["x"]), "www.e.com/d", "Gathered", json.dumps(names)),
        row(nxt + 1, "Garlic Chicken and Rice", json.dumps(raw_lines(names)), json.dumps(["x"]), "www.e.com/d2", "Gathered", json.dumps(names)),  # duplicate of row 0
        row(nxt + 2, "Broken List", "not a list at all", json.dumps(["x"]), "www.e.com/b", "Gathered", "[]"),
        row(nxt + 3, "", json.dumps(["1 cup rice", "2 eggs"]), json.dumps(["x"]), "www.e.com/t", "Gathered", "[]"),
        row(nxt + 4, "No Directions", json.dumps(["1 cup rice", "2 eggs"]), "[]", "www.e.com/n", "Gathered", "[]"),
        row(nxt + 5, "No Link", json.dumps(["1 cup rice", "2 eggs"]), json.dumps(["x"]), "", "Gathered", "[]"),
        row(nxt + 6, "One Ingredient", json.dumps(["1 cup rice"]), json.dumps(["x"]), "www.e.com/o", "Gathered", json.dumps(["rice"])),
        row(nxt + 7, "Evil Literal", "__import__('os').system('echo pwned')", json.dumps(["x"]), "www.e.com/e", "Gathered", "[]"),
        # Python-literal (single quotes) list: valid through the ast.literal_eval fallback.
        row(nxt + 8, "Single Quoted Lists", "['1 cup rice', '2 eggs', '1 tbsp soy sauce']", "['Mix.', 'Cook.']", "https://www.example.com/q", "Recipes1M", "['rice', 'eggs', 'soy sauce']"),
    ]
    with open(OUT, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "title", "ingredients", "directions", "link", "source", "NER"])
        writer.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
