"""Ingestion for the Amazon Product Dataset 2020 (PromptCloud, Kaggle).

Raw CSV  ->  cleaned, normalized DataFrame  ->  data/processed/products.parquet

Design notes
------------
* The raw file has 28 columns, ~13 of them empty. Every step here tolerates a
  missing or empty column instead of assuming the brief's ideal schema.
* The dataset has no star ratings, brands, ingredients or descriptions. We:
    - keep `rating` as a nullable column (filled later from live web if needed),
    - guess `brand` from the title (`brand_source` records how it was found),
    - parse ASIN and manufacturer-recommended age out of the squashed spec text,
    - mine piece count, players, materials and attribute / hazard tags from
      title + features, using the domain pack chosen by `SliceConfig`.
* The chosen slice is Toys & Games (~6.6k of 10k rows). A cleaning pack is
  kept as `SliceConfig.household_cleaning()`, but this sample has essentially
  no cleaning products (the filter's ~10 hits are hampers and straws).
* Every row gets a stable `doc_id` used for citations everywhere downstream.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

# Target schema written to products.parquet (order matters for readability).
TARGET_COLUMNS = [
    "id", "doc_id", "sku", "title", "brand", "brand_source",
    "category", "category_path", "price", "price_max",
    "age_min_years", "age_max_years", "age_source", "piece_count", "players_min", "players_max",
    "size_oz", "price_per_oz", "rating", "features", "description",
    "ingredients", "materials", "attributes", "hazards",
    "url", "image_url", "embed_text",
]

# Boilerplate Amazon prepends to many "About Product" fields.
_BOILERPLATE = re.compile(
    r"^\s*make sure this fits by entering your model number\.?\s*$", re.I
)


# ---- Domain packs: what to tag in the text, per slice ----------------------

# Cleaning pack (kept for reference; this sample has essentially no cleaners).
CLEANING_ATTRIBUTES: dict[str, str] = {
    "eco_friendly": r"eco[- ]?friendly|environmentally friendly|green seal|safer choice",
    "plant_based": r"plant[- ]based|plant[- ]derived|botanical|derived from plants",
    "biodegradable": r"biodegradable|compostable",
    "non_toxic": r"non[- ]?toxic",
    "fragrance_free": r"fragrance[- ]free|unscented|free (of|from) (dyes and )?fragrances?",
    "ammonia_free": r"ammonia[- ]free|no ammonia|without ammonia",
    "bleach_free": r"bleach[- ]free|no bleach|without bleach",
    "stainless_safe": r"stainless[- ]steel",
    "disinfectant": r"disinfect|kills 99(\.9)?%|antibacterial|sanitiz",
    "refill": r"\brefill",
}
CLEANING_HAZARDS: dict[str, str] = {
    "contains_bleach": r"(?<!no )(?<!free of )\b(sodium hypochlorite|chlorine bleach|with bleach|contains bleach)\b",
    "contains_ammonia": r"(?<!no )(?<!free )\bammonia\b(?![- ]free)",
    "contains_acid": r"\b(hydrochloric|muriatic|phosphoric|sulfamic|oxalic) acid\b",
    "corrosive": r"\bcorrosive\b",
    "flammable": r"\bflammable\b",
    "keep_from_children": r"keep out of (the )?reach of children",
}
CLEANING_INGREDIENTS = [
    r"plant[- ]based (surfactants?|ingredients|cleaning agents)",
    r"(coconut|corn|soy)[- ]derived \w+",
    r"essential oils?",
    r"(lemon|lavender|eucalyptus|tea tree|citrus|orange|mineral) oil",
    r"\b(vinegar|baking soda|hydrogen peroxide|citric acid|alcohol|"
    r"sodium hypochlorite|enzymes?|surfactants?)\b",
]

# Toys pack (the chosen slice).
TOY_ATTRIBUTES: dict[str, str] = {
    "stem_educational": r"\bstem\b|educational|\blearning\b|science|\bmath\b|coding|engineering",
    "building": r"building (set|kit|blocks?|toy)|construction (set|toy)|\bbricks?\b|model kit",
    "arts_crafts": r"\bcrafts?\b|painting|paint set|drawing|coloring|\bbeads?\b|slime",
    "puzzle": r"\bpuzzles?\b|jigsaw",
    "board_or_card_game": r"board game|card game|party game|family game|strategy game",
    "pretend_play": r"pretend play|role play|dress[- ]up|costume|play kitchen|playset",
    "plush": r"\bplush\b|stuffed (animal|toy)|\bsoft toy",
    "remote_control": r"remote[- ]control(led)?|\brc\b|app[- ]controlled",
    "outdoor": r"\boutdoor\b|backyard|\bpool\b|\bbeach\b",
    "wooden": r"\bwood(en)?\b",
    "batteries_required": r"batter(y|ies) (are )?(required|not included)|requires? \d* ?(aa|aaa|c|d)\b",
    "batteries_included": r"batter(y|ies) (are )?included",
    "no_batteries": r"no batteries (needed|required)",
    "eco_friendly": r"eco[- ]?friendly|sustainabl|recycled|biodegradable",
    "non_toxic": r"non[- ]?toxic|bpa[- ]free|phthalate[- ]free|lead[- ]free",
    "washable": r"washable",
    "collectible": r"collectible|collector'?s",
}
# Toy hazards feed the safety guardrail (age-appropriateness, choking, magnets, button cells).
TOY_HAZARDS: dict[str, str] = {
    "choking_hazard": r"choking hazard|(?<!no )small parts|(?<!no )small balls?|\bmarbles?\b",
    "not_for_under_3": r"not (suitable )?for (children )?under (3|three|36 months)|not for children under 3",
    "magnets": r"\bmagnet(s|ic)?\b",
    "button_battery": r"button (cell )?batter|coin (cell )?batter|lithium coin",
    "balloons": r"\bballoons?\b",
    "projectile": r"\bblasters?\b|\bdarts?\b|projectile|slingshot",
    "heat_source": r"hot glue gun|soldering iron|(?<!no )soldering required|heated nozzle|hot tip|heating element",
    "adult_supervision": r"adult supervision",
}
TOY_MATERIALS = [
    r"\b(solid |natural )?wood(en)?\b", r"\babs plastic\b", r"\bplastic\b", r"\bdie[- ]cast\b",
    r"\bmetal\b", r"\bcotton\b", r"\bpolyester\b", r"\bplush\b", r"\bfelt\b",
    r"\bsilicone\b", r"\bfoam\b", r"\brubber\b", r"\bcardboard\b", r"\bfabric\b",
]


@dataclass
class SliceConfig:
    """Which products make up the indexed slice, and what to tag in their text.

    A product is in the slice if its category path matches `category_pattern`
    (or, when `title_pattern` is set, its title matches) and it is not excluded.
    """
    name: str = "toys_and_games"
    category_pattern: str = r"^Toys & Games\b"
    title_pattern: str | None = None
    exclude_pattern: str | None = None
    attribute_patterns: dict = field(default_factory=lambda: dict(TOY_ATTRIBUTES))
    hazard_patterns: dict = field(default_factory=lambda: dict(TOY_HAZARDS))
    material_patterns: list = field(default_factory=lambda: list(TOY_MATERIALS))
    min_rows_warning: int = 150

    @classmethod
    def toys_and_games(cls) -> "SliceConfig":
        return cls()

    @classmethod
    def household_cleaning(cls) -> "SliceConfig":
        return cls(
            name="household_cleaning",
            category_pattern=(
                r"household (supplies|cleaning)|cleaning supplies|cleaners|laundry|"
                r"dishwashing|all-purpose|disinfect|air fresheners"
            ),
            title_pattern=(
                r"\b(cleaner|cleaning|cleanser|detergent|disinfect\w*|wipes?|polish|degreaser|"
                r"stain remover|scrub(ber|bing)?|sponges?|dish ?soap|laundry|sanitiz\w+|mop|broom)\b"
            ),
            exclude_pattern=(
                r"\b(toy|toys|pretend play|play ?set|costume|pipe cleaners?|lens cleaner|"
                r"screen cleaner|shampoo|toothbrush|party|plates?|cups?|napkins?)\b"
            ),
            attribute_patterns=dict(CLEANING_ATTRIBUTES),
            hazard_patterns=dict(CLEANING_HAZARDS),
            material_patterns=list(CLEANING_INGREDIENTS),
        )


# Size units -> ounces. fl oz treated ~= oz by weight (fine for ranking, not chemistry).
_UNIT_TO_OZ = {
    "oz": 1.0, "ounce": 1.0, "ounces": 1.0, "fl oz": 1.0, "fl. oz": 1.0,
    "fluid ounce": 1.0, "fluid ounces": 1.0,
    "lb": 16.0, "lbs": 16.0, "pound": 16.0, "pounds": 16.0,
    "gal": 128.0, "gallon": 128.0, "gallons": 128.0,
    "qt": 32.0, "quart": 32.0, "quarts": 32.0,
    "ml": 0.033814, "l": 33.814, "liter": 33.814, "liters": 33.814, "litre": 33.814,
    "g": 0.035274, "gram": 0.035274, "grams": 0.035274, "kg": 35.274,
}
_SIZE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*(fl\.? oz|fluid ounces?|ounces?|oz|lbs?|pounds?|"
    r"gallons?|gal|quarts?|qt|ml|liters?|litres?|l|grams?|g|kg)\b",
    re.I,
)
_PACK_RE = re.compile(
    r"(?:pack of|set of|case of)\s*(\d+)|(\d+)\s*[- ]?(?:pack|pk|count|ct)\b", re.I
)
# The raw spec text is usually squashed with no spaces, e.g.
# "ASIN:B07Q1JVZMW|Itemmodelnumber:42932|Manufacturerrecommendedage:6yearsandup"
_SPEC_ASIN = re.compile(r"asin:([a-z0-9]{10})", re.I)
_SPEC_AGE = re.compile(
    r"manufacturerrecommendedage:(\d+)(months|years)?(?:-(\d+)(months|years))?", re.I
)


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


def parse_spec_asin(spec: str) -> str | None:
    m = _SPEC_ASIN.search(_squash(spec))
    return m.group(1).upper() if m else None


def parse_spec_age(spec: str) -> tuple[float | None, float | None]:
    """'Manufacturerrecommendedage:4-15years' -> (4.0, 15.0) in years; '...:36months' -> (3.0, None)."""
    m = _SPEC_AGE.search(_squash(spec))
    if not m:
        return None, None
    lo, lo_unit, hi, hi_unit = m.groups()
    lo_unit = lo_unit or hi_unit or "years"
    to_years = lambda n, u: round(float(n) / 12, 2) if u == "months" else float(n)
    return to_years(lo, lo_unit), (to_years(hi, hi_unit) if hi else None)




# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def snake(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _text(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v).strip()


def parse_price(raw) -> tuple[float | None, float | None]:
    """'$12.49' -> (12.49, 12.49); '$9.99 - $24.99' -> (9.99, 24.99)."""
    nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", _text(raw))]
    nums = [n for n in nums if 0 < n < 100_000]
    if not nums:
        return None, None
    return min(nums), max(nums)


def parse_size_oz(*texts: str) -> float | None:
    """Total ounces from strings like '32 fl oz, Pack of 3' -> 96.0."""
    blob = " ".join(t for t in texts if t)
    m = _SIZE_RE.search(blob)
    if not m:
        return None
    qty, unit = float(m.group(1)), m.group(2).lower().replace(".", "").strip()
    unit = "fl oz" if unit.startswith("fl") else unit
    per = _UNIT_TO_OZ.get(unit)
    if per is None or qty <= 0:
        return None
    oz = qty * per
    pk = _PACK_RE.search(blob)
    if pk:
        n = int(pk.group(1) or pk.group(2))
        if 1 < n <= 48:
            oz *= n
    return round(oz, 2) if 0.1 <= oz <= 5000 else None


def split_features(raw) -> list[str]:
    parts = [p.strip(" .") for p in re.split(r"\s*\|\s*", _text(raw))]
    return [p for p in parts if p and not _BOILERPLATE.match(p) and len(p) > 3]


def category_parts(raw) -> list[str]:
    return [p.strip() for p in _text(raw).split("|") if p.strip()]


def find_tags(text: str, patterns: dict[str, str]) -> list[str]:
    return sorted(tag for tag, pat in patterns.items() if re.search(pat, text, re.I))


_PIECES_RE = re.compile(r"(\d[\d,]{0,5})\s*-?\s*(?:pc|pcs|piece|pieces)\b", re.I)
_PLAYERS_RE = re.compile(r"(\d{1,2})\s*(?:-|to|–)\s*(\d{1,2})\s*players?|(\d{1,2})\s*\+?\s*players?", re.I)


_TEXT_AGE = re.compile(
    r"\bages?\s*(\d{1,2})\s*(?:(?:-|to|–)\s*(\d{1,2}))?\s*(?:\+|and up|years|yrs|\b)"
    r"|\b(\d{1,2})\s*(?:years?|yrs?)\s*(?:old\s*)?(?:and up|\+|& up)",
    re.I,
)


def parse_text_age(*texts: str) -> tuple[float | None, float | None]:
    """'Ages 3+' -> (3, None); 'for ages 8-12' -> (8, 12); '6 years and up' -> (6, None)."""
    for t in texts:
        for m in _TEXT_AGE.finditer(t or ""):
            lo = m.group(1) or m.group(3)
            hi = m.group(2)
            lo_f = float(lo)
            hi_f = float(hi) if hi else None
            if 0 <= lo_f <= 18 and (hi_f is None or lo_f <= hi_f <= 99):
                return lo_f, hi_f
    return None, None


def parse_piece_count(*texts: str) -> int | None:
    for t in texts:
        m = _PIECES_RE.search(t or "")
        if m:
            n = int(m.group(1).replace(",", ""))
            if 2 <= n <= 20000:
                return n
    return None


def parse_players(*texts: str) -> tuple[int | None, int | None]:
    for t in texts:
        m = _PLAYERS_RE.search(t or "")
        if m:
            if m.group(1):
                lo, hi = int(m.group(1)), int(m.group(2))
                if 1 <= lo <= hi <= 30:
                    return lo, hi
            elif m.group(3):
                n = int(m.group(3))
                if 1 <= n <= 30:
                    return n, None
    return None, None


def find_phrases(text: str, patterns: list[str], limit: int = 6) -> str:
    seen: list[str] = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.I):
            phrase = m.group(0).lower().strip()
            if phrase not in seen:
                seen.append(phrase)
    return ", ".join(seen[:limit])


_BRAND_STOP = {
    "for", "with", "and", "the", "of", "in", "by", "to", "a", "an", "-", "|", ",", ":",
    "kit", "set", "toy", "toys", "game", "games", "puzzle", "skin", "pack", "kids",
    "compatible", "new", "official", "premium", "deluxe", "mini", "large", "small",
}


def _brand_tokens(title: str, n: int = 3) -> list[str]:
    toks = re.split(r"\s+", title.strip())[:n]
    return [t.strip(",;:|") for t in toks if t.strip(",;:|")]


def guess_brands(titles: pd.Series, min_count: int = 3, keep_ratio: float = 0.5) -> pd.Series:
    """Brand = the longest 1-3 word title prefix that many titles share.

    The dataset has no brand field, so this is a heuristic and is labelled
    `brand_source="title_guess"`. 'Melissa & Doug Wooden...' -> 'Melissa & Doug'
    because most titles starting 'Melissa' continue '& Doug'; 'Funko Pop! ...'
    stays 'Funko' because 'Funko' is followed by many different words. A prefix
    seen only once falls back to the first word (plus 'The'/'A' lead-ins).
    Casing is unified to the most common spelling ('amscan' -> 'Amscan').
    """
    from collections import Counter
    toks = titles.fillna("").map(_brand_tokens)
    counts: Counter = Counter()
    for t in toks:
        for k in range(1, len(t) + 1):
            counts[" ".join(t[:k]).lower()] += 1

    def pick(t: list[str]) -> str:
        if not t:
            return ""
        k_best = 1
        if t[0].lower() in {"the", "a"} and len(t) > 1:
            k_best = 2
        base = counts[" ".join(t[:k_best]).lower()]
        for k in range(k_best + 1, len(t) + 1):
            word = t[k - 1]
            if word != "&" and (word.lower() in _BRAND_STOP or re.fullmatch(r"[\d\W]+", word)):
                break
            c = counts[" ".join(t[:k]).lower()]
            if c >= min_count and c >= keep_ratio * base:
                k_best = k
            else:
                break
        return " ".join(t[:k_best]).rstrip("&- ").strip()

    picked = toks.map(pick)
    casing: dict[str, Counter] = {}
    for b in picked:
        casing.setdefault(b.lower(), Counter())[b] += 1
    canon = {k: c.most_common(1)[0][0] for k, c in casing.items()}
    return picked.map(lambda b: canon[b.lower()])


def make_doc_id(uid: str) -> str:
    return "amz2020-" + hashlib.sha1(uid.encode()).hexdigest()[:10]


# --------------------------------------------------------------------------
# Pipeline steps
# --------------------------------------------------------------------------

def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=True)
    df.columns = [snake(c) for c in df.columns]
    return df


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Fill rate and an example value per column."""
    rows = []
    for c in df.columns:
        s = df[c].dropna().astype(str).str.strip()
        s = s[s != ""]
        rows.append({
            "column": c,
            "filled": len(s),
            "fill_rate": round(len(s) / max(len(df), 1), 3),
            "example": (s.iloc[0][:80] if len(s) else ""),
        })
    return pd.DataFrame(rows).sort_values("fill_rate", ascending=False, ignore_index=True)


def top_categories(df: pd.DataFrame, depth: int = 2, n: int = 25) -> pd.Series:
    col = df.get("category", pd.Series(dtype=str))
    paths = col.fillna("").map(lambda s: " | ".join(category_parts(s)[:depth]) or "(none)")
    return paths.value_counts().head(n)


def normalize(df: pd.DataFrame, cfg: "SliceConfig | None" = None) -> pd.DataFrame:
    """Raw snake_cased frame -> target schema (all rows, no slicing).

    `cfg` picks the domain pack used for attribute / hazard / material tags.
    """
    cfg = cfg or SliceConfig()
    g = lambda c: df[c] if c in df.columns else pd.Series([""] * len(df), index=df.index)

    out = pd.DataFrame(index=df.index)
    out["id"] = g("uniq_id").map(_text)
    out.loc[out["id"] == "", "id"] = [f"row{i}" for i in out.index[out["id"] == ""]]
    out["doc_id"] = out["id"].map(make_doc_id)
    sku = g("asin").map(_text)
    sku = sku.where(sku != "", g("sku").map(_text))
    spec_asin = g("product_specification").map(lambda t: parse_spec_asin(_text(t)) or "")
    sku = sku.where(sku != "", spec_asin)
    out["sku"] = sku.replace("", None)
    ages = g("product_specification").map(lambda t: parse_spec_age(_text(t)))
    out["age_min_years"] = pd.to_numeric(pd.Series([a[0] for a in ages], index=df.index), errors="coerce")
    out["age_max_years"] = pd.to_numeric(pd.Series([a[1] for a in ages], index=df.index), errors="coerce")
    out["title"] = g("product_name").map(_text)

    raw_brand = g("brand_name").map(_text)
    title_brand = guess_brands(out["title"])
    brands = []
    for i, title in out["title"].items():
        if raw_brand[i]:
            brands.append((raw_brand[i], "dataset"))
            continue
        # The spec text never holds a brand ("Discontinuedbymanufacturer:Yes" is not one).
        brands.append((title_brand[i], "title_guess" if title_brand[i] else "missing"))
    out["brand"] = [b for b, _ in brands]
    out["brand_source"] = [s for _, s in brands]

    cats = g("category").map(category_parts)
    out["category_path"] = cats
    out["category"] = cats.map(lambda p: p[-1] if p else "")

    prices = g("selling_price").map(parse_price)
    list_prices = g("list_price").map(parse_price)
    out["price"] = [p[0] if p[0] is not None else lp[0] for p, lp in zip(prices, list_prices)]
    out["price_max"] = [p[1] if p[1] is not None else lp[1] for p, lp in zip(prices, list_prices)]

    out["features"] = g("about_product").map(split_features)
    out["description"] = g("product_description").map(_text)

    size_src = g("size_quantity_variant").map(_text)
    out["size_oz"] = [
        parse_size_oz(t, s, " ".join(f)) for t, s, f in zip(out["title"], size_src, out["features"])
    ]
    out["price_per_oz"] = [
        round(p / s, 3) if (p and s) else None for p, s in zip(out["price"], out["size_oz"])
    ]

    # The dataset ships no ratings; keep the column so the schema is stable.
    if "rating" in df.columns:
        out["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    else:
        out["rating"] = np.nan

    blob = (
        out["title"] + " . " + out["features"].map(" . ".join) + " . " + out["description"]
        + " . " + g("ingredients").map(_text) + " . " + g("direction_to_use").map(_text)
    )
    out["ingredients"] = g("ingredients").map(_text)          # empty in this dataset
    out["materials"] = blob.map(lambda t: find_phrases(t, cfg.material_patterns))
    out["attributes"] = blob.map(lambda t: find_tags(t, cfg.attribute_patterns))
    out["hazards"] = blob.map(lambda t: find_tags(t, cfg.hazard_patterns))

    feats = out["features"].map(" ".join)
    # Age fallback: many listings say "Ages 8+" / "for ages 3 and up" in title or bullets.
    text_ages = [parse_text_age(t, f) for t, f in zip(out["title"], feats)]
    need = out["age_min_years"].isna()
    out["age_source"] = np.where(out["age_min_years"].notna(), "spec", None)
    nan = lambda v: np.nan if v is None else v
    if need.any():
        out.loc[need, "age_min_years"] = [nan(a[0]) for a, n in zip(text_ages, need) if n]
        out.loc[need, "age_max_years"] = [nan(a[1]) for a, n in zip(text_ages, need) if n]
    out.loc[need & out["age_min_years"].notna(), "age_source"] = "text"
    out["piece_count"] = [parse_piece_count(t, f) for t, f in zip(out["title"], feats)]
    players = [parse_players(t, f) for t, f in zip(out["title"], feats)]
    out["players_min"] = [p[0] for p in players]
    out["players_max"] = [p[1] for p in players]

    out["url"] = g("product_url").map(_text).replace("", None)
    out["image_url"] = g("image").map(lambda s: _text(s).split("|")[0].strip()).replace("", None)
    out["embed_text"] = [build_embed_text(r) for r in out.itertuples()]

    out = out[out["title"] != ""].drop_duplicates(subset=["title", "price"]).reset_index(drop=True)
    return out[TARGET_COLUMNS]


def build_embed_text(r, max_chars: int = 1500) -> str:
    """Title + features + description, the text that gets embedded in Phase 2."""
    parts = [r.title]
    if r.brand:
        parts.append(f"Brand: {r.brand}")
    if r.category_path:
        parts.append("Category: " + " > ".join(r.category_path))
    if r.features:
        parts.append("Features: " + "; ".join(r.features))
    if r.description:
        parts.append(r.description)
    if r.age_min_years is not None and r.age_min_years == r.age_min_years:
        hi = r.age_max_years
        parts.append(f"Recommended age: {r.age_min_years:g}" + (f"-{hi:g}" if hi == hi and hi is not None else "+") + " years")
    if r.piece_count == r.piece_count and r.piece_count is not None:
        parts.append(f"Pieces: {int(r.piece_count)}")
    if r.players_min == r.players_min and r.players_min is not None:
        hi = r.players_max
        parts.append(f"Players: {int(r.players_min)}" + (f"-{int(hi)}" if hi == hi and hi is not None else "+"))
    if r.materials:
        parts.append(f"Materials: {r.materials}")
    if r.attributes:
        parts.append("Attributes: " + ", ".join(a.replace("_", " ") for a in r.attributes))
    return "\n".join(parts)[:max_chars]


def select_slice(products: pd.DataFrame, cfg: "SliceConfig | None" = None) -> pd.DataFrame:
    cfg = cfg or SliceConfig()

    def has(s: pd.Series, pat: str) -> pd.Series:
        # Make groups non-capturing so pandas doesn't warn about match groups.
        return s.str.contains(re.sub(r"\((?!\?)", "(?:", pat), case=False, regex=True)

    path = products["category_path"].map(" | ".join)
    none = pd.Series(False, index=products.index)
    in_cat = has(path, cfg.category_pattern)
    in_title = has(products["title"], cfg.title_pattern) if cfg.title_pattern else none
    excluded = has(products["title"] + " " + path, cfg.exclude_pattern) if cfg.exclude_pattern else none
    sl = products[(in_cat | in_title) & ~excluded].copy()
    sl["slice"] = cfg.name
    return sl.reset_index(drop=True)


@dataclass
class IngestReport:
    raw_rows: int
    normalized_rows: int
    slice_rows: int
    slice_name: str
    coverage: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


def coverage(df: pd.DataFrame) -> dict:
    def rate(mask) -> float:
        return round(float(mask.mean()), 3) if len(df) else 0.0
    return {
        "price": rate(df["price"].notna()),
        "sku": rate(df["sku"].notna()),
        "age_min_years": rate(df["age_min_years"].notna()),
        "piece_count": rate(df["piece_count"].notna()),
        "players": rate(df["players_min"].notna()),
        "materials": rate(df["materials"].str.len() > 0),
        "has_hazard_tag": rate(df["hazards"].map(len) > 0),
        "size_oz": rate(df["size_oz"].notna()),
        "price_per_oz": rate(df["price_per_oz"].notna()),
        "rating": rate(df["rating"].notna()),
        "brand_guessed_from_title": rate(df["brand_source"] == "title_guess"),
        "has_attributes": rate(df["attributes"].map(len) > 0),
        "has_features": rate(df["features"].map(len) > 0),
        "url": rate(df["url"].notna()),
    }


def run(raw_csv: str | Path, out_dir: str | Path = "data/processed",
        cfg: "SliceConfig | None" = None) -> IngestReport:
    """Full pipeline. Writes products_all.parquet, products.parquet, ingest_report.json."""
    cfg = cfg or SliceConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(raw_csv)
    allp = normalize(raw, cfg)
    sl = select_slice(allp, cfg)

    allp.to_parquet(out_dir / "products_all.parquet", index=False)
    sl.to_parquet(out_dir / "products.parquet", index=False)

    rep = IngestReport(len(raw), len(allp), len(sl), cfg.name, coverage(sl))
    if len(sl) < cfg.min_rows_warning:
        rep.warnings.append(
            f"Slice '{cfg.name}' has only {len(sl)} rows; consider widening SliceConfig "
            "or adding a second category so retrieval has enough to compare."
        )
    if rep.coverage.get("rating", 0) == 0:
        rep.warnings.append("No ratings in the dataset: rank by price/attributes, or fetch ratings via web.search.")
    (out_dir / "ingest_report.json").write_text(rep.to_json())
    return rep
