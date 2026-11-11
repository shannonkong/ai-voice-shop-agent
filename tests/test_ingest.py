"""Tests for the ingestion pipeline, run on a 7-row synthetic fixture
(tests/fixtures/sample_raw.csv) that mimics the real CSV's 28 columns:
3 cleaners, 1 toy cleaning set, 1 shoe, and 2 toys."""
from pathlib import Path

import pandas as pd

from voiceshop.data import ingest

FIXTURE = Path(__file__).parent / "fixtures" / "sample_raw.csv"


def test_parse_price():
    assert ingest.parse_price("$12.49") == (12.49, 12.49)
    assert ingest.parse_price("$9.99 - $24.99") == (9.99, 24.99)
    assert ingest.parse_price("") == (None, None)


def test_parse_size_oz():
    assert ingest.parse_size_oz("Cleaner 16 oz") == 16.0
    assert ingest.parse_size_oz("32 fl oz, Pack of 2") == 64.0
    assert ingest.parse_size_oz("1 gallon refill") == 128.0
    assert ingest.parse_size_oz("no size here") is None


def test_spec_parsing():
    spec = "ProductDimensions:1.8x5.5x5.1inches|ASIN:B07NSSSWKS|Manufacturerrecommendedage:4-15years"
    assert ingest.parse_spec_asin(spec) == "B07NSSSWKS"
    assert ingest.parse_spec_age(spec) == (4.0, 15.0)
    assert ingest.parse_spec_age("Manufacturerrecommendedage:36months") == (3.0, None)
    assert ingest.parse_spec_age("Manufacturer recommended age: 8 years and up") == (8.0, None)
    assert ingest.parse_spec_age("no age") == (None, None)


def test_piece_and_player_parsing():
    assert ingest.parse_piece_count("Woodstock Collage 500 pc Puzzle") == 500
    assert ingest.parse_piece_count("1,000 Pieces jigsaw") == 1000
    assert ingest.parse_players("Board Game, 2-6 Players") == (2, 6)
    assert ingest.parse_players("for 2+ players") == (2, None)
    assert ingest.parse_players("no players") == (None, None)


def test_toys_pipeline(tmp_path):
    rep = ingest.run(FIXTURE, tmp_path)            # default slice: Toys & Games
    allp = pd.read_parquet(tmp_path / "products_all.parquet")
    sl = pd.read_parquet(tmp_path / "products.parquet").set_index("id")

    assert list(allp.columns) == ingest.TARGET_COLUMNS
    assert rep.raw_rows == 7 and rep.slice_name == "toys_and_games"
    # Toy cleaning set + two real toys; cleaners and shoes are out.
    assert set(sl.index) == {"a3", "t1", "t2"}

    robot = sl.loc["t1"]
    assert robot["sku"] == "B0TOY00001"
    assert (robot["age_min_years"], robot["age_max_years"]) == (8.0, 12.0)
    assert robot["piece_count"] == 120
    assert {"stem_educational", "building", "batteries_required"} <= set(robot["attributes"])
    assert {"choking_hazard", "not_for_under_3"} <= set(robot["hazards"])
    assert "abs plastic" in robot["materials"]
    assert not any("make sure this fits" in f.lower() for f in robot["features"])
    assert "Recommended age: 8-12 years" in robot["embed_text"]

    game = sl.loc["t2"]
    assert (game["players_min"], game["players_max"]) == (2, 6)
    assert game["age_min_years"] == 3.0
    assert "board_or_card_game" in set(game["attributes"])

    assert sl["doc_id"].str.startswith("amz2020-").all() and sl["doc_id"].is_unique


def test_cleaning_pack_still_works(tmp_path):
    ingest.run(FIXTURE, tmp_path, ingest.SliceConfig.household_cleaning())
    sl = pd.read_parquet(tmp_path / "products.parquet").set_index("id")
    assert set(sl.index) == {"a1", "a2", "a5"}

    eco = sl.loc["a1"]
    assert {"plant_based", "ammonia_free", "stainless_safe"} <= set(eco["attributes"])
    assert "contains_ammonia" not in set(eco["hazards"])
    assert eco["price_per_oz"] == round(11.49 / 16, 3)

    bleach = sl.loc["a2"]
    assert bleach["size_oz"] == 64.0 and "contains_bleach" in set(bleach["hazards"])


def test_text_age_and_brands():
    assert ingest.parse_text_age("Puzzle, 36 Pieces, Ages 3+") == (3.0, None)
    assert ingest.parse_text_age("Great for ages 8-12") == (8.0, 12.0)
    assert ingest.parse_text_age("Suitable for 6 years and up") == (6.0, None)
    assert ingest.parse_text_age("Includes 4 balls") == (None, None)

    titles = pd.Series(["Melissa & Doug Wooden Puzzle", "Melissa & Doug Play Food",
                        "Melissa & Doug Easel", "Domez My Hero Academia 4-Piece"])
    assert list(ingest.guess_brands(titles)) == ["Melissa & Doug"] * 3 + ["Domez"]
