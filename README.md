# Voice Shopping Agent

A voice-to-voice, multi-agent assistant for toy and game discovery. You speak a request ("a STEM building kit for an 8-year-old under $30"), and it plans a search over a private Amazon catalog and the live web, then answers with grounded, cited picks, both spoken and on screen.

**Pipeline:** Voice → ASR → Router → Planner → Retriever (RAG + MCP tools) → Answerer/Critic → TTS, evaluated end to end with ProofAgent.

## Status

| Phase | What | State |
|---|---|---|
| 1 | Data ingestion → `products.parquet` (Toys & Games, 6,658 products) | ✅ pipeline + executed notebook |
| 2 | Embeddings index + MCP server (`rag.search`, `web.search`) | ⏳ |
| 3 | LangGraph agents | ⏳ |
| 4 | Voice (Whisper, TTS) + Streamlit UI | ⏳ |
| 5 | ProofAgent evaluation | ⏳ |
| 6 | Docs, demo | ⏳ |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # Phase 1 only needs: pandas pyarrow jupyter pytest
cp .env.example .env                   # fill in keys as phases need them
```

## Phase 1: build the catalog

1. Download the **Amazon Product Dataset 2020** CSV from Kaggle (PromptCloud, ~10k rows) into `data/raw/`.
2. Build the catalog with either option:
   ```bash
   python scripts/build_data.py                 # scripted
   jupyter notebook notebooks/01_ingest.ipynb   # same steps, with profiling and charts
   ```
3. The build writes the following to `data/processed/`:
   - `products.parquet`: the Toys & Games slice (6,658 products) that gets indexed
   - `products_all.parquet`: every product, normalized
   - `ingest_report.json`: row counts, field coverage and warnings

Run the tests with `PYTHONPATH=src pytest -q`.

### Why Toys & Games, not Household Cleaning

The brief suggests Household Cleaning as an example slice, but this 10k sample has almost no cleaning products. Health & Household has 23 products, and a cleaning filter's ~10 hits are laundry hampers and drinking straws. Toys & Games has 6,662 products (two-thirds of the file), with good price, ASIN and recommended-age coverage. The cleaning configuration is kept as `SliceConfig.household_cleaning()` for reference.

| Top-level category | Products |
|---|---|
| Toys & Games | 6,662 |
| (no category) | 830 |
| Home & Kitchen | 708 |
| Clothing, Shoes & Jewelry | 630 |
| Sports & Outdoors | 540 |
| Baby Products | 214 |
| Health & Household | 23 |

### Catalog schema (`products.parquet`)

| Column | Notes | Toys coverage |
|---|---|---|
| `id`, `doc_id` | `doc_id` (`amz2020-…`) is the citation key used everywhere downstream | 100% |
| `sku` | ASIN, parsed from the spec text | 87% |
| `title`, `brand`, `brand_source` | Brand is guessed from shared title prefixes (`title_guess`); match it fuzzily | 100% (guessed) |
| `category`, `category_path` | Leaf category and the full path | 100% |
| `price`, `price_max` | Selling price; a range is split into min and max | 99% |
| `age_min_years`, `age_max_years`, `age_source` | Manufacturer recommended age from the spec, else "Ages 8+" from the text | 74% |
| `piece_count`, `players_min`, `players_max` | Parsed from the title or features | 12% / 4% |
| `size_oz`, `price_per_oz` | Size normalized to ounces (mainly for the cleaning pack) | 5% |
| `rating` | **Empty, because the dataset has no ratings.** It's kept so the schema stays stable. | 0% |
| `features`, `description` | Cleaned feature bullets. Description is empty in this dataset. | 97% / 0% |
| `ingredients` | The dataset's field, which is empty | 0% |
| `materials` | Material phrases mined from the text (wood, ABS plastic, plush…) | 32% |
| `attributes` | Tags such as `stem_educational`, `building`, `puzzle`, `batteries_required`, `non_toxic` | 56% |
| `hazards` | Tags such as `choking_hazard`, `magnets`, `balloons`, `button_battery`, `projectile`, used by the safety guardrail | 5% |
| `url`, `image_url` | Product page and image links | 100% |
| `embed_text` | Title, brand, category, features, age, pieces, players, materials and tags: the text that gets embedded | 100% |

### Known data limitations

- 13 of the 28 raw columns are empty, including brand, ingredients, description and list price.
- There are no star ratings and no reviews, so `reviews.parquet` is skipped (it's optional in the brief). Ranking uses constraint fit and price. Live ratings can come from `web.search` and must be labelled as live.
- Brand is a heuristic guess from the title.
- Hazard tags only cover what the listing text says. A missing tag does not mean a toy is safe, and a missing age is treated as unknown, not as suitable for all ages.
- About 80 items cost over $300 (playsets, lab equipment). They stay in the catalog, and budget filters handle them.

The agent's prompts must not claim more than the catalog actually holds.

## Repository layout

```
data/raw/            raw Kaggle CSV (gitignored)
data/processed/      parquet outputs (gitignored, rebuilt by scripts)
notebooks/           01_ingest.ipynb
scripts/             build_data.py (later: build_index.py, run_mcp.py)
src/voiceshop/       package: data/, later rag/, mcp/, agents/, voice/, ui/
prompts/             prompt disclosure, one file per LangGraph node
tests/               pytest + synthetic fixture
docs/                proposal, architecture, evaluation write-ups
```
