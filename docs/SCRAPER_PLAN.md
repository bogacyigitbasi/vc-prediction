# Data Scraper Pipeline — Modern VC Investment Dataset

> **Goal:** Generate `investments.csv`, `rounds.csv`, and `companies.csv` in the same schema as Crunchbase 2015, but with 2020–2026 data from publicly available sources.

---

## Target Output Schema

```
investments.csv
├── company_permalink        (unique company ID)
├── company_name
├── company_category_list    (sector/industry)
├── company_country_code
├── company_state_code
├── company_region
├── company_city
├── investor_permalink       (unique investor ID)
├── investor_name
├── investor_country_code
├── investor_state_code
├── investor_region
├── investor_city
├── funding_round_permalink  (unique round ID)
├── funding_round_type       (seed, venture, angel, etc.)
├── funding_round_code       (A, B, C, D...)
├── funded_at                (YYYY-MM-DD)
└── raised_amount_usd

rounds.csv
├── company_permalink
├── company_name
├── company_category_list
├── company_country_code
├── company_state_code
├── company_region
├── company_city
├── funding_round_permalink
├── funding_round_type
├── funding_round_code
├── funded_at
└── raised_amount_usd

companies.csv
├── permalink
├── name
├── homepage_url
├── category_list
├── funding_total_usd
├── status                   (operating, acquired, closed, ipo)
├── country_code
├── state_code
├── region
├── city
├── funding_rounds           (count)
├── founded_at
├── first_funding_at
└── last_funding_at
```

---

## Data Sources (Ranked by Feasibility)

### 1. SEC EDGAR — Form D Filings (BEST SOURCE)

**What:** Every US company raising private capital under Regulation D must file with the SEC. This is **public government data** — no legal risk.

**Contains:**
- Company name, address, industry
- Date of first sale (≈ round date)
- Total amount sold
- Investor names (related persons / executive officers listed)
- Type of securities offered

**Coverage:** ~30,000+ new filings per year

**URL:** `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=D&dateb=&owner=include&count=100`

**Steps:**
```
1. Scrape EDGAR full-text search index for Form D filings (2020–2026)
2. Parse XML/HTML for each filing:
   - Company: name, address, industry code (SIC)
   - Round: date of first sale, total offering amount
   - People: related persons (often investors / directors)
3. Deduplicate companies across multiple filings (same company files amendment per round)
4. Map SIC codes → category_list
5. Infer round_type from amount + sequence:
   - < $2M first filing → seed
   - $2M–$15M → Series A
   - $15M–$50M → Series B (approximate heuristic)
6. Output: investments.csv, rounds.csv, companies.csv
```

**API:** SEC EDGAR XBRL API is free, no key needed, rate limit 10 req/sec.

---

### 2. TechCrunch / VentureBeat / Funding News

**What:** Funding announcement articles consistently follow a structure: "[Company] raises $[X]M in [Round Type] led by [Investor], with participation from [Investor2, Investor3]."

**Contains:**
- Company name
- Amount raised
- Round type (Seed, Series A, B, C)
- Lead investor(s)
- Participating investors
- Date (article publication)
- Sometimes: valuation, sector

**Steps:**
```
1. Scrape TechCrunch /category/fundraising/ archive (paginated)
2. For each article:
   a. Extract structured data from headline + first paragraph
   b. Use regex patterns for amount, round type, company
   c. Use NER or LLM (local, e.g. Llama) to extract investor names
3. Deduplicate companies (normalize names)
4. Cross-reference with SEC EDGAR for validation
5. Output: investments.csv rows (one per investor mention)
```

**Legal:** Public articles, scraping for research is generally fine (hiQ v. LinkedIn precedent).

---

### 3. Crunchbase Basic API (Free Academic Tier)

**What:** Crunchbase offers free API access for academic research with limited daily calls.

**Contains:** Full structured data — companies, rounds, investors, relationships.

**Steps:**
```
1. Apply for Crunchbase Research Access:
   https://about.crunchbase.com/partners/crunchbase-for-research/
2. Use basic API endpoints:
   - /organizations (company lookup)
   - /funding_rounds (round details)
   - /people (investor profiles)
3. Rate-limited: ~200 calls/day on free tier
4. Strategy: fetch only companies that filed Form D (from source #1)
   to enrich with Crunchbase category + investor details
5. Output: direct mapping to target schema
```

**Note:** Slow due to rate limits, but highest data quality.

---

### 4. OpenVC Investor Database

**What:** Free database of 16,000+ investors with portfolio info.

**URL:** `https://www.openvc.app/investor-database`

**Contains:**
- Investor name, type (VC, Angel, CVC)
- Stages they invest in
- Sectors
- Notable portfolio companies
- Geography

**Steps:**
```
1. Scrape investor listing pages (paginated)
2. For each investor, extract:
   - Name, type, location
   - Investment stages
   - Portfolio companies (names only, no dates/amounts)
3. Use as ENRICHMENT layer — match investor names from other sources
4. Fills: investor_country_code, investor_region, investor_city
```

**Limitation:** No per-round data. Use for investor node features only.

---

### 5. Y Combinator / Accelerator Public Directories

**What:** YC, Techstars, 500 Startups publish their portfolio.

**Contains:**
- Company name, batch/year, sector, status, description
- Sometimes: funding stage, team size

**Steps:**
```
1. Scrape YC directory: https://www.ycombinator.com/companies
2. For each company: name, batch, industry, status, description
3. Cross-reference with SEC EDGAR:
   - Match company names
   - YC batch date ≈ seed round date
   - "YC" becomes an investor_name in investments.csv
4. Similarly for Techstars, 500 Startups
```

---

### 6. GitHub: "Openbook" VC Database

**URL:** `https://github.com/iloveitaly/openbook`

**What:** Open-source scraper + database of VC firms with team members.

**Steps:**
```
1. Clone repo, examine database schema
2. Extract: firm names, partners, investment focus
3. Use as investor metadata enrichment
```

---

## Implementation Architecture

```
scrapers/
├── README.md
├── requirements.txt
├── config.yaml                 # API keys, rate limits, date ranges
├── src/
│   ├── edgar/
│   │   ├── fetch_filings.py    # Download Form D XML from EDGAR
│   │   ├── parse_form_d.py     # Extract structured data from filings
│   │   └── infer_rounds.py     # Heuristic round type classification
│   ├── news/
│   │   ├── scrape_techcrunch.py
│   │   ├── extract_funding.py  # NER / regex extraction
│   │   └── normalize.py        # Company name normalization
│   ├── crunchbase/
│   │   ├── api_client.py       # Rate-limited API wrapper
│   │   └── enrich.py           # Match + enrich from other sources
│   ├── accelerators/
│   │   ├── scrape_yc.py
│   │   └── scrape_techstars.py
│   └── pipeline/
│       ├── merge.py            # Combine all sources
│       ├── deduplicate.py      # Entity resolution (fuzzy matching)
│       ├── validate.py         # Schema validation + quality checks
│       └── export.py           # Output final CSVs in target schema
├── data/
│   ├── raw/                    # Source-specific raw dumps
│   ├── interim/                # Per-source cleaned data
│   └── output/                 # Final merged investments.csv, rounds.csv, companies.csv
└── tests/
    ├── test_edgar_parser.py
    ├── test_entity_resolution.py
    └── test_schema_validation.py
```

---

## Execution Order

```
Phase 1: SEC EDGAR (Week 1) ─── highest volume, most reliable
   │
   ├── Fetch all Form D filings 2020-2026
   ├── Parse company + round + amount
   ├── Heuristic round classification
   └── Output: ~50,000+ rounds, partial investor names

Phase 2: News Scraping (Week 2) ─── fills investor names
   │
   ├── TechCrunch funding archive
   ├── NER extraction for investors
   ├── Match to EDGAR companies
   └── Output: investor→company edges with named investors

Phase 3: Enrichment (Week 3) ─── fills metadata gaps
   │
   ├── Crunchbase API (limited calls for validation)
   ├── OpenVC (investor metadata)
   ├── Accelerator directories (YC, Techstars)
   └── Output: enriched node features

Phase 4: Merge & Validate (Week 3-4) ─── final dataset
   │
   ├── Entity resolution (fuzzy company/investor matching)
   ├── Schema validation against target format
   ├── Quality metrics (coverage, completeness, duplicates)
   └── Output: investments.csv + rounds.csv + companies.csv
```

---

## Entity Resolution Strategy

The hardest part is matching the same company/investor across sources:

```
"Stripe, Inc." (EDGAR) = "Stripe" (TechCrunch) = "stripe" (YC)
"a]6z" (news) = "Andreessen Horowitz" (EDGAR) = "a16z" (OpenVC)
```

**Approach:**
1. Normalize: lowercase, strip Inc/LLC/Corp, remove punctuation
2. Fuzzy match: Levenshtein distance + token sort ratio (fuzzywuzzy)
3. Domain match: if homepage_url is same → same company
4. Manual review: flag low-confidence matches for human review
5. Permalink generation: slugified canonical name as ID

---

## Legal / Ethical Notes

| Source | Legal Status | Notes |
|--------|-------------|-------|
| SEC EDGAR | Public government data | Fully legal, no restrictions |
| TechCrunch | Public articles | Research use OK, respect robots.txt |
| Crunchbase API | API terms apply | Academic tier is explicitly permitted |
| OpenVC | Public database | Free to use, check ToS |
| YC Directory | Public website | Research use, rate limit respectfully |

---

## Expected Output Size

| Metric | Conservative | Optimistic |
|--------|-------------|-----------|
| Companies | 20,000 | 50,000+ |
| Investors | 5,000 | 15,000+ |
| Investment edges | 40,000 | 120,000+ |
| Date range | 2020–2026 | 2018–2026 |
| Coverage | US-heavy | US + EU + Asia |

---

## Key Decisions to Make Before Starting

1. **Scope:** US-only first, or international from the start?
2. **LLM for NER:** Local (Llama 3) vs. API (Claude) for investor name extraction from articles?
3. **Storage:** SQLite during scraping, CSV for final export?
4. **Incremental vs. batch:** One-shot scrape, or set up for continuous updates?
5. **Separate repo or subfolder?** (Recommend: separate repo, shared output schema)
