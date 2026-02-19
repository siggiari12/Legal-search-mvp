#!/usr/bin/env python3
"""
evaluate_retrieval.py -Research-grade vector retrieval benchmark for the Icelandic legal corpus.

Embeds each query with text-embedding-3-small, calls match_documents, and records
similarity distribution metrics.  Uses token-prefix keyword heuristics and law
reference lookup to classify each query.  Produces a structured table, per-category
breakdown, aggregate dispersion stats, and a rule-based architectural recommendation.

No LLM calls.  Summary ->stdout.  Per-query progress ->stderr.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --debug
    python scripts/evaluate_retrieval.py --k 10 --threshold-high 0.65 --threshold-mid 0.45
    python scripts/evaluate_retrieval.py --csv results/eval.csv
"""

import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from supabase import create_client


# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL        = "text-embedding-3-small"
EMBEDDING_DIM          = 1536
DEFAULT_THRESHOLD_HIGH = 0.70
DEFAULT_THRESHOLD_MID  = 0.50


# ── Evaluation query set ──────────────────────────────────────────────────────

class QueryCase(NamedTuple):
    query_text:             str
    category:               str             = "general"
    expected_law_reference: str | None      = None   # verified if found in top 3
    expected_keywords:      tuple[str, ...] = ()     # token-prefix checked vs top-1 text


QUERIES: list[QueryCase] = [
    # ── Employment ────────────────────────────────────────────────────────────
    QueryCase("Hvenær má segja starfsmanni upp?",
              category="employment",
              expected_keywords=("uppsögn", "starfsmaður", "vinnuveitandi")),
    QueryCase("Hver er löglegur uppsagnarfrestur?",
              category="employment",
              expected_keywords=("uppsagnarfrestur", "mánuð")),
    QueryCase("Hverjar eru skyldur vinnuveitanda gagnvart starfsmanni?",
              category="employment",
              expected_keywords=("vinnuveitandi", "skyldu", "starfsmaður")),
    QueryCase("Hvað er orlofsréttur launþega?",
              category="employment",
              expected_keywords=("orlof", "launþeg")),
    # ── Civil / contract ──────────────────────────────────────────────────────
    QueryCase("Hvenær fyrnist krafa?",
              category="civil",
              expected_keywords=("fyrning", "krafa")),
    QueryCase("Getur leigjandi rifið leigusamning?",
              category="civil",
              expected_keywords=("leigusamning", "leigjandi")),
    QueryCase("Hvernig er hjúskapur stofnaður?",
              category="civil",
              expected_keywords=("hjúskap", "gifting")),
    # ── Criminal ──────────────────────────────────────────────────────────────
    QueryCase("Hvað er þjófnaður samkvæmt hegningarlögum?",
              category="criminal",
              expected_keywords=("þjófnaður", "refsi", "hegning")),
    QueryCase("Hvenær telst verknaður af ásetningi freminn?",
              category="criminal",
              expected_keywords=("ásetning", "refsi")),
    # ── Corporate / commercial ────────────────────────────────────────────────
    QueryCase("Hvernig er hlutafélag stofnað?",
              category="corporate",
              expected_keywords=("hlutafélag", "stofn")),
    QueryCase("Hverjar eru skyldur stjórnar hlutafélags?",
              category="corporate",
              expected_keywords=("stjórn", "hlutafélag")),
    # ── Health / licensing ────────────────────────────────────────────────────
    QueryCase("Hverjir mega starfa sem læknar?",
              category="health",
              expected_keywords=("læknir", "starfsréttindi", "leyfi")),
    # ── Dense legal phrasing ──────────────────────────────────────────────────
    QueryCase("Samningur gerður við mann sem er ekki lögráða er ógildur",
              category="legal_phrasing",
              expected_keywords=("lögráða", "samningur", "ógild")),
    QueryCase("Bótaábyrgð vegna tjóns af völdum bifreiðar",
              category="legal_phrasing",
              expected_keywords=("bótaábyrgð", "tjón", "bifreið")),
    QueryCase("Skilyrði fyrir veitingu rekstrarleyfis",
              category="legal_phrasing",
              expected_keywords=("rekstrarleyfi", "skilyrði", "leyfi")),
]


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str = "") -> None:
    """Per-query progress to stderr; keeps stdout clean for the summary."""
    print(msg, file=sys.stderr)


# ── Environment / clients ─────────────────────────────────────────────────────

def load_env() -> tuple[str, str, str]:
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
    required = ["SUPABASE_URL", "SUPABASE_KEY", "OPENAI_API_KEY"]
    missing  = [k for k in required if not os.getenv(k)]
    if missing:
        log("[ERROR] Missing env vars: " + ", ".join(missing))
        sys.exit(1)
    return os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"), os.getenv("OPENAI_API_KEY")


def init_openai(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def init_supabase(url: str, key: str):
    return create_client(url, key)


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed(client: OpenAI, text: str) -> list[float]:
    try:
        embedding = client.embeddings.create(
            model=EMBEDDING_MODEL, input=text
        ).data[0].embedding
    except OpenAIError as e:
        log(f"[ERROR] Embedding failed: {e}")
        sys.exit(1)
    if len(embedding) != EMBEDDING_DIM:
        log(f"[ERROR] Unexpected embedding dimension: {len(embedding)} (expected {EMBEDDING_DIM}).")
        sys.exit(1)
    return embedding


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(supabase_client, embedding: list[float], k: int) -> list[dict]:
    try:
        result = supabase_client.rpc(
            "match_documents",
            {"query_embedding": embedding, "match_count": k},
        ).execute()
    except Exception as e:
        log(f"[ERROR] match_documents RPC failed: {e}")
        log("Ensure match_documents exists in your Supabase project.")
        sys.exit(1)

    hits = result.data or []

    for i, hit in enumerate(hits):
        sim = hit.get("similarity")
        if sim is None or not isinstance(sim, (int, float)):
            log(f"[ERROR] Row {i} from match_documents missing numeric similarity field.")
            log(f"        Row: {hit}")
            sys.exit(1)
        if not (-1e-6 <= float(sim) <= 1.0 + 1e-6):
            log(f"[ERROR] Row {i} similarity {sim:.6f} is outside [0, 1]. RPC may be misconfigured.")
            sys.exit(1)

    return hits


# ── Keyword heuristic ─────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """
    Lowercase and split into word tokens.  Preserves Icelandic characters
    (þ ð æ á é í ó ú ý ö etc.) which are semantically load-bearing in legal text.
    """
    return re.findall(r"[^\s\W_]+", text.lower(), re.UNICODE)


def _keyword_hit(keywords: tuple[str, ...], text: str) -> bool:
    """
    Token-prefix matching: a keyword hits if any token in `text` starts with it
    (case-insensitive).  Handles Icelandic inflection: "launþeg" matches
    "launþega", "launþegar", "launþegum", etc.

    Limitation: does not handle ablaut/umlaut stem changes.  A proper Icelandic
    stemmer (e.g. Snowball/IS) would improve recall for irregular forms.
    """
    if not keywords:
        return False
    tokens = _tokenize(text)
    for kw in keywords:
        kw_lower = kw.lower()
        if any(tok.startswith(kw_lower) for tok in tokens):
            return True
    return False


# ── Statistics ────────────────────────────────────────────────────────────────

def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sv  = sorted(values)
    mid = len(sv) // 2
    return sv[mid] if len(sv) % 2 else (sv[mid - 1] + sv[mid]) / 2


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (p in [0, 100])."""
    sv  = sorted(values)
    idx = (len(sv) - 1) * p / 100
    lo  = int(idx)
    hi  = lo + 1
    if hi >= len(sv):
        return sv[lo]
    return sv[lo] + (idx - lo) * (sv[hi] - sv[lo])


# ── Metrics ───────────────────────────────────────────────────────────────────

class QueryResult(NamedTuple):
    case:        QueryCase
    hits:        list[dict]
    max_sim:     float
    avg_sim:     float
    delta:       float
    above_75:    int
    above_60:    int
    above_50:    int
    top_law_ref: str
    keyword_hit: bool
    ref_hit:     bool
    verdict:     str   # "likely_good" | "borderline" | "weak"


def compute_metrics(
    case:               QueryCase,
    hits:               list[dict],
    threshold_high:     float,
    threshold_moderate: float,
    threshold_mid:      float,
) -> QueryResult:
    """
    Compute retrieval quality metrics and classify verdict for one query.

    Hits are sorted defensively by similarity DESC -never rely on RPC ordering.

    Verdict rules (first match wins):
      Rule A: expected_law_reference in top 3       ->likely_good
      Rule B: max_sim >= threshold_high             ->likely_good
      Rule C: max_sim >= threshold_moderate
              AND keyword hit in top-1 text         ->likely_good
      Rule D: max_sim >= threshold_mid              ->borderline
      Else                                          ->weak

    threshold_moderate is the explicit midpoint between high and mid (or CLI override).
    No hidden multipliers.
    """
    if not hits:
        return QueryResult(
            case=case, hits=[], max_sim=0.0, avg_sim=0.0, delta=0.0,
            above_75=0, above_60=0, above_50=0, top_law_ref="",
            keyword_hit=False, ref_hit=False, verdict="weak",
        )

    # Defensive sort -do not assume RPC returns rows ordered by similarity
    hits = sorted(hits, key=lambda h: float(h["similarity"]), reverse=True)

    sims     = [float(h["similarity"]) for h in hits]
    max_sim  = sims[0]
    avg_sim  = sum(sims) / len(sims)
    delta    = sims[0] - (sims[1] if len(sims) >= 2 else 0.0)
    above_75 = sum(1 for s in sims if s >= 0.75)
    above_60 = sum(1 for s in sims if s >= 0.60)
    above_50 = sum(1 for s in sims if s >= 0.50)

    top_law_ref = str(hits[0].get("law_reference") or "")
    top_text    = str(hits[0].get("text") or "")

    keyword_hit = _keyword_hit(case.expected_keywords, top_text)
    ref_hit     = bool(
        case.expected_law_reference and
        any(h.get("law_reference") == case.expected_law_reference for h in hits[:3])
    )

    if ref_hit:
        verdict = "likely_good"
    elif max_sim >= threshold_high:
        verdict = "likely_good"
    elif max_sim >= threshold_moderate and keyword_hit:
        verdict = "likely_good"
    elif max_sim >= threshold_mid:
        verdict = "borderline"
    else:
        verdict = "weak"

    return QueryResult(
        case=case, hits=hits,
        max_sim=max_sim, avg_sim=avg_sim, delta=delta,
        above_75=above_75, above_60=above_60, above_50=above_50,
        top_law_ref=top_law_ref, keyword_hit=keyword_hit,
        ref_hit=ref_hit, verdict=verdict,
    )


# ── Output: per-query table ───────────────────────────────────────────────────

CW_Q = 38
CW_C = 14
CW_N =  7
CW_R = 13


def print_table(results: list[QueryResult]) -> None:
    if not results:
        print("No results to display.")
        return
    header = (
        f"{'Query':<{CW_Q}}"
        f"{'category':<{CW_C}}"
        f"{'max':>{CW_N}}"
        f"{'avg':>{CW_N}}"
        f"{'delta':>{CW_N}}"
        f"  {'>=75':>4}"
        f"  {'>=60':>4}"
        f"  {'>=50':>4}"
        f"  {'kw':>2}"
        f"  {'ref':>3}"
        f"  {'top_law_ref':<{CW_R}}"
        f"  verdict"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r.case.query_text[:CW_Q - 1]:<{CW_Q}}"
            f"{r.case.category[:CW_C - 1]:<{CW_C}}"
            f"{r.max_sim:>{CW_N}.4f}"
            f"{r.avg_sim:>{CW_N}.4f}"
            f"{r.delta:>{CW_N}.4f}"
            f"  {r.above_75:>4}"
            f"  {r.above_60:>4}"
            f"  {r.above_50:>4}"
            f"  {'Y' if r.keyword_hit else 'N':>2}"
            f"  {'Y' if r.ref_hit else 'N':>3}"
            f"  {r.top_law_ref:<{CW_R}}"
            f"  {r.verdict}"
        )
    print(sep)


# ── Output: category breakdown ────────────────────────────────────────────────

def print_category_breakdown(results: list[QueryResult]) -> None:
    if not results:
        return
    cats: dict[str, list[QueryResult]] = {}
    for r in results:
        cats.setdefault(r.case.category, []).append(r)

    print()
    print("CATEGORY BREAKDOWN")
    hdr = f"  {'category':<16}  {'N':>3}  {'avg max':>8}  {'med max':>8}  {'>=60':>5}  {'kw%':>5}  verdict dist"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for cat in sorted(cats):
        cr  = cats[cat]
        n   = len(cr)
        avg = sum(r.max_sim for r in cr) / n
        med = _median([r.max_sim for r in cr])
        p60 = 100 * sum(1 for r in cr if r.above_60 >= 1) / n
        kw  = 100 * sum(1 for r in cr if r.keyword_hit) / n
        dist = (
            f"G={sum(1 for r in cr if r.verdict == 'likely_good')} "
            f"B={sum(1 for r in cr if r.verdict == 'borderline')} "
            f"W={sum(1 for r in cr if r.verdict == 'weak')}"
        )
        print(f"  {cat:<16}  {n:>3}  {avg:>8.4f}  {med:>8.4f}  {p60:>5.0f}%  {kw:>4.0f}%  {dist}")


# ── Output: aggregate stats ───────────────────────────────────────────────────

def print_aggregate(
    results:            list[QueryResult],
    threshold_high:     float,
    threshold_moderate: float,
    threshold_mid:      float,
    debug:              bool = False,
) -> None:
    if not results:
        print("No queries evaluated.")
        return

    n        = len(results)
    max_sims = [r.max_sim for r in results]
    deltas   = [r.delta   for r in results]

    avg_max        = sum(max_sims) / n
    med_max        = _median(max_sims)
    std_max        = _stddev(max_sims)
    avg_delta      = sum(deltas) / n
    med_delta      = _median(deltas)
    flatness_ratio = avg_delta / avg_max if avg_max > 0 else 0.0
    pct_60         = 100 * sum(1 for r in results if r.above_60 >= 1) / n
    pct_50         = 100 * sum(1 for r in results if r.above_50 >= 1) / n

    kw_queries  = [r for r in results if r.case.expected_keywords]
    ref_queries = [r for r in results if r.case.expected_law_reference is not None]
    top1_kw  = (100 * sum(1 for r in kw_queries  if r.keyword_hit) / len(kw_queries))  if kw_queries  else None
    top1_ref = (100 * sum(1 for r in ref_queries if r.ref_hit)     / len(ref_queries)) if ref_queries else None

    likely     = sum(1 for r in results if r.verdict == "likely_good")
    borderline = sum(1 for r in results if r.verdict == "borderline")
    weak       = sum(1 for r in results if r.verdict == "weak")

    print()
    print("AGGREGATE STATS")
    print(f"  Queries evaluated           : {n}")
    print(f"  Avg  max similarity         : {avg_max:.4f}")
    print(f"  Med  max similarity         : {med_max:.4f}")
    std_note = (
        "(low  ->uniform distribution across queries)"  if std_max < 0.04 else
        "(mod  ->some domain variance)"                 if std_max < 0.08 else
        "(high ->strong domain sensitivity)"
    )
    print(f"  Std  max similarity         : {std_max:.4f}  {std_note}")
    print(f"  Avg  delta (max-2nd)        : {avg_delta:.4f}")
    print(f"  Med  delta                  : {med_delta:.4f}")
    flat_note = (
        "(very flat -poor top-1 differentiation)" if flatness_ratio < 0.03 else
        "(moderate top-1 separation)"              if flatness_ratio < 0.08 else
        "(clear top-1 signal)"
    )
    print(f"  Flatness ratio (d/avg_max)  : {flatness_ratio:.4f}  {flat_note}")
    print(f"  Queries with hit >=0.60     : {pct_60:.0f}%")
    print(f"  Queries with hit >=0.50     : {pct_50:.0f}%")
    print(f"  Top-1 keyword hit rate      : {f'{top1_kw:.0f}%' if top1_kw is not None else 'n/a'}"
          f"  ({len(kw_queries)} queries with keywords)")
    print(f"  Top-1 ref hit rate          : {f'{top1_ref:.0f}%' if top1_ref is not None else 'n/a (no expected refs set)'}")
    print(f"  Verdict -likely_good       : {likely}/{n}")
    print(f"  Verdict -borderline        : {borderline}/{n}")
    print(f"  Verdict -weak              : {weak}/{n}")

    if debug:
        # Histogram buckets derived from configured thresholds
        bucket_lo = threshold_mid - 0.10
        labels = [
            f">={threshold_high:.2f}",
            f"{threshold_moderate:.2f}-{threshold_high:.2f}",
            f"{threshold_mid:.2f}-{threshold_moderate:.2f}",
            f"{bucket_lo:.2f}-{threshold_mid:.2f}",
            f"<{bucket_lo:.2f}",
        ]
        counts = [
            sum(1 for s in max_sims if s >= threshold_high),
            sum(1 for s in max_sims if threshold_moderate <= s < threshold_high),
            sum(1 for s in max_sims if threshold_mid <= s < threshold_moderate),
            sum(1 for s in max_sims if bucket_lo <= s < threshold_mid),
            sum(1 for s in max_sims if s < bucket_lo),
        ]
        print()
        print("  Max-similarity histogram (threshold-aligned):")
        for label, count in zip(labels, counts):
            bar = "#" * count
            print(f"    {label:<14}  {bar:<{n}} ({count})")


# ── Output: diagnostic summary ────────────────────────────────────────────────

def print_diagnostic_summary(
    results:            list[QueryResult],
    threshold_high:     float,
    threshold_moderate: float,
    threshold_mid:      float,
) -> None:
    if not results:
        return

    n        = len(results)
    max_sims = [r.max_sim for r in results]
    deltas   = [r.delta   for r in results]

    avg_max        = sum(max_sims) / n
    avg_delta      = sum(deltas) / n
    std_max        = _stddev(max_sims)
    flatness_ratio = avg_delta / avg_max if avg_max > 0 else 0.0
    pct_60         = 100 * sum(1 for r in results if r.above_60 >= 1) / n

    p25 = _percentile(max_sims, 25)
    p50 = _percentile(max_sims, 50)
    p75 = _percentile(max_sims, 75)

    # Recommended threshold: P25 floored to nearest 0.05,
    # clamped to [threshold_mid - 0.05, threshold_high].
    raw_rec     = math.floor(p25 / 0.05) * 0.05
    recommended = max(threshold_mid - 0.05, min(threshold_high, raw_rec))

    print()
    print("RETRIEVAL DIAGNOSTIC SUMMARY")
    print("-" * 54)

    print(f"  Percentiles  P25 / P50 / P75 : {p25:.4f} / {p50:.4f} / {p75:.4f}")
    if p25 < threshold_mid:
        print(f"  WARNING: P25 ({p25:.3f}) < threshold_mid ({threshold_mid:.2f}) -"
              "mid threshold is optimistic for this corpus.")

    print()
    sharpness = (
        "STRONG   -clear top-1 signal (avg delta > 0.10)" if avg_delta > 0.10 else
        "MODERATE -some differentiation between top hits" if avg_delta > 0.04 else
        "FLAT     -densely clustered scores (avg delta <= 0.04); hybrid BM25 likely needed"
    )
    print(f"  Sharpness    : {sharpness}")

    score_lv = (
        f"HIGH   -avg max {avg_max:.3f}; strong embedding signal" if avg_max >= 0.70 else
        f"MEDIUM -avg max {avg_max:.3f}; reasonable signal"        if avg_max >= 0.60 else
        f"LOW    -avg max {avg_max:.3f}; marginal signal"          if avg_max >= 0.50 else
        f"POOR   -avg max {avg_max:.3f}; embedding may not suit corpus style"
    )
    print(f"  Score level  : {score_lv}")

    disp = (
        f"LOW  (std={std_max:.3f}) -uniform quality across query types" if std_max < 0.04 else
        f"MOD  (std={std_max:.3f}) -some domain variance"               if std_max < 0.08 else
        f"HIGH (std={std_max:.3f}) -strong domain sensitivity; consider per-domain thresholds"
    )
    print(f"  Dispersion   : {disp}")

    print()
    print(f"  Recommended min_similarity   : {recommended:.2f}")
    print(f"  Derivation: P25={p25:.3f} ->floor to 0.05 = {raw_rec:.2f}, "
          f"clamped to [{threshold_mid - 0.05:.2f}, {threshold_high:.2f}]")
    if recommended < threshold_high - 0.05:
        print(f"  NOTE: threshold-high ({threshold_high:.2f}) exceeds recommended. "
              f"Consider lowering to {recommended:.2f}.")

    print()
    print("ARCHITECTURAL RECOMMENDATION")
    print("-" * 54)

    if avg_max >= 0.70 and pct_60 >= 70 and flatness_ratio >= 0.05:
        arch = "VECTOR-ONLY RECOMMENDED"
        note = f"Strong scores (avg max {avg_max:.3f}), {pct_60:.0f}% coverage >=0.60, clear top-1 signal."
    elif avg_max < 0.50:
        arch = "RE-CHUNKING LIKELY NEEDED"
        note = (f"Avg max {avg_max:.3f} is very low. Article-level chunks may be too coarse. "
                "Consider sub-article or paragraph-level granularity.")
    elif flatness_ratio < 0.03 or avg_delta < 0.03:
        arch = "HYBRID BM25 + VECTOR REQUIRED"
        note = (f"Flatness ratio {flatness_ratio:.3f} -poor top-1 differentiation. "
                "Lexical BM25 on article text would add the precision vector alone cannot.")
    elif pct_60 < 50:
        arch = "HYBRID BM25 + VECTOR REQUIRED"
        note = (f"Only {pct_60:.0f}% of queries have a hit >=0.60. "
                "Vector retrieval alone cannot reliably serve this query set.")
    elif avg_max >= 0.55 and pct_60 >= 50:
        arch = "VECTOR + LEXICAL RE-RANKING RECOMMENDED"
        note = (f"Moderate scores (avg max {avg_max:.3f}). "
                "BM25 re-ranking of top-k vector results improves precision "
                "without full hybrid pipeline complexity.")
    else:
        arch = "HYBRID BM25 + VECTOR REQUIRED"
        note = "Signal insufficient for reliable vector-only retrieval at any practical threshold."

    print(f"  {arch}")
    print(f"  {note}")


# ── CSV export ────────────────────────────────────────────────────────────────

def write_csv(results: list[QueryResult], path: str) -> None:
    fieldnames = [
        "query_text", "category", "max_sim", "avg_sim", "delta",
        "above_75", "above_60", "above_50",
        "keyword_hit", "ref_hit", "top_law_ref", "verdict",
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "query_text":  r.case.query_text,
                "category":    r.case.category,
                "max_sim":     f"{r.max_sim:.6f}",
                "avg_sim":     f"{r.avg_sim:.6f}",
                "delta":       f"{r.delta:.6f}",
                "above_75":    r.above_75,
                "above_60":    r.above_60,
                "above_50":    r.above_50,
                "keyword_hit": "Y" if r.keyword_hit else "N",
                "ref_hit":     "Y" if r.ref_hit else "N",
                "top_law_ref": r.top_law_ref,
                "verdict":     r.verdict,
            })
    print(f"\nCSV written ->{out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate vector retrieval quality across a fixed Icelandic legal query set.",
    )
    parser.add_argument("--k",                  type=int,   default=8,
                        metavar="N",    help="Chunks to retrieve per query (default: 8)")
    parser.add_argument("--threshold-high",     type=float, default=DEFAULT_THRESHOLD_HIGH,
                        metavar="F",    help=f"High-confidence threshold (default: {DEFAULT_THRESHOLD_HIGH})")
    parser.add_argument("--threshold-moderate", type=float, default=None,
                        metavar="F",    help="Moderate threshold (default: midpoint of high and mid)")
    parser.add_argument("--threshold-mid",      type=float, default=DEFAULT_THRESHOLD_MID,
                        metavar="F",    help=f"Borderline threshold (default: {DEFAULT_THRESHOLD_MID})")
    parser.add_argument("--debug",              action="store_true",
                        help="Print top 3 chunks per query (stderr) and histogram (stdout)")
    parser.add_argument("--csv",                default=None, metavar="FILE",
                        help="Export per-query metrics to CSV file")
    args = parser.parse_args()

    threshold_moderate = (
        args.threshold_moderate
        if args.threshold_moderate is not None
        else args.threshold_mid + (args.threshold_high - args.threshold_mid) * 0.5
    )

    supabase_url, supabase_key, openai_api_key = load_env()
    openai_client   = init_openai(openai_api_key)
    supabase_client = init_supabase(supabase_url, supabase_key)

    results: list[QueryResult] = []

    log(f"Evaluating {len(QUERIES)} queries  "
        f"(k={args.k}  high={args.threshold_high}  "
        f"mod={threshold_moderate:.2f}  mid={args.threshold_mid})")
    log()

    for i, case in enumerate(QUERIES, 1):
        log(f"[{i:02d}/{len(QUERIES)}] {case.query_text}")
        embedding = embed(openai_client, case.query_text)
        hits      = retrieve(supabase_client, embedding, args.k)
        result    = compute_metrics(
            case, hits, args.threshold_high, threshold_moderate, args.threshold_mid
        )
        results.append(result)
        log(f"       max={result.max_sim:.4f}  avg={result.avg_sim:.4f}  "
            f"delta={result.delta:.4f}  kw={'Y' if result.keyword_hit else 'N'}  "
            f"verdict={result.verdict}")

        if args.debug:
            log("       Top 3 hits:")
            for j, hit in enumerate(result.hits[:3], 1):
                sim     = float(hit["similarity"])
                snippet = str(hit.get("text") or "")[:120].replace("\n", " ")
                log(f"         [{j}] {sim:.4f}  {str(hit.get('law_reference', '')):<12}  {snippet}")
            log()

    log()
    print_table(results)
    print_category_breakdown(results)
    print_aggregate(results, args.threshold_high, threshold_moderate, args.threshold_mid, debug=args.debug)
    print_diagnostic_summary(results, args.threshold_high, threshold_moderate, args.threshold_mid)

    if args.csv:
        write_csv(results, args.csv)


if __name__ == "__main__":
    main()
