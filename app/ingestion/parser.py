"""
SGML Parser for Lagasafn (Icelandic Law Collection) — 156b release

Structure discovered by inspecting real files (not assumed):

  Encoding:  Windows-1252 (cp1252) — open files with that encoding.
  Tags:      BeautifulSoup html.parser lowercases all tag names.

  HEAD (metadata):
    <lyr>  YYYY           year of enactment    (real format)
    <lno>  N              law number, may be empty for ancient laws (real)
    <ldt>  text           publication date text (real)
    <title> text          law title             (real)
    <ar>   YYYY           year                  (fixture / alt format)
    <heiti> text          title                 (fixture / alt format)

  BODY — three structural patterns:

  Pattern A  Standard modern law (post-~1900):
    <body>
      <chapter>?                    optional grouping
        <chnm1><chka>I.</chka>      chapter name
        <gr>                        article (grein)
          <gn>1. gr.</gn>           article header
          <mgr>paragraph text</mgr> one or more paragraphs
            <fnn>1)</fnn>           inline footnote ref  ← STRIP
          <fnpart>…</fnpart>        footnote block       ← STRIP

  Pattern B  Medieval compilations (Jónsbók 1281, etc.):
    <body>
      <part>?
        <chapter>
          <chnm1><chka>chapter name</chka>
          <p>text…</p>              no <gr> or <mgr>, prose in <p>

  Pattern C  Very short / very old:
    Just a <p> in <body>, no article structure.

  Additionally, test fixtures use:
    <log> root, <nr> law-number, <ar> year, <heiti> title,
    <grein> article, <nr> article-number, <mgr><nr> para-number, text.
"""

import copy
import json
import re
import warnings
from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from app.services.canonicalize import canonicalize
from app.models.schemas import ParsedArticle, ParsedLaw


SCHEMA_VERSION = "1"

# Tags whose content is excluded from article/paragraph text.
# - fnpart / fnn : footnote definitions and inline reference markers
# - nr           : paragraph-number annotations in fixture format (<mgr><nr>1</nr>text)
#                  Safe to exclude from body text because real SGML never uses <nr>
#                  inside <mgr> or article prose.
_EXCLUDE_TAGS = frozenset({"fnpart", "fnn", "nr"})


# ── Text helpers ──────────────────────────────────────────────────────────────

def _clean_text(tag) -> str:
    """
    Extract clean text from a BS4 tag without mutating the original tree.
    Strips excluded markup, then normalises whitespace.
    """
    t = copy.deepcopy(tag)
    for name in _EXCLUDE_TAGS:
        for elem in t.find_all(name):
            elem.decompose()
    raw = t.get_text(separator=" ")
    return re.sub(r"\s+", " ", raw).strip()


def _tag_text(parent, *names: str) -> str:
    """Return the text of the first matching tag found in *parent*, or ''."""
    for name in names:
        found = parent.find(name)
        if found:
            return found.get_text(strip=True)
    return ""


# ── Metadata extraction ───────────────────────────────────────────────────────

def _extract_metadata(
    soup: BeautifulSoup, source_file: str
) -> tuple[str, str, str, str, List[str]]:
    """
    Return (law_number, law_year, title, publication_date, warnings).

    Tries the real 156b format first (lno / lyr / title / ldt),
    then the fixture / alt format (nr / ar / heiti).
    Falls back to filename-derived values when tags are absent or empty.
    """
    warn: List[str] = []
    head = soup.find("head")
    search_root = head if head else soup

    # Year: <lyr> (real) or <ar> (fixture)
    law_year = _tag_text(search_root, "lyr", "ar")

    # Number: <lno> (real) or <nr> that is NOT inside an article tag (fixture)
    lno_tag = search_root.find("lno")
    if lno_tag is not None:
        law_number = lno_tag.get_text(strip=True)
    else:
        # Fixture format: the law-number <nr> is a near-root child, not inside
        # <gr> / <grein> / <mgr>.  BeautifulSoup.find() returns the first
        # occurrence in document order; for well-formed fixtures that is the
        # law-number element.
        nr_tag = soup.find("nr")
        if nr_tag and nr_tag.parent and nr_tag.parent.name not in (
            "gr", "grein", "mgr", "lfrv", "saga"
        ):
            law_number = nr_tag.get_text(strip=True)
        else:
            law_number = ""

    # Title: <title> (real) or <heiti> (fixture)
    title = _tag_text(search_root, "title", "heiti")

    # Publication date: <ldt> (real only)
    publication_date = _tag_text(search_root, "ldt")

    # Fill gaps from filename: first 4 chars = year, rest = number
    stem = Path(source_file).name.split(".")[0]  # "1944033" from "1944033.sgml"
    if not law_year and len(stem) >= 4:
        law_year = stem[:4]
        warn.append(f"year inferred from filename: {law_year}")
    if not law_number and len(stem) > 4:
        n = stem[4:].lstrip("0")
        law_number = n if n else ""
        if law_number:
            warn.append(f"law_number inferred from filename: {law_number}")

    return law_number, law_year, title, publication_date, warn


def _build_law_ref(
    law_number: str, law_year: str, title: str, warnings_list: List[str]
) -> str:
    """Build the canonical law-reference string (e.g. '33/1944')."""
    if law_number and law_number not in ("0", ""):
        return f"{law_number}/{law_year}"
    # Ancient law without a formal number
    safe = re.sub(r"\s+", "_", title)[:40]
    warnings_list.append("no formal law number — using title-based reference")
    return f"[{safe}]/{law_year}"


# ── Locator construction ──────────────────────────────────────────────────────

def build_locator(
    law_number: str,
    law_year: str,
    article_number: Optional[str] = None,
    paragraph_number: Optional[str] = None,
) -> str:
    """
    Build a locator string from explicit number + year components.

    Format:  "Lög nr. {N}/{YYYY}"
             "Lög nr. {N}/{YYYY} - {art}. gr."
             "Lög nr. {N}/{YYYY} - {art}. gr., {para}. mgr."

    Kept for backward compatibility with the ingestion pipeline.
    """
    base = f"Lög nr. {law_number}/{law_year}"
    if article_number:
        base += f" - {article_number}. gr."
        if paragraph_number:
            base += f", {paragraph_number}. mgr."
    return base


def _loc(
    law_ref: str,
    art_num: Optional[str] = None,
    para_num: Optional[str] = None,
) -> str:
    """Build locator from an already-computed law_reference string."""
    base = f"Lög nr. {law_ref}"
    if art_num:
        base += f" - {art_num}. gr."
        if para_num:
            base += f", {para_num}. mgr."
    return base


# ── Chapter helpers ───────────────────────────────────────────────────────────

def _chapter_label(chapter_tag) -> Optional[str]:
    """Return the chapter name from a <chapter> tag, or None."""
    chnm = chapter_tag.find("chnm1")
    if not chnm:
        return None
    chka = chnm.find("chka")
    label = (chka or chnm).get_text(separator=" ", strip=True)
    return label or None


def _chapter_ancestor(tag) -> Optional[str]:
    """Walk up the tree and return the nearest chapter name, or None."""
    for anc in tag.parents:
        if anc.name == "chapter":
            return _chapter_label(anc)
        if anc.name in ("body", "law", "log"):
            break
    return None


# ── Article number extraction ─────────────────────────────────────────────────

def _article_number(article_tag) -> Optional[str]:
    """
    Extract the article number from a <gr> or <grein> tag.

    Tries <gn> first (real format: "1. gr." → "1"),
    then <nr> as a direct child (fixture format: <nr>1</nr>).
    Returns None if no number can be found.
    """
    # Real format: <gn>1. gr.</gn> or amended <gn>[30. gr.]</gn>
    gn = article_tag.find("gn")
    if gn:
        m = re.match(r"^\[*(\d+)", gn.get_text(strip=True))
        if m:
            return m.group(1)

    # Fixture format: <nr>N</nr> as a direct child of the article tag
    for child in article_tag.children:
        if getattr(child, "name", None) == "nr":
            m = re.match(r"^(\d+)", child.get_text(strip=True))
            if m:
                return m.group(1)

    return None


# ── Paragraph extraction ──────────────────────────────────────────────────────

def _paragraphs(article_tag, law_ref: str, art_num: str) -> List[dict]:
    """
    Extract paragraphs from an article tag.

    Precedence:
      1. <mgr> tags  (real format and fixture format)
      2. <p> tags    (medieval format, e.g. Jónsbók)
      3. Full article text minus the article-number tag (fallback)
    """
    mgr_tags = article_tag.find_all("mgr")
    if mgr_tags:
        result = []
        para_num = 0
        for mgr in mgr_tags:
            t = _clean_text(mgr)
            if t:
                para_num += 1
                result.append(
                    {
                        "number": str(para_num),
                        "text": t,
                        "locator": _loc(law_ref, art_num, str(para_num)),
                    }
                )
        return result

    p_tags = article_tag.find_all("p")
    if p_tags:
        # For medieval laws, include any article sub-heading (<h2>) as a
        # prefix to the first paragraph so it is searchable.
        heading_parts = [
            _clean_text(h)
            for h in article_tag.find_all(["h2", "h3", "h4", "h5", "h6"])
            if _clean_text(h)
        ]
        heading = " ".join(heading_parts)

        result = []
        para_num = 0
        for i, p in enumerate(p_tags):
            t = _clean_text(p)
            if t:
                para_num += 1
                if i == 0 and heading:
                    t = f"{heading} {t}"
                result.append(
                    {
                        "number": str(para_num),
                        "text": t,
                        "locator": _loc(law_ref, art_num, str(para_num)),
                    }
                )
        return result

    # Fallback: strip the article-number tag, then extract remaining text.
    art_copy = copy.deepcopy(article_tag)
    for tag_name in ("gn", "nr"):
        for elem in art_copy.find_all(tag_name):
            elem.decompose()
            break  # Remove only the first occurrence (the number tag)
    t = _clean_text(art_copy)
    if t:
        return [
            {"number": "1", "text": t, "locator": _loc(law_ref, art_num, "1")}
        ]
    return []


# ── Core parse function ───────────────────────────────────────────────────────

class SGMLParseError(Exception):
    """Raised when SGML parsing fails critically (strict mode)."""
    pass


def parse_sgml(content: str, source_file: str) -> dict:
    """
    Parse an SGML string (already decoded to str) into the internal law schema.

    Args:
        content:     File content decoded to str.
                     Use cp1252 encoding when reading real Lagasafn files.
        source_file: Filename used for warnings and the schema's source_file field.

    Returns:
        dict conforming to the law JSON schema (schema_version, articles, …).

    Raises:
        ValueError: If no articles can be extracted.
    """
    parse_warnings: List[str] = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(content, "html.parser")

    # Metadata
    law_number, law_year, title, publication_date, meta_w = _extract_metadata(
        soup, source_file
    )
    parse_warnings.extend(meta_w)
    law_ref = _build_law_ref(law_number, law_year, title, parse_warnings)

    # Body search root
    body = soup.find("body") or soup

    # ── Article extraction: try three structural patterns ─────────────────────
    articles: List[dict] = []

    # Pattern A & fixture: <gr> or <grein> article tags
    article_tags = body.find_all(["gr", "grein"])
    if article_tags:
        for tag in article_tags:
            art_num = _article_number(tag)
            if art_num is None:
                parse_warnings.append("<gr>/<grein> with no numeric id — skipped")
                continue

            chapter = _chapter_ancestor(tag)
            paras = _paragraphs(tag, law_ref, art_num)
            if not paras:
                parse_warnings.append(f"article {art_num} has no text — skipped")
                continue

            articles.append(
                {
                    "number": art_num,
                    "locator": _loc(law_ref, art_num),
                    "chapter": chapter,
                    "paragraphs": paras,
                }
            )

    else:
        # Pattern B: no article tags — use <chapter> blocks as article units
        chapter_tags = body.find_all("chapter")
        if chapter_tags:
            for i, chap in enumerate(chapter_tags, 1):
                chap_label = _chapter_label(chap)
                chap_copy = copy.deepcopy(chap)
                for chnm in chap_copy.find_all("chnm1"):
                    chnm.decompose()
                t = _clean_text(chap_copy)
                if t:
                    art_num = str(i)
                    articles.append(
                        {
                            "number": art_num,
                            "locator": _loc(law_ref, art_num),
                            "chapter": chap_label,
                            "paragraphs": [
                                {
                                    "number": "1",
                                    "text": t,
                                    "locator": _loc(law_ref, art_num, "1"),
                                }
                            ],
                        }
                    )
            parse_warnings.append(
                f"no <gr>/<grein> tags — used {len(articles)} <chapter> blocks as articles"
            )

        else:
            # Pattern C: no structure at all — entire body as single article
            t = _clean_text(body)
            if t:
                articles.append(
                    {
                        "number": "1",
                        "locator": _loc(law_ref, "1"),
                        "chapter": None,
                        "paragraphs": [
                            {
                                "number": "1",
                                "text": t,
                                "locator": _loc(law_ref, "1", "1"),
                            }
                        ],
                    }
                )
            parse_warnings.append(
                "no <gr>/<grein> or <chapter> tags — entire body as single article"
            )

    if not articles:
        raise ValueError(f"No articles extracted from {source_file!r}")

    return {
        "schema_version": SCHEMA_VERSION,
        "source_file": source_file,
        "law_number": law_number,
        "law_year": law_year,
        "law_reference": law_ref,
        "title": title,
        "publication_date": publication_date,
        "articles": articles,
        "parse_warnings": parse_warnings,
        "article_count": len(articles),
    }


def parse_file(path: Path) -> dict:
    """
    Read one SGML file with cp1252 encoding and return the parsed schema dict.
    """
    content = path.read_text(encoding="cp1252", errors="replace")
    return parse_sgml(content, source_file=path.name)


# ── Backward-compatible SGMLParser class ─────────────────────────────────────

class SGMLParser:
    """
    Backward-compatible wrapper around parse_sgml().
    Returns ParsedLaw / ParsedArticle objects for the ingestion pipeline.

    The pipeline (app/ingestion/pipeline.py) uses this class; changing its
    public API would break that code.
    """

    def __init__(self, strict: bool = False):
        self.strict = strict
        self.warnings: List[str] = []

    def parse(self, sgml_content: str, source_info: str = "unknown") -> ParsedLaw:
        """
        Parse SGML content and return a ParsedLaw object.

        Args:
            sgml_content: Decoded SGML string.
            source_info:  Identifier used in warnings (e.g. filename).

        Raises:
            SGMLParseError: In strict mode if no articles can be extracted.
        """
        self.warnings = []

        try:
            result = parse_sgml(sgml_content, source_file=source_info)
        except ValueError as exc:
            if self.strict:
                raise SGMLParseError(str(exc)) from exc
            self.warnings.append(str(exc))
            return ParsedLaw(
                law_number="0",
                law_year="0000",
                title="(parse failed)",
                articles=[],
                full_text="",
                metadata={
                    "source_info": source_info,
                    "parser_warnings": [str(exc)],
                },
            )

        self.warnings = result["parse_warnings"]

        # Convert to ParsedArticle objects with canonicalized text
        parsed_articles = []
        for a in result["articles"]:
            para_texts = [p["text"] for p in a["paragraphs"]]
            article_text = canonicalize(" ".join(para_texts))
            parsed_paragraphs = [
                {"number": p["number"], "text": canonicalize(p["text"])}
                for p in a["paragraphs"]
            ]
            parsed_articles.append(
                ParsedArticle(
                    number=a["number"],
                    text=article_text,
                    paragraphs=parsed_paragraphs,
                )
            )

        # Build full document text (title + articles)
        parts = [result["title"]]
        for art in parsed_articles:
            parts.append(f"{art.number}. gr.")
            parts.append(art.text)
        full_text = canonicalize("\n".join(parts))

        return ParsedLaw(
            law_number=result["law_number"],
            law_year=result["law_year"],
            title=canonicalize(result["title"]),
            articles=parsed_articles,
            full_text=full_text,
            metadata={
                "source_info": source_info,
                "parser_warnings": self.warnings,
                "publication_date": result.get("publication_date", ""),
            },
        )
