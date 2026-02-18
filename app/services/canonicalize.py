"""
Canonical Text Normalization Module

This is THE SINGLE SOURCE OF TRUTH for text normalization in the Legal Search MVP.
This function MUST be used at ALL of these points:
  - Ingestion: raw SGML/text -> stored document/chunk text
  - Storage: texts written to DB are canonicalized
  - Retrieval: texts read from DB are canonicalized before use
  - LLM context: every chunk passed into the LLM is canonicalized
  - Validation: both LLM quote and stored chunk are canonicalized before exact-match

DO NOT duplicate this logic. Import and use this module everywhere.
"""

import re
import unicodedata


def canonicalize(text: str) -> str:
    """
    Canonical text normalization for the Legal Search MVP.

    This function applies a deterministic, reversible-ish normalization that:
    a) Applies Unicode NFC normalization (canonical composition)
    b) Replaces NBSP (U+00A0) and other Unicode spaces with regular space
    c) Collapses all whitespace (\\s+) into a single space
    d) Trims leading/trailing whitespace
    e) Does NOT change case or punctuation

    Icelandic characters (þ, ð, æ, ö, Þ, Ð, Æ, Ö) are preserved exactly.

    Args:
        text: Input text to normalize

    Returns:
        Canonicalized text

    Examples:
        >>> canonicalize("  Þórður  Jónsson  ")
        'Þórður Jónsson'
        >>> canonicalize("1.\\xa0gr.")  # NBSP
        '1. gr.'
        >>> canonicalize("Lög\\n\\nnr.\\t33/1944")
        'Lög nr. 33/1944'
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # Step 1: Unicode NFC normalization (canonical composition)
    # This ensures that characters like é are represented consistently
    # (single codepoint vs base + combining accent)
    text = unicodedata.normalize('NFC', text)

    # Step 2: Replace all Unicode space characters with regular ASCII space
    # This includes:
    # - U+00A0 NO-BREAK SPACE (NBSP)
    # - U+2000-U+200B various width spaces
    # - U+202F NARROW NO-BREAK SPACE
    # - U+205F MEDIUM MATHEMATICAL SPACE
    # - U+3000 IDEOGRAPHIC SPACE
    # - U+FEFF ZERO WIDTH NO-BREAK SPACE (BOM)
    unicode_spaces = (
        '\u00A0'   # NBSP
        '\u1680'   # OGHAM SPACE MARK
        '\u2000'   # EN QUAD
        '\u2001'   # EM QUAD
        '\u2002'   # EN SPACE
        '\u2003'   # EM SPACE
        '\u2004'   # THREE-PER-EM SPACE
        '\u2005'   # FOUR-PER-EM SPACE
        '\u2006'   # SIX-PER-EM SPACE
        '\u2007'   # FIGURE SPACE
        '\u2008'   # PUNCTUATION SPACE
        '\u2009'   # THIN SPACE
        '\u200A'   # HAIR SPACE
        '\u200B'   # ZERO WIDTH SPACE
        '\u202F'   # NARROW NO-BREAK SPACE
        '\u205F'   # MEDIUM MATHEMATICAL SPACE
        '\u3000'   # IDEOGRAPHIC SPACE
        '\uFEFF'   # ZERO WIDTH NO-BREAK SPACE (BOM)
    )
    for space_char in unicode_spaces:
        text = text.replace(space_char, ' ')

    # Step 3: Collapse all whitespace (space, tab, newline, etc.) into single space
    # \s matches [ \t\n\r\f\v] in ASCII mode
    text = re.sub(r'\s+', ' ', text)

    # Step 4: Strip leading and trailing whitespace
    text = text.strip()

    return text


def canonicalize_for_search(text: str) -> str:
    """
    Canonicalize text for search indexing/querying.

    Same as canonicalize() - provided for semantic clarity in search contexts.
    This ensures the same normalization is used for both indexed text and queries.
    """
    return canonicalize(text)


def canonicalize_for_validation(text: str) -> str:
    """
    Canonicalize text for quote validation.

    Same as canonicalize() - provided for semantic clarity in validation contexts.
    Both the LLM-generated quote and the source chunk text should be passed
    through this function before exact-match comparison.
    """
    return canonicalize(text)


# Convenience function to check if two texts are equivalent after canonicalization
def texts_match(text1: str, text2: str) -> bool:
    """
    Check if two texts are equivalent after canonicalization.

    Args:
        text1: First text
        text2: Second text

    Returns:
        True if canonicalized versions are identical
    """
    return canonicalize(text1) == canonicalize(text2)


def quote_exists_in_source(quote: str, source: str) -> bool:
    """
    Check if a quote exists verbatim in source text (after canonicalization).

    This is the core validation function for citation verification.

    Args:
        quote: The quote to find (e.g., from LLM response)
        source: The source text to search in (e.g., chunk text)

    Returns:
        True if canonicalized quote is found in canonicalized source
    """
    canonical_quote = canonicalize(quote)
    canonical_source = canonicalize(source)

    if not canonical_quote:
        return False

    return canonical_quote in canonical_source
