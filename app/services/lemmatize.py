"""
lemmatize.py — Icelandic lemmatization using the Greynir (reynir) engine.

get_lemmatized_text(text) converts inflected Icelandic text to base forms
using the BÍN (Beygingarlýsing íslensks nútímamáls) morphological database.

Examples
--------
    "starfsmanni"   -> "starfsmaður"
    "hegningarlögum" -> "hegningarlög"
    "fyrnist"       -> "fyrna"
    "leigusamning"  -> "leigusamningur"
    "stofnaður"     -> "stofna"

This is used to normalise both query text and document text so that
FTS matching works across inflected forms.

Requires: reynir==3.6.0  (Python 3.11, pre-built wheel)
    pip install reynir==3.6.0 --only-binary :all:

NOTE: Reynir.tokenize() is used instead of Reynir.parse_single() because:
  - tokenize() performs BÍN lookup (lemmatisation) without triggering the
    full syntactic parser, so no Greynir.grammar.bin file is required.
  - It is significantly faster and sufficient for lemmatisation tasks.
"""

from __future__ import annotations

from reynir import Reynir, TOK

# Module-level singleton — Reynir initialisation is expensive; reuse across calls.
_reynir: Reynir | None = None


def _get_reynir() -> Reynir:
    global _reynir
    if _reynir is None:
        _reynir = Reynir()
    return _reynir


def get_lemmatized_text(text: str) -> str:
    """
    Return a space-separated string of lemmas (base forms) for every word
    token in *text*.  Non-word tokens (punctuation, numbers, etc.) are
    dropped.  Words not found in BÍN are kept in lowercase as-is.

    Parameters
    ----------
    text : str
        Raw Icelandic text — a query string or a document paragraph.

    Returns
    -------
    str
        Space-joined lemmas, suitable for building a PostgreSQL tsquery or
        tsvector entry with the 'simple' dictionary.

    Examples
    --------
    >>> get_lemmatized_text("Hvenær fyrnist krafa?")
    'hvenær fyrna krafa'

    >>> get_lemmatized_text("starfsmanni samkvæmt hegningarlögum")
    'starfsmaður samkvæmt hegningarlög'
    """
    if not text or not text.strip():
        return ""

    r = _get_reynir()
    lemmas: list[str] = []

    for tok in r.tokenize(text):
        if tok.kind != TOK.WORD:
            # Skip punctuation, sentence boundaries, numbers, etc.
            continue
        if tok.val:
            # tok.val is a list of BÍN meanings, each a tuple:
            # (lemma, bin_id, word_class, category, word_form, grammar_tag)
            # Take the lemma of the first (most common) meaning.
            lemmas.append(tok.val[0][0])
        else:
            # Word not found in BÍN (proper noun, foreign word, etc.) —
            # keep the lowercased original.
            lemmas.append(tok.txt.lower())

    return " ".join(lemmas)
