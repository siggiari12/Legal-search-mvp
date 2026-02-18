"""
Tests for Canonical Text Normalization

These tests ensure that the canonicalize() function works correctly for:
- Icelandic characters (þ, ð, æ, ö, Þ, Ð, Æ, Ö)
- Non-breaking spaces (NBSP)
- Newlines and tabs
- Mixed whitespace
- Unicode normalization (NFC)
- Edge cases

CRITICAL: If any of these tests fail, quote validation will break.
"""

import unicodedata
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.canonicalize import (
    canonicalize,
    canonicalize_for_search,
    canonicalize_for_validation,
    texts_match,
    quote_exists_in_source,
)


class TestIcelandicCharacters:
    """Tests for Icelandic character preservation."""

    def test_lowercase_thorn(self):
        """þ (lowercase thorn) must be preserved."""
        assert canonicalize("þetta") == "þetta"
        assert canonicalize("Þórður") == "Þórður"

    def test_uppercase_thorn(self):
        """Þ (uppercase thorn) must be preserved."""
        assert canonicalize("ÞÓRÐUR") == "ÞÓRÐUR"

    def test_lowercase_eth(self):
        """ð (lowercase eth) must be preserved."""
        assert canonicalize("maður") == "maður"
        assert canonicalize("góðan dag") == "góðan dag"

    def test_uppercase_eth(self):
        """Ð (uppercase eth) must be preserved."""
        assert canonicalize("MAÐUR") == "MAÐUR"

    def test_lowercase_ae(self):
        """æ (lowercase ae) must be preserved."""
        assert canonicalize("æðri") == "æðri"
        assert canonicalize("sæll") == "sæll"

    def test_uppercase_ae(self):
        """Æ (uppercase ae) must be preserved."""
        assert canonicalize("ÆÐRI") == "ÆÐRI"

    def test_lowercase_o_umlaut(self):
        """ö (lowercase o with umlaut) must be preserved."""
        assert canonicalize("lögin") == "lögin"
        assert canonicalize("börn") == "börn"

    def test_uppercase_o_umlaut(self):
        """Ö (uppercase o with umlaut) must be preserved."""
        assert canonicalize("LÖGIN") == "LÖGIN"

    def test_mixed_icelandic_text(self):
        """Real Icelandic legal text must be preserved correctly."""
        text = "Lög um þjóðfána Íslendinga og ríkisskjaldarmerkið"
        assert canonicalize(text) == text

    def test_icelandic_legal_reference(self):
        """Legal reference format must be preserved."""
        text = "1. mgr. 12. gr. laga nr. 33/1944"
        assert canonicalize(text) == text

    def test_full_icelandic_article(self):
        """Full article text with all Icelandic characters."""
        text = "Þjóðfáni Íslendinga er himinblár með mjóum hvítum krossi, rauðum að innanverðu."
        assert canonicalize(text) == text


class TestNonBreakingSpaces:
    """Tests for NBSP and other Unicode space handling."""

    def test_nbsp_replaced_with_space(self):
        """NBSP (U+00A0) must be replaced with regular space."""
        text_with_nbsp = "1.\u00A0gr."  # 1. gr. with NBSP
        assert canonicalize(text_with_nbsp) == "1. gr."

    def test_nbsp_in_legal_reference(self):
        """NBSP in legal references must be normalized."""
        text = "Lög\u00A0nr.\u00A012/2020"
        assert canonicalize(text) == "Lög nr. 12/2020"

    def test_multiple_nbsp(self):
        """Multiple NBSPs must be collapsed to single space."""
        text = "word\u00A0\u00A0\u00A0word"
        assert canonicalize(text) == "word word"

    def test_narrow_nbsp(self):
        """Narrow NBSP (U+202F) must be replaced."""
        text = "100\u202F000"  # Thousand separator
        assert canonicalize(text) == "100 000"

    def test_zero_width_nbsp(self):
        """Zero-width NBSP / BOM (U+FEFF) must be removed."""
        text = "\uFEFFHello"
        assert canonicalize(text) == "Hello"

    def test_various_unicode_spaces(self):
        """Various Unicode space characters must be normalized."""
        # EN SPACE
        assert canonicalize("a\u2002b") == "a b"
        # EM SPACE
        assert canonicalize("a\u2003b") == "a b"
        # THIN SPACE
        assert canonicalize("a\u2009b") == "a b"
        # HAIR SPACE
        assert canonicalize("a\u200Ab") == "a b"


class TestWhitespaceCollapsing:
    """Tests for whitespace collapsing behavior."""

    def test_multiple_spaces(self):
        """Multiple spaces must be collapsed to single space."""
        assert canonicalize("word    word") == "word word"

    def test_tabs(self):
        """Tabs must be replaced with single space."""
        assert canonicalize("word\tword") == "word word"

    def test_newlines(self):
        """Newlines must be replaced with single space."""
        assert canonicalize("word\nword") == "word word"

    def test_carriage_return(self):
        """Carriage returns must be replaced."""
        assert canonicalize("word\rword") == "word word"

    def test_crlf(self):
        """CRLF must be collapsed to single space."""
        assert canonicalize("word\r\nword") == "word word"

    def test_multiple_newlines(self):
        """Multiple newlines must be collapsed."""
        assert canonicalize("word\n\n\nword") == "word word"

    def test_mixed_whitespace(self):
        """Mixed whitespace types must be collapsed."""
        assert canonicalize("word \t\n  \r\n word") == "word word"

    def test_leading_whitespace(self):
        """Leading whitespace must be stripped."""
        assert canonicalize("   word") == "word"
        assert canonicalize("\n\nword") == "word"
        assert canonicalize("\t  \n word") == "word"

    def test_trailing_whitespace(self):
        """Trailing whitespace must be stripped."""
        assert canonicalize("word   ") == "word"
        assert canonicalize("word\n\n") == "word"
        assert canonicalize("word \t  \n") == "word"

    def test_both_ends_whitespace(self):
        """Both leading and trailing whitespace must be stripped."""
        assert canonicalize("  word  ") == "word"


class TestUnicodeNormalization:
    """Tests for Unicode NFC normalization."""

    def test_nfc_composed_preserved(self):
        """Already-composed characters stay composed."""
        # é as single codepoint (U+00E9)
        composed = "caf\u00E9"
        assert canonicalize(composed) == "café"

    def test_nfd_becomes_nfc(self):
        """Decomposed characters become composed (NFC)."""
        # é as e + combining acute accent (U+0301)
        decomposed = "cafe\u0301"
        result = canonicalize(decomposed)
        # After NFC, should be single codepoint
        assert result == "café"
        assert len(result) == 4  # Not 5

    def test_icelandic_accents_nfc(self):
        """Icelandic accented characters normalize correctly."""
        # í as composed
        composed = "Íslendinga"
        assert canonicalize(composed) == "Íslendinga"

        # í as decomposed (i + combining acute)
        decomposed = "I\u0301slendinga"
        result = canonicalize(decomposed)
        assert result == "Íslendinga"

    def test_nfc_equivalence(self):
        """Composed and decomposed versions become equal."""
        composed = "Lög um þjóðfána Íslendinga"
        # Create decomposed version
        decomposed = unicodedata.normalize('NFD', composed)

        assert canonicalize(composed) == canonicalize(decomposed)


class TestCaseAndPunctuationPreservation:
    """Tests to ensure case and punctuation are NOT changed."""

    def test_case_preserved(self):
        """Case must NOT be changed."""
        assert canonicalize("ABC") == "ABC"
        assert canonicalize("abc") == "abc"
        assert canonicalize("AbC") == "AbC"

    def test_punctuation_preserved(self):
        """Punctuation must NOT be changed."""
        assert canonicalize("Hello, World!") == "Hello, World!"
        assert canonicalize("§ 1. gr.") == "§ 1. gr."
        assert canonicalize("a; b: c.") == "a; b: c."

    def test_legal_punctuation(self):
        """Legal citation punctuation preserved."""
        text = "Lög nr. 33/1944 - 1. gr., 2. mgr."
        assert canonicalize(text) == text

    def test_quotes_preserved(self):
        """Various quote types preserved."""
        assert canonicalize('"quote"') == '"quote"'
        assert canonicalize("'quote'") == "'quote'"
        # Icelandic quotes (low-high style)
        icelandic_quotes = '\u201Equote\u201C'  # „quote"
        assert canonicalize(icelandic_quotes) == icelandic_quotes

    def test_special_legal_symbols(self):
        """Legal symbols preserved."""
        assert canonicalize("§") == "§"
        assert canonicalize("№") == "№"
        assert canonicalize("©") == "©"


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert canonicalize("") == ""

    def test_none_returns_empty(self):
        """None returns empty string."""
        assert canonicalize(None) == ""

    def test_whitespace_only(self):
        """Whitespace-only string returns empty."""
        assert canonicalize("   ") == ""
        assert canonicalize("\n\t\r") == ""

    def test_single_character(self):
        """Single characters work correctly."""
        assert canonicalize("a") == "a"
        assert canonicalize("þ") == "þ"
        assert canonicalize(" ") == ""

    def test_very_long_text(self):
        """Long text is handled correctly."""
        long_text = "word " * 10000
        result = canonicalize(long_text)
        assert result == ("word " * 9999 + "word").strip()

    def test_numeric_input(self):
        """Non-string input is converted to string."""
        assert canonicalize(123) == "123"
        assert canonicalize(12.34) == "12.34"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_texts_match_simple(self):
        """texts_match() works for simple cases."""
        assert texts_match("hello", "hello")
        assert texts_match("hello ", " hello")
        assert not texts_match("hello", "world")

    def test_texts_match_whitespace(self):
        """texts_match() handles whitespace differences."""
        assert texts_match("a  b", "a b")
        assert texts_match("a\nb", "a b")
        assert texts_match("a\u00A0b", "a b")

    def test_texts_match_icelandic(self):
        """texts_match() works for Icelandic text."""
        assert texts_match("Þórður", "Þórður")
        assert texts_match("Þórður ", " Þórður")

    def test_quote_exists_in_source_basic(self):
        """quote_exists_in_source() basic functionality."""
        source = "This is a test sentence with some content."
        assert quote_exists_in_source("test sentence", source)
        assert quote_exists_in_source("some content", source)
        assert not quote_exists_in_source("missing text", source)

    def test_quote_exists_with_whitespace_differences(self):
        """quote_exists_in_source() handles whitespace in both."""
        source = "Lög  nr.  33/1944"
        quote = "Lög nr. 33/1944"
        assert quote_exists_in_source(quote, source)

    def test_quote_exists_with_nbsp(self):
        """quote_exists_in_source() handles NBSP differences."""
        source = "1.\u00A0gr.\u00A0laga"  # NBSPs in source
        quote = "1. gr. laga"  # Regular spaces in quote
        assert quote_exists_in_source(quote, source)

    def test_quote_exists_icelandic(self):
        """quote_exists_in_source() works for Icelandic quotes."""
        source = "Þjóðfáni Íslendinga er himinblár með mjóum hvítum krossi."
        quote = "himinblár með mjóum"
        assert quote_exists_in_source(quote, source)

    def test_quote_exists_empty_quote(self):
        """Empty quote returns False."""
        assert not quote_exists_in_source("", "some source")
        assert not quote_exists_in_source("   ", "some source")

    def test_canonicalize_for_search_same_as_canonicalize(self):
        """canonicalize_for_search is same as canonicalize."""
        text = "Test  text\u00A0with\nwhitespace"
        assert canonicalize_for_search(text) == canonicalize(text)

    def test_canonicalize_for_validation_same_as_canonicalize(self):
        """canonicalize_for_validation is same as canonicalize."""
        text = "Test  text\u00A0with\nwhitespace"
        assert canonicalize_for_validation(text) == canonicalize(text)


class TestRealLegalText:
    """Tests using real Icelandic legal text patterns."""

    def test_law_title(self):
        """Law titles normalize correctly."""
        title = "Lög  um\u00A0þjóðfána\n Íslendinga  og ríkisskjaldarmerkið"
        expected = "Lög um þjóðfána Íslendinga og ríkisskjaldarmerkið"
        assert canonicalize(title) == expected

    def test_article_reference(self):
        """Article references normalize correctly."""
        ref = "sbr.\u00A0 1.\u00A0mgr.\u00A012.\u00A0gr."
        expected = "sbr. 1. mgr. 12. gr."
        assert canonicalize(ref) == expected

    def test_law_number_format(self):
        """Law number formats are preserved."""
        assert canonicalize("lög nr. 33/1944") == "lög nr. 33/1944"
        assert canonicalize("l.  nr.\u00A033/1944") == "l. nr. 33/1944"

    def test_locator_format(self):
        """Locator format is preserved correctly."""
        locator = "Lög nr. 33/1944 - 1. gr., 2. mgr."
        assert canonicalize(locator) == locator

    def test_quote_validation_scenario(self):
        """
        Real-world validation scenario:
        - Source has NBSP and extra spaces
        - LLM quote has regular spaces
        - Should match after canonicalization
        """
        # Simulated source text from database (with NBSP)
        source = "Þjóðfáni\u00A0Íslendinga er\u00A0himinblár  með mjóum hvítum krossi"

        # Simulated LLM output quote (regular spaces)
        llm_quote = "Þjóðfáni Íslendinga er himinblár með mjóum hvítum krossi"

        assert quote_exists_in_source(llm_quote, source)

    def test_partial_quote_validation(self):
        """Partial quotes from within articles."""
        source = """
        Þjóðfáni Íslendinga er himinblár með mjóum hvítum krossi,
        rauðum að innanverðu. Armar krossins ná að jöðrum fánans.
        """

        # Quote from middle of text
        quote = "himinblár með mjóum hvítum krossi, rauðum að innanverðu"
        assert quote_exists_in_source(quote, source)


def run_all_tests():
    """Run all canonicalize tests."""
    print("=" * 60)
    print("CANONICALIZE TESTS")
    print("=" * 60)

    # Icelandic characters
    print("\n--- Icelandic Character Tests ---")
    ice_tests = TestIcelandicCharacters()
    ice_tests.test_lowercase_thorn()
    print("  test_lowercase_thorn: PASS")
    ice_tests.test_lowercase_eth()
    print("  test_lowercase_eth: PASS")
    ice_tests.test_lowercase_ae()
    print("  test_lowercase_ae: PASS")
    ice_tests.test_lowercase_o_umlaut()
    print("  test_lowercase_o_umlaut: PASS")
    ice_tests.test_mixed_icelandic_text()
    print("  test_mixed_icelandic_text: PASS")

    # NBSP
    print("\n--- Non-Breaking Space Tests ---")
    nbsp_tests = TestNonBreakingSpaces()
    nbsp_tests.test_nbsp_replaced_with_space()
    print("  test_nbsp_replaced_with_space: PASS")
    nbsp_tests.test_multiple_nbsp()
    print("  test_multiple_nbsp: PASS")

    # Whitespace
    print("\n--- Whitespace Collapsing Tests ---")
    ws_tests = TestWhitespaceCollapsing()
    ws_tests.test_multiple_spaces()
    print("  test_multiple_spaces: PASS")
    ws_tests.test_tabs()
    print("  test_tabs: PASS")
    ws_tests.test_newlines()
    print("  test_newlines: PASS")

    # Unicode NFC
    print("\n--- Unicode NFC Tests ---")
    nfc_tests = TestUnicodeNormalization()
    nfc_tests.test_nfc_composed_preserved()
    print("  test_nfc_composed_preserved: PASS")
    nfc_tests.test_nfd_becomes_nfc()
    print("  test_nfd_becomes_nfc: PASS")

    # Helper functions
    print("\n--- Helper Function Tests ---")
    helper_tests = TestHelperFunctions()
    helper_tests.test_texts_match_simple()
    print("  test_texts_match_simple: PASS")
    helper_tests.test_quote_exists_in_source_basic()
    print("  test_quote_exists_in_source_basic: PASS")
    helper_tests.test_quote_exists_with_nbsp()
    print("  test_quote_exists_with_nbsp: PASS")

    # Real legal text
    print("\n--- Real Legal Text Tests ---")
    legal_tests = TestRealLegalText()
    legal_tests.test_quote_validation_scenario()
    print("  test_quote_validation_scenario: PASS")

    print("\n" + "=" * 60)
    print("ALL CANONICALIZE TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
