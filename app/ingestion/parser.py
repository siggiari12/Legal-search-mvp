"""
SGML Parser for Lagasafn (Icelandic Law Collection)

This module parses Lagasafn SGML files into structured data.
It extracts:
- Law metadata (number, year, title)
- Articles (greinar)
- Paragraphs (málsgreinar) where available

IMPORTANT: SGML parsing is inherently risky. This parser:
- Uses BeautifulSoup with lenient HTML parser (not strict XML)
- Fails loudly on structural errors
- Keeps locators conservative (law + article only) rather than guessing
- Applies canonicalize() to all text

Usage:
    parser = SGMLParser()
    parsed_law = parser.parse(sgml_content)
"""

import re
import warnings
from typing import List, Optional, Tuple, Dict
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from app.services.canonicalize import canonicalize
from app.models.schemas import ParsedArticle, ParsedLaw


class SGMLParseError(Exception):
    """Raised when SGML parsing fails."""
    pass


class SGMLParser:
    """
    Parser for Lagasafn SGML files.

    Supports multiple SGML tag conventions found in Lagasafn:
    - <log> or <law> for law container
    - <nr> for law number
    - <ar> for year
    - <heiti> or <title> for title
    - <grein> or <gr> for article
    - <mgr> for paragraph
    """

    # Regex patterns for extracting law number/year from text
    LAW_NUMBER_PATTERN = re.compile(r'(\d+)[./](\d{4})')
    YEAR_PATTERN = re.compile(r'\b(19\d{2}|20\d{2})\b')
    NUMBER_PATTERN = re.compile(r'^\s*(\d+)\s*$')

    def __init__(self, strict: bool = False):
        """
        Initialize parser.

        Args:
            strict: If True, raise errors on any parsing issue.
                   If False, try to recover and log warnings.
        """
        self.strict = strict
        self.warnings: List[str] = []

    def parse(self, sgml_content: str, source_info: str = "unknown") -> ParsedLaw:
        """
        Parse SGML content into a structured ParsedLaw object.

        Args:
            sgml_content: Raw SGML string
            source_info: Identifier for error messages (e.g., filename)

        Returns:
            ParsedLaw object with extracted data

        Raises:
            SGMLParseError: If parsing fails and strict mode is enabled
        """
        self.warnings = []

        # Preprocess SGML for common issues
        cleaned = self._preprocess_sgml(sgml_content)

        # Parse with lenient HTML parser (intentional for malformed SGML)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(cleaned, 'html.parser')

        # Extract law metadata
        law_number, law_year = self._extract_law_number_year(soup, source_info)
        title = self._extract_title(soup, law_number, law_year)

        # Extract articles
        articles = self._extract_articles(soup, source_info)

        # Build full text from articles
        full_text = self._build_full_text(title, articles)

        # Validate minimum requirements
        self._validate_parsed_law(law_number, law_year, title, articles, source_info)

        return ParsedLaw(
            law_number=law_number,
            law_year=law_year,
            title=canonicalize(title),
            articles=articles,
            full_text=canonicalize(full_text),
            metadata={
                "source_info": source_info,
                "parser_warnings": self.warnings,
            }
        )

    def _preprocess_sgml(self, content: str) -> str:
        """
        Preprocess SGML to handle common issues.

        - Converts HTML entities
        - Handles unclosed tags (best effort)
        """
        # Replace common SGML entities
        content = content.replace('&apos;', "'")
        content = content.replace('&quot;', '"')

        # Some Lagasafn files use <gr.> instead of <gr>
        content = re.sub(r'<gr\.>', '<gr>', content, flags=re.IGNORECASE)
        content = re.sub(r'</gr\.>', '</gr>', content, flags=re.IGNORECASE)

        return content

    def _extract_law_number_year(
        self,
        soup: BeautifulSoup,
        source_info: str
    ) -> Tuple[str, str]:
        """
        Extract law number and year from SGML.

        Tries multiple strategies:
        1. Look for <nr> and <ar> tags
        2. Look for combined format in tag text (e.g., "33/1944")
        3. Parse from title or header text
        """
        law_number = None
        law_year = None

        # Strategy 1: Look for dedicated tags
        nr_tag = soup.find(['nr', 'number', 'lognr'])
        ar_tag = soup.find(['ar', 'year', 'artal'])

        if nr_tag and nr_tag.get_text(strip=True):
            text = nr_tag.get_text(strip=True)
            # May be "33" or "33/1944"
            match = self.LAW_NUMBER_PATTERN.search(text)
            if match:
                law_number, law_year = match.groups()
            else:
                num_match = self.NUMBER_PATTERN.match(text)
                if num_match:
                    law_number = num_match.group(1)

        if ar_tag and ar_tag.get_text(strip=True):
            text = ar_tag.get_text(strip=True)
            year_match = self.YEAR_PATTERN.search(text)
            if year_match:
                law_year = year_match.group(1)

        # Strategy 2: Look in header/title for "nr. X/YYYY" pattern
        if not (law_number and law_year):
            header_tags = soup.find_all(['heiti', 'title', 'header', 'h1', 'log'])
            for tag in header_tags:
                text = tag.get_text()
                match = self.LAW_NUMBER_PATTERN.search(text)
                if match:
                    law_number = law_number or match.group(1)
                    law_year = law_year or match.group(2)
                    break

        # Strategy 3: Search entire document
        if not (law_number and law_year):
            full_text = soup.get_text()
            match = self.LAW_NUMBER_PATTERN.search(full_text[:500])  # Check first 500 chars
            if match:
                law_number = law_number or match.group(1)
                law_year = law_year or match.group(2)
                self.warnings.append(f"Law number extracted from body text: {match.group()}")

        # Validate results
        if not law_number:
            if self.strict:
                raise SGMLParseError(f"Could not extract law number from {source_info}")
            law_number = "0"
            self.warnings.append("Law number not found, using '0'")

        if not law_year:
            if self.strict:
                raise SGMLParseError(f"Could not extract law year from {source_info}")
            law_year = "0000"
            self.warnings.append("Law year not found, using '0000'")

        return law_number, law_year

    def _extract_title(
        self,
        soup: BeautifulSoup,
        law_number: str,
        law_year: str
    ) -> str:
        """
        Extract law title from SGML.
        """
        # Try dedicated title tags
        title_tags = soup.find_all(['heiti', 'title', 'fyrirsogn'])
        for tag in title_tags:
            text = tag.get_text(strip=True)
            if text and len(text) > 5:  # Skip very short "titles"
                return text

        # Try header tags
        header_tags = soup.find_all(['h1', 'h2', 'header'])
        for tag in header_tags:
            text = tag.get_text(strip=True)
            if text and len(text) > 10:
                return text

        # Fallback: use law reference
        self.warnings.append("Title not found, using law reference as title")
        return f"Lög nr. {law_number}/{law_year}"

    def _extract_articles(
        self,
        soup: BeautifulSoup,
        source_info: str
    ) -> List[ParsedArticle]:
        """
        Extract articles (greinar) from SGML.
        """
        articles = []

        # Find article containers
        article_tags = soup.find_all(['grein', 'gr', 'article', 'art'])

        for i, tag in enumerate(article_tags):
            article_num = self._extract_article_number(tag, i + 1)
            article_text = tag.get_text(separator=' ', strip=True)

            if not article_text:
                self.warnings.append(f"Empty article {article_num} skipped")
                continue

            # Extract paragraphs if available
            paragraphs = self._extract_paragraphs(tag)

            articles.append(ParsedArticle(
                number=article_num,
                text=canonicalize(article_text),
                paragraphs=paragraphs
            ))

        # If no article tags found, try to split by pattern
        if not articles:
            articles = self._split_by_article_pattern(soup, source_info)

        return articles

    def _extract_article_number(self, tag, fallback_index: int) -> str:
        """
        Extract article number from a tag.
        """
        # Check for nr attribute
        nr_attr = tag.get('nr') or tag.get('number') or tag.get('num')
        if nr_attr:
            return str(nr_attr)

        # Check for nested <nr> tag
        nr_tag = tag.find('nr')
        if nr_tag:
            text = nr_tag.get_text(strip=True)
            num_match = re.search(r'(\d+)', text)
            if num_match:
                return num_match.group(1)

        # Check text beginning for "X. gr." pattern
        text = tag.get_text(strip=True)[:50]
        match = re.search(r'^(\d+)\s*\.?\s*gr', text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Fallback to index
        return str(fallback_index)

    def _extract_paragraphs(self, article_tag) -> List[Dict[str, str]]:
        """
        Extract paragraphs (málsgreinar) from an article.
        """
        paragraphs = []

        mgr_tags = article_tag.find_all(['mgr', 'malsgrein', 'para', 'p'])

        for i, tag in enumerate(mgr_tags):
            text = tag.get_text(separator=' ', strip=True)
            if text:
                paragraphs.append({
                    "number": str(i + 1),
                    "text": canonicalize(text)
                })

        return paragraphs

    def _split_by_article_pattern(
        self,
        soup: BeautifulSoup,
        source_info: str
    ) -> List[ParsedArticle]:
        """
        Fallback: split text by "X. gr." pattern when no article tags exist.
        """
        full_text = soup.get_text()

        # Pattern: number followed by ". gr" (article marker in Icelandic)
        pattern = re.compile(r'(\d+)\s*\.\s*gr\.?', re.IGNORECASE)

        articles = []
        matches = list(pattern.finditer(full_text))

        if not matches:
            # No article structure found - treat entire text as single chunk
            self.warnings.append("No article structure found, treating as single article")
            articles.append(ParsedArticle(
                number="1",
                text=canonicalize(full_text),
                paragraphs=[]
            ))
            return articles

        for i, match in enumerate(matches):
            article_num = match.group(1)
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)

            article_text = full_text[start:end].strip()
            if article_text:
                articles.append(ParsedArticle(
                    number=article_num,
                    text=canonicalize(article_text),
                    paragraphs=[]
                ))

        return articles

    def _build_full_text(self, title: str, articles: List[ParsedArticle]) -> str:
        """
        Build full document text from title and articles.
        """
        parts = [title, ""]

        for article in articles:
            parts.append(f"{article.number}. gr.")
            parts.append(article.text)
            parts.append("")

        return "\n".join(parts)

    def _validate_parsed_law(
        self,
        law_number: str,
        law_year: str,
        title: str,
        articles: List[ParsedArticle],
        source_info: str
    ) -> None:
        """
        Validate parsed law meets minimum requirements.
        """
        errors = []

        if not law_number or law_number == "0":
            errors.append("Invalid law number")

        if not law_year or law_year == "0000":
            errors.append("Invalid law year")

        if not title:
            errors.append("Empty title")

        if not articles:
            errors.append("No articles extracted")

        # Check for empty articles
        empty_articles = [a for a in articles if not a.text.strip()]
        if empty_articles:
            errors.append(f"{len(empty_articles)} empty articles found")

        if errors and self.strict:
            raise SGMLParseError(f"Validation failed for {source_info}: {', '.join(errors)}")

        for error in errors:
            self.warnings.append(error)


def build_locator(
    law_number: str,
    law_year: str,
    article_number: Optional[str] = None,
    paragraph_number: Optional[str] = None
) -> str:
    """
    Build a locator string for a chunk.

    Format: "Lög nr. {number}/{year} - {article}. gr., {paragraph}. mgr."

    IMPORTANT: Locators are built ONLY from parsed structure.
    Never infer or guess article/paragraph numbers.
    """
    base = f"Lög nr. {law_number}/{law_year}"

    if article_number:
        base += f" - {article_number}. gr."

        if paragraph_number:
            base += f", {paragraph_number}. mgr."

    return base
