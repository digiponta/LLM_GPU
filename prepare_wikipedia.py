# prepare_wikipedia.py
#
# Convert a Japanese Wikipedia XML dump into plain-text training data.
#
# Input:
#   jawiki-latest-pages-articles.xml.bz2
#
# Output:
#   general-ja.txt
#
# Features:
#   - Reads .xml.bz2 directly without extracting it first
#   - Uses only Python standard library
#   - Extracts namespace 0 articles
#   - Skips redirects
#   - Removes common MediaWiki markup
#   - Writes UTF-8 plain text
#   - Can limit article count for small experiments

from __future__ import annotations

import argparse
import bz2
import html
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import IO, Optional


COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
REF_BLOCK_RE = re.compile(r"<ref\b[^>]*>.*?</ref>", re.IGNORECASE | re.DOTALL)
REF_SINGLE_RE = re.compile(r"<ref\b[^>]*/\s*>", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)
FILE_LINK_RE = re.compile(r"\[\[(?:ファイル|画像|File|Image):.*?\]\]", re.IGNORECASE | re.DOTALL)
CATEGORY_RE = re.compile(r"\[\[(?:Category|カテゴリ):.*?\]\]", re.IGNORECASE | re.DOTALL)
EXTERNAL_LINK_RE = re.compile(r"\[(https?://[^\s\]]+)(?:\s+([^\]]+))?\]")
URL_RE = re.compile(r"https?://\S+")
HEADING_RE = re.compile(r"^\s*=+\s*(.*?)\s*=+\s*$", re.MULTILINE)
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def remove_balanced_templates(text: str) -> str:
    result = []
    depth = 0
    i = 0
    while i < len(text):
        if text.startswith("{{", i):
            depth += 1
            i += 2
            continue
        if text.startswith("}}", i) and depth > 0:
            depth -= 1
            i += 2
            continue
        if depth == 0:
            result.append(text[i])
        i += 1
    return "".join(result)


def replace_internal_links(text: str) -> str:
    pattern = re.compile(r"\[\[([^\[\]]+?)\]\]")

    def replace(match: re.Match) -> str:
        content = match.group(1)
        if "|" in content:
            return content.split("|")[-1].strip()
        return content.split("#", 1)[0].strip()

    previous = None
    while previous != text:
        previous = text
        text = pattern.sub(replace, text)
    return text


def clean_wikitext(text: str) -> str:
    if not text:
        return ""

    text = html.unescape(text)
    text = COMMENT_RE.sub("", text)
    text = REF_BLOCK_RE.sub("", text)
    text = REF_SINGLE_RE.sub("", text)
    text = remove_balanced_templates(text)
    text = FILE_LINK_RE.sub("", text)
    text = CATEGORY_RE.sub("", text)
    text = replace_internal_links(text)

    def replace_external(match: re.Match) -> str:
        label = match.group(2)
        return label.strip() if label else ""

    text = EXTERNAL_LINK_RE.sub(replace_external, text)
    text = URL_RE.sub("", text)
    text = HEADING_RE.sub(r"\1", text)
    text = TAG_RE.sub("", text)
    text = text.replace("'''", "").replace("''", "")
    text = text.replace("{|", "").replace("|}", "")

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            lines.append("")
            continue
        line = re.sub(r"^[*#:;]+\s*", "", line)
        if line.startswith("|-"):
            continue
        if line.startswith("!"):
            line = line.lstrip("!").strip()
        if line.startswith("|"):
            line = line.lstrip("|").strip()
        line = MULTI_SPACE_RE.sub(" ", line)
        lines.append(line)

    text = "\n".join(lines)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def is_redirect(text: str) -> bool:
    stripped = text.lstrip()
    prefixes = ("#REDIRECT", "#redirect", "#転送", "#リダイレクト")
    return stripped.startswith(prefixes)


def local_name(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def child_text(element: ET.Element, child_name: str) -> Optional[str]:
    for child in element:
        if local_name(child.tag) == child_name:
            return child.text
    return None


def find_revision_text(page: ET.Element) -> Optional[str]:
    for child in page:
        if local_name(child.tag) != "revision":
            continue
        for revision_child in child:
            if local_name(revision_child.tag) == "text":
                return revision_child.text or ""
    return None


def open_dump(filename: str) -> IO[bytes]:
    path = Path(filename)
    if path.suffix == ".bz2":
        return bz2.open(path, "rb")
    return open(path, "rb")


def prepare_wikipedia(
    input_filename: str,
    output_filename: str,
    max_articles: Optional[int] = None,
    min_chars: int = 100,
    include_title: bool = False,
    progress_interval: int = 1000,
) -> dict:
    article_count = 0
    skipped_redirects = 0
    skipped_namespace = 0
    skipped_short = 0
    chars_written = 0

    with open_dump(input_filename) as input_file:
        with open(output_filename, "w", encoding="utf-8") as output_file:
            context = ET.iterparse(input_file, events=("end",))

            for _event, element in context:
                if local_name(element.tag) != "page":
                    continue

                title = child_text(element, "title")
                namespace = child_text(element, "ns")

                if namespace != "0":
                    skipped_namespace += 1
                    element.clear()
                    continue

                raw_text = find_revision_text(element)
                if not raw_text:
                    element.clear()
                    continue

                if is_redirect(raw_text):
                    skipped_redirects += 1
                    element.clear()
                    continue

                cleaned = clean_wikitext(raw_text)
                if len(cleaned) < min_chars:
                    skipped_short += 1
                    element.clear()
                    continue

                if include_title and title:
                    document = title.strip() + "\n\n" + cleaned
                else:
                    document = cleaned

                output_file.write(document)
                output_file.write("\n\n")

                article_count += 1
                chars_written += len(document) + 2

                if progress_interval > 0 and article_count % progress_interval == 0:
                    print(
                        f"{article_count:,} articles written ({chars_written:,} chars)",
                        file=sys.stderr,
                    )

                element.clear()

                if max_articles is not None and article_count >= max_articles:
                    break

    return {
        "articles": article_count,
        "characters": chars_written,
        "redirects_skipped": skipped_redirects,
        "non_main_namespace_skipped": skipped_namespace,
        "short_articles_skipped": skipped_short,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Japanese Wikipedia pages-articles XML dump "
            "into plain UTF-8 training text."
        )
    )

    parser.add_argument("input", help="Wikipedia XML or XML.bz2 dump file")
    parser.add_argument(
        "output",
        nargs="?",
        default="general-ja.txt",
        help="Output UTF-8 text file (default: general-ja.txt)",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Stop after this number of accepted articles.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=100,
        help="Minimum cleaned article length (default: 100)",
    )
    parser.add_argument(
        "--include-title",
        action="store_true",
        help="Write the Wikipedia article title before each article.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Print progress every N articles (default: 1000; 0 disables)",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("Preparing Wikipedia corpus...")
    print(f"Input : {args.input}")
    print(f"Output: {args.output}")

    result = prepare_wikipedia(
        input_filename=args.input,
        output_filename=args.output,
        max_articles=args.max_articles,
        min_chars=args.min_chars,
        include_title=args.include_title,
        progress_interval=args.progress_interval,
    )

    print()
    print("Completed.")
    print(f"Articles : {result['articles']:,}")
    print(f"Characters: {result['characters']:,}")
    print(f"Redirects skipped: {result['redirects_skipped']:,}")
    print(
        "Non-main namespace skipped: "
        f"{result['non_main_namespace_skipped']:,}"
    )
    print(f"Short articles skipped: {result['short_articles_skipped']:,}")


if __name__ == "__main__":
    main()
