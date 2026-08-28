#!/usr/bin/env python
"""Rendu PDF du mémoire (02-Memoire.md -> 02-Memoire.pdf).

Usage:
    uv run python scripts/make_memoire_pdf.py [--md 02-Memoire.md] [--pdf 02-Memoire.pdf]

Mise en page type thèse : serif partout, quasi monochrome, filets fins.
Le sommaire est généré automatiquement : placer une ligne « [[sommaire]] » dans le
Markdown, à l'endroit voulu. Les titres qui la suivent y figurent avec leur numéro
de page réel, résolu au rendu — il reste donc juste après n'importe quelle coupe.
Sauts de page forcés UNIQUEMENT devant les quatre intercalaires « Partie N » —
jamais devant les chapitres : le document mélange chapitres rédigés et chapitres
encore à l'état de plan (quelques lignes de tableau), et un saut par chapitre
laisserait la page des chapitres courts presque vide.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import markdown
from weasyprint import HTML

CSS = """
@page {
    size: A4;
    margin: 2.6cm 2.4cm 2.4cm 2.4cm;
    @bottom-center {
        content: counter(page);
        font-family: "Liberation Serif", Georgia, serif;
        font-size: 9pt;
        color: #444;
    }
}
body {
    font-family: "Liberation Serif", Georgia, serif;
    font-size: 10.8pt;
    line-height: 1.55;
    color: #111;
    text-align: justify;
    orphans: 3;
    widows: 3;
}
h1, h2, h3, h4, h5 {
    font-family: "Liberation Serif", Georgia, serif;
    font-weight: 700;
    color: #111;
    text-align: left;
    break-after: avoid;
    line-height: 1.25;
}
h1 {
    font-size: 19pt;
    text-align: center;
    margin: 0 0 0.3em 0;
    letter-spacing: 0.02em;
}
h2 {
    font-size: 14pt;
    margin: 1.8em 0 0.7em 0;
    padding-bottom: 0.15em;
    border-bottom: 0.75pt solid #999;
}
h2:first-of-type { margin-top: 0; }
h3 {
    font-size: 12.5pt;
    margin: 1.5em 0 0.6em 0;
    font-variant: small-caps;
    letter-spacing: 0.02em;
}
h4 {
    font-size: 11pt;
    font-style: italic;
    font-weight: 700;
    margin: 1.2em 0 0.4em 0;
}
h2.break { break-before: page; }
p { margin: 0 0 0.6em 0; text-indent: 0; }
hr { border: none; border-top: 0.75pt solid #999; margin: 1.5em 0; }
blockquote {
    margin: 0.7em 0 0.7em 0.2em;
    padding: 0 0 0 0.9em;
    border-left: 1.5pt solid #999;
    font-style: italic;
    color: #222;
    text-align: left;
}
blockquote p { margin: 0.3em 0; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.8em 0 1em 0;
    font-size: 9.3pt;
    break-inside: avoid;
}
th, td {
    border-top: 0.5pt solid #bbb;
    border-bottom: 0.5pt solid #bbb;
    padding: 4px 8px;
    text-align: left;
    vertical-align: top;
}
thead th {
    border-top: 1pt solid #333;
    border-bottom: 1pt solid #333;
    font-weight: 700;
    text-align: left;
}
tbody tr:last-child td { border-bottom: 1pt solid #333; }
code {
    font-family: "Liberation Mono", "Courier New", monospace;
    font-size: 0.9em;
    color: #222;
}
pre code { display: block; padding: 0.6em; overflow-x: auto; background: #f5f5f5; }
strong { font-weight: 700; }
em { font-style: italic; }
a { color: #111; text-decoration: underline; }
ul, ol { margin: 0.4em 0 0.6em 0; padding-left: 1.4em; }
li { margin: 0.15em 0; }
img {
    max-width: 100%;
    display: block;
    margin: 1em auto 0.3em auto;
    break-inside: avoid;
}

/* --- Sommaire : généré depuis les titres, numéros de page réels -------------
   Disposition en grille plutôt qu'en points de conduite : un titre de section
   long se replie sans qu'une ligne entière de points ne vienne le suivre, et le
   numéro reste aligné à droite sur la première ligne de l'entrée. */
nav.toc { break-before: page; break-after: page; }
nav.toc h2 { margin: 0 0 1em 0; }
nav.toc ul { list-style: none; margin: 0; padding: 0; }
nav.toc li { margin: 0; break-inside: avoid; }
nav.toc a {
    display: grid;
    grid-template-columns: 1fr auto;
    column-gap: 0.8em;
    align-items: baseline;
    text-decoration: none;
    color: #111;
    line-height: 1.3;
    padding: 0.08em 0;
}
nav.toc a::after {
    content: target-counter(attr(href), page);
    font-variant-numeric: tabular-nums;
    color: #555;
    font-size: 9.5pt;
}
nav.toc .toc-part a {
    font-weight: 700;
    font-size: 11.5pt;
    margin-top: 0.85em;
    padding-top: 0.3em;
    border-top: 0.5pt solid #999;
}
nav.toc .toc-chap a { font-weight: 700; margin-top: 0.4em; }
nav.toc .toc-sect a {
    padding-left: 1.3em;
    font-size: 9.6pt;
    color: #333;
}
nav.toc .toc-sect a::after { color: #777; font-size: 9pt; }

/* Légende : le paragraphe en italique qui suit immédiatement une figure. */
img + em, p > img { break-after: avoid; }
"""

HTML_TEMPLATE = """<!doctype html>
<html lang="fr">
<head><meta charset="utf-8"><style>{css}</style></head>
<body>{body}</body>
</html>"""

PART_RE = re.compile(r"^Partie\s+[IVX]")


def tag_page_breaks(html: str) -> str:
    def repl(match: re.Match) -> str:
        text = match.group(1)
        plain = re.sub(r"<[^>]+>", "", text)
        cls = ' class="break"' if PART_RE.match(plain) else ""
        return f"<h2{cls}>{text}</h2>"

    return re.sub(r"<h2>(.*?)</h2>", repl, html, flags=re.DOTALL)



SOMMAIRE_MARQUEUR = "[[sommaire]]"
_ACCENTS = str.maketrans("àâäéèêëîïôöùûüç", "aaaeeeeiioouuuc")


def _ancre(texte: str, vus: dict) -> str:
    """Identifiant stable et lisible pour un titre."""
    plat = re.sub(r"<[^>]+>", "", texte)
    base = re.sub(r"[^a-z0-9]+", "-", plat.lower().translate(_ACCENTS)).strip("-")[:60]
    vus[base] = vus.get(base, 0) + 1
    return base if vus[base] == 1 else f"{base}-{vus[base]}"


def build_toc(html: str) -> str:
    """Ancre les titres et remplace le marqueur par le sommaire.

    Seuls les titres SITUÉS APRÈS le marqueur entrent au sommaire : la page de
    garde (titre, problématique, sous-questions) n'a rien à y faire. Les numéros
    de page ne sont pas calculés ici — ``target-counter`` les résout au rendu,
    ce qui garantit qu'ils restent justes après n'importe quelle révision.
    """
    coupe = html.find(SOMMAIRE_MARQUEUR)
    entrees, vus = [], {}

    def ancrer(match: re.Match) -> str:
        niveau, attrs, texte = match.group(1), match.group(2), match.group(3)
        ident = _ancre(texte, vus)
        if match.start() > coupe >= 0:
            classe = {"2": "toc-part", "3": "toc-chap", "4": "toc-sect"}.get(niveau)
            if classe:
                entrees.append((classe, ident, re.sub(r"<[^>]+>", "", texte)))
        return f'<h{niveau}{attrs} id="{ident}">{texte}</h{niveau}>'

    html = re.sub(r"<h([234])([^>]*)>(.*?)</h\1>", ancrer, html, flags=re.DOTALL)
    if coupe < 0:
        return html

    lignes = "".join(
        f'<li class="{c}"><a href="#{i}">{t}</a></li>' for c, i, t in entrees
    )
    toc = f'<nav class="toc"><h2>Sommaire</h2><ul>{lignes}</ul></nav>'
    return re.sub(r"<p>\s*\[\[sommaire\]\]\s*</p>", toc, html)


def convert(md_path: Path, pdf_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    body_html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    body_html = tag_page_breaks(body_html)
    body_html = build_toc(body_html)
    html = HTML_TEMPLATE.format(css=CSS, body=body_html)
    HTML(string=html, base_url=str(md_path)).write_pdf(str(pdf_path))
    print(f"OK: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--md", type=Path, default=Path("02-Memoire.md"))
    parser.add_argument("--pdf", type=Path, default=Path("02-Memoire.pdf"))
    args = parser.parse_args()
    convert(args.md, args.pdf)


if __name__ == "__main__":
    main()
