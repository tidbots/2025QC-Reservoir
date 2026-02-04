#!/usr/bin/env python3
"""
Markdown to PDF Converter
=========================
Converts markdown files to PDF with image support.

Usage:
    python3 md2pdf.py <input.md> [output.pdf]
"""

import sys
import os

# Add user site-packages to path
sys.path.insert(0, os.path.expanduser('~/.local/lib/python3.10/site-packages'))

import markdown
from weasyprint import HTML, CSS
from pathlib import Path


CSS_STYLE = """
@page {
    size: A4;
    margin: 2cm;
}

body {
    font-family: 'DejaVu Sans', 'Noto Sans CJK JP', sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
}

h1 {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 0.3em;
    font-size: 24pt;
}

h2 {
    color: #2980b9;
    border-bottom: 1px solid #bdc3c7;
    padding-bottom: 0.2em;
    font-size: 18pt;
    margin-top: 1.5em;
}

h3 {
    color: #27ae60;
    font-size: 14pt;
    margin-top: 1.2em;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
}

th, td {
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
}

th {
    background-color: #3498db;
    color: white;
}

tr:nth-child(even) {
    background-color: #f9f9f9;
}

code {
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'DejaVu Sans Mono', monospace;
    font-size: 10pt;
}

pre {
    background-color: #2d2d2d;
    color: #f8f8f2;
    padding: 1em;
    border-radius: 5px;
    overflow-x: auto;
    font-size: 9pt;
}

pre code {
    background-color: transparent;
    color: inherit;
    padding: 0;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1em auto;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 2em 0;
}

ul, ol {
    margin: 0.5em 0;
    padding-left: 2em;
}

li {
    margin: 0.3em 0;
}

strong {
    color: #2c3e50;
}

blockquote {
    border-left: 4px solid #3498db;
    margin: 1em 0;
    padding: 0.5em 1em;
    background-color: #f9f9f9;
}
"""


def convert_md_to_pdf(input_path, output_path=None):
    """Convert markdown file to PDF."""
    input_path = Path(input_path).resolve()

    if output_path is None:
        output_path = input_path.with_suffix('.pdf')
    else:
        output_path = Path(output_path).resolve()

    # Read markdown content
    with open(input_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert markdown to HTML
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'])
    html_content = md.convert(md_content)

    # Fix image paths to be absolute
    base_dir = input_path.parent
    html_content = html_content.replace('src="images/', f'src="file://{base_dir}/images/')
    html_content = html_content.replace("src='images/", f"src='file://{base_dir}/images/")

    # Wrap in HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{input_path.stem}</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Convert to PDF
    html = HTML(string=full_html, base_url=str(base_dir))
    css = CSS(string=CSS_STYLE)
    html.write_pdf(output_path, stylesheets=[css])

    print(f"PDF generated: {output_path}")
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 md2pdf.py <input.md> [output.pdf]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    convert_md_to_pdf(input_path, output_path)


if __name__ == '__main__':
    main()
