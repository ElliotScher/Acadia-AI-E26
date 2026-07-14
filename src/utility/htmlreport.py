"""
HTML Report Helpers
"""

import html
from typing import Any, List

_CELL_FONT = "font-family:Calibri,Arial,sans-serif; font-size:11pt;"
_HEADING_FONT = "font-family:Calibri,Arial,sans-serif;"


def html_table(headers: List[str], rows: List[List[Any]]) -> str:
    """
    Renders a bordered HTML table.

    Args:
        headers (List[str]): Column header labels.
        rows (List[List[Any]]): Row values; each is str()'d and HTML-escaped.

    Returns:
        str: An HTML <table> fragment.
    """
    th_style = (
        f"border:1px solid #444444; background-color:#f2f2f2; "
        f"padding:4px 10px; text-align:left; font-weight:bold; {_CELL_FONT}"
    )
    td_style = (
        f"border:1px solid #444444; padding:4px 10px; text-align:left; {_CELL_FONT}"
    )

    header_cells = "".join(
        f'<th style="{th_style}">{html.escape(h)}</th>' for h in headers
    )
    body_rows = []
    for row in rows:
        cells = "".join(
            f'<td style="{td_style}">{html.escape(str(value))}</td>' for value in row
        )
        body_rows.append(f"<tr>{cells}</tr>")

    return (
        '<table border="1" cellpadding="4" cellspacing="0" '
        'style="border-collapse:collapse;">\n'
        f"<tr>{header_cells}</tr>\n" + "\n".join(body_rows) + "\n</table>"
    )


def html_heading(text: str, level: int = 2) -> str:
    """
    Renders an inline-styled heading matching html_table's font.

    Args:
        text (str): Heading text.
        level (int): Heading level (2 for h2, 3 for h3, etc.). Defaults to 2.

    Returns:
        str: An HTML heading element.
    """
    return f'<h{level} style="{_HEADING_FONT}">{html.escape(text)}</h{level}>'


def wrap_html_document(sections: List[str]) -> str:
    """
    Wraps a list of HTML fragments (headings, tables, <br> spacers) into a
    complete, standalone HTML document.

    Args:
        sections (List[str]): HTML fragments, in display order.

    Returns:
        str: A complete HTML document.
    """
    return "<html><body>\n" + "\n".join(sections) + "\n</body></html>"
