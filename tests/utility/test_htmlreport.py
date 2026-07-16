from src.utility.htmlreport import html_table, html_heading, wrap_html_document


def test_html_table_basic_structure():
    table = html_table(["Name", "Count"], [["cars", 3], ["bikes", 1]])
    assert table.startswith('<table border="1"')
    assert table.count("<th") == 2
    assert table.count("<td") == 4
    assert "Name" in table
    assert "cars" in table
    assert "3" in table


def test_html_table_escapes_headers_and_values():
    table = html_table(["<script>"], [["<b>bold</b> & stuff"]])
    assert "<script>" not in table
    assert "&lt;script&gt;" in table
    assert "&lt;b&gt;bold&lt;/b&gt; &amp; stuff" in table


def test_html_table_stringifies_non_string_values():
    table = html_table(["Value"], [[None], [3.14], [True]])
    assert "None" in table
    assert "3.14" in table
    assert "True" in table


def test_html_table_empty_rows():
    table = html_table(["A", "B"], [])
    assert "<th" in table
    assert "<td" not in table


def test_html_heading_default_level():
    heading = html_heading("Summary")
    assert heading.startswith("<h2")
    assert heading.endswith("</h2>")
    assert "Summary" in heading


def test_html_heading_custom_level():
    heading = html_heading("Details", level=3)
    assert heading.startswith("<h3")
    assert heading.endswith("</h3>")


def test_html_heading_escapes_text():
    heading = html_heading("<b>bold</b>")
    assert "<b>bold</b>" not in heading
    assert "&lt;b&gt;bold&lt;/b&gt;" in heading


def test_wrap_html_document_combines_sections_in_order():
    doc = wrap_html_document(["<h2>A</h2>", "<table></table>"])
    assert doc.startswith("<html><body>")
    assert doc.endswith("</body></html>")
    assert doc.index("<h2>A</h2>") < doc.index("<table></table>")


def test_wrap_html_document_empty_sections():
    doc = wrap_html_document([])
    assert doc == "<html><body>\n\n</body></html>"
