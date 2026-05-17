from pathlib import Path

from rag_pipeline.ingest.parsers import parse


def test_parse_markdown(tmp_path: Path):
    p = tmp_path / "a.md"
    p.write_text("# Title\n\nbody text")
    stype, text = parse(p)
    assert stype == "md"
    assert "body text" in text


def test_parse_html_strips_tags(tmp_path: Path):
    p = tmp_path / "a.html"
    p.write_text("<html><body><script>x()</script><p>visible</p></body></html>")
    stype, text = parse(p)
    assert stype == "html"
    assert "visible" in text
    assert "x()" not in text
