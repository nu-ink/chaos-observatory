from __future__ import annotations

from ml.ml_embeddings import ArticleInput, build_embedding_text, clean_text


def test_clean_text_normalizes_whitespace() -> None:
    assert clean_text(" one\n\n two\tthree ") == "one two three"


def test_build_embedding_text_uses_available_article_fields() -> None:
    article = ArticleInput(
        article_id="a1",
        source="ReliefWeb",
        title="Flood response expands",
        url="https://example.test/a1",
        published_at="2026-05-30T10:00:00Z",
        summary="Regional response grows.",
        text="Emergency teams widened operations.",
    )

    text = build_embedding_text(article)

    assert "Title: Flood response expands" in text
    assert "Source: ReliefWeb" in text
    assert "Summary: Regional response grows." in text
    assert "Text: Emergency teams widened operations." in text
