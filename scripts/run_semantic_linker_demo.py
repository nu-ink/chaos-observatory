from __future__ import annotations

from ml.ml_embeddings import ArticleInput
from ml.ml_semantic_linker import SemanticLinker
from ml.ml_similarity_thresholds import SemanticSimilarityConfig


def main() -> None:
    config = SemanticSimilarityConfig()
    linker = SemanticLinker(config)

    articles = [
        ArticleInput(
            article_id="bbc_world_20260531_001",
            source="BBC World",
            title="Oil prices rise after shipping disruption",
            url="https://example.com/bbc/oil-prices-rise",
            published_at="2026-05-31T08:30:00Z",
            summary="Markets react to supply-chain concerns in a key shipping corridor.",
            text="Oil prices rose after renewed disruption in a major shipping corridor. Insurers and traders responded quickly.",
        ),
        ArticleInput(
            article_id="reuters_world_20260531_002",
            source="Reuters",
            title="Shipping companies reroute vessels after Red Sea incidents",
            url="https://example.com/reuters/red-sea-reroute",
            published_at="2026-05-31T09:00:00Z",
            summary="Carriers alter routes after attacks raise risk in the region.",
            text="Major shipping firms rerouted vessels after incidents in the Red Sea increased risk for commercial traffic.",
        ),
        ArticleInput(
            article_id="ap_world_20260531_003",
            source="Associated Press",
            title="Insurance rates climb for cargo vessels",
            url="https://example.com/ap/cargo-insurance",
            published_at="2026-05-31T09:30:00Z",
            summary="War-risk premiums rise after maritime threats intensify.",
            text="Insurance costs climbed for cargo ships operating near a tense maritime route after a series of attacks.",
        ),
    ]

    for article in articles:
        matches = linker.process_article(article)
        print(f"\nArticle: {article.article_id}")
        if not matches:
            print("  No prior matches found.")
            continue

        for idx, match in enumerate(matches, start=1):
            print(
                f"  {idx}. matched_article_id={match.matched_article_id} "
                f"score={match.similarity_score:.4f} "
                f"relationship={match.relationship_type}"
            )

    review_csv = linker.export_review_csv()
    print(f"\nReview CSV written to: {review_csv}")


if __name__ == "__main__":
    main()
