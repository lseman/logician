
# Plot Chroma message embeddings WITHOUT calling the agent
# Requirements (install if needed):
# pip install sentence-transformers chromadb umap-learn scikit-learn matplotlib
import json, numpy as np
import matplotlib.pyplot as plt


def _embed_texts(model_name: str, texts):
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(model_name)
    X = enc.encode(list(texts), normalize_embeddings=True)
    return X.astype(np.float32)


def _latest_session_id(db_path: str, chroma_path: str) -> str:
    # Fallback to SQL
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT session_id, MAX(timestamp) FROM messages GROUP BY session_id ORDER BY MAX(timestamp) DESC LIMIT 1"
        ).fetchone()
    if not row:
        raise RuntimeError("No sessions found in DB.")
    return row[0]


def _infer_semantic_label(text: str, role: str, name: str) -> str:
    t = f"{role} {name} {text}".lower()
    KWS = {
        "data cleaning": [
            "regularize",
            "missing",
            "impute",
            "fill",
            "hampel",
            "nan",
            "outlier",
            "clean",
        ],
        "transform": [
            "detrend",
            "difference",
            "box-cox",
            "scale",
            "normalize",
            "stl",
            "log ",
        ],
        "diagnostics": [
            "acf",
            "pacf",
            "diagnostic",
            "histogram",
            "decomposition",
            "seasonal strength",
        ],
        "forecasting": [
            "forecast",
            "horizon",
            "neuralforecast",
            "patchtst",
            "nhits",
            "sarima",
            "holt",
        ],
        "visualization": ["plot", "figure", "chart", "display", "show"],
        "io": ["load_csv", "set_numpy", "create_sample", "dataset", "save"],
        "agent/tooling": [
            "tool_call",
            "schema",
            "trace",
            "memory",
            "faiss",
            "semantic",
            "retrieve",
        ],
        "anomaly": ["anomaly", "changepoint", "pelt", "isolation forest"],
    }
    for label, kws in KWS.items():
        if any(k in t for k in kws):
            return label
    return "other"


def _reduce_2d(
    X: np.ndarray,
    method="umap",
    n_neighbors=15,
    min_dist=0.10,
    perplexity=30.0,
    random_state=42,
):
    method = (method or "umap").lower()
    if method == "umap":
        try:
            import umap
        except Exception as e:
            raise RuntimeError(
                "UMAP not installed. Try `pip install umap-learn` or use method='tsne'."
            ) from e
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=random_state,
            metric="cosine",
        )
        Z = reducer.fit_transform(X)
        return Z, "UMAP"
    else:
        from sklearn.manifold import TSNE

        # t-SNE perplexity must be < n_samples; auto-clamp safely
        pk = max(5.0, min(perplexity, max(5.0, (len(X) - 1) / 3)))
        tsne = TSNE(
            n_components=2,
            perplexity=pk,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
            metric="cosine",
        )
        Z = tsne.fit_transform(X)
        return Z, f"t-SNE (perp={pk:.0f})"


def plot_session_embedding_map(
    db_path: str = "agent_sessions.db",
    chroma_path: str = "message_history.chroma",
    session_id: str | None = None,
    method: str = "umap",  # "umap" | "tsne"
    label_mode: str = "keyword",  # "keyword" | "role" | "tool"
    embedding_model: str = "all-MiniLM-L6-v2",
    n_neighbors: int = 15,
    min_dist: float = 0.10,
    perplexity: float = 30.0,
    annotate_when_leq: int = 40,
    save_path: str | None = None,
):
    """
    Reads messages for a session from Chroma (embeddings + metadata), projects to 2D, and plots a single scatter figure.
    Fallback to re-embed if needed.
    """
    import chromadb

    # 1) Resolve session
    sid = session_id or _latest_session_id(db_path, chroma_path)
    # 2) Load from Chroma (exact get by metadata)
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(name="messages")
    results = collection.get(
        where={"session_id": sid}, include=["metadatas", "documents", "embeddings"]
    )
    if not results["documents"]:
        raise RuntimeError(f"No messages for session_id={sid}")
    texts = results["documents"]
    metas = results["metadatas"]
    # 3) Embeddings: Use stored or re-compute
    if results["embeddings"] and results["embeddings"][0]:
        X = np.array(results["embeddings"])
    else:
        # Fallback re-embed
        X = _embed_texts(embedding_model, texts)
    roles = [meta["role"] for meta in metas]
    names = [meta.get("name", "") for meta in metas]
    # 4) Labels
    if label_mode == "role":
        labels = roles
    elif label_mode == "tool":
        tool_ids = [meta.get("tool_call_id", "") for meta in metas]
        labels = [
            n if n else ("TOOL" if tid else role)
            for n, tid, role in zip(names, tool_ids, roles)
        ]
    else:  # "keyword"
        labels = [
            _infer_semantic_label(t, r, n) for t, r, n in zip(texts, roles, names)
        ]
    # 5) Reduce to 2D
    Z, label = _reduce_2d(
        X,
        method=method,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        perplexity=perplexity,
    )
    # 6) Plot (one figure, default colors, no seaborn)
    cats = sorted(set(labels))
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(cats)))  # Better colors
    for i, cat in enumerate(cats):
        idx = [j for j, c in enumerate(labels) if c == cat]
        ax.scatter(Z[idx, 0], Z[idx, 1], label=cat, alpha=0.85, s=28, color=colors[i])
    # annotate lightly if small
    if len(texts) <= annotate_when_leq:
        for i, (x, y) in enumerate(Z):
            tag = names[i] if names[i] else roles[i]
            ax.annotate(
                tag[:24],
                (x, y),
                fontsize=7,
                alpha=0.7,
                xytext=(5, 5),
                textcoords="offset points",
            )
    ax.set_title(
        f"{label} Projection of {len(texts)} Session Messages — Colored by {label_mode}"
    )
    ax.set_xlabel("Dim-1")
    ax.set_ylabel("Dim-2")
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.2)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
    # Optional: Silhouette score
    try:
        from sklearn.metrics import silhouette_score

        if len(set(labels)) > 1 and len(Z) >= 3:
            lut = {c: i for i, c in enumerate(cats)}
            y = np.array([lut[c] for c in labels])
            sil = float(silhouette_score(Z, y, metric="euclidean"))
            print(f"Silhouette score: {sil:.3f} (higher = better clustering)")
    except Exception as e:
        print(f"Silhouette computation failed: {e}")
    return {
        "session_id": sid,
        "n_points": len(texts),
        "method": label,
        "label_mode": label_mode,
        "saved": bool(save_path),
    }
