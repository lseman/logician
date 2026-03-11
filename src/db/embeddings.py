from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

@contextlib.contextmanager
def _suppress_fd_output() -> Any:
    """Temporarily silence low-level stdout/stderr from noisy native loaders."""
    old_fd1 = os.dup(1)
    old_fd2 = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)
    os.close(null_fd)
    try:
        yield
    finally:
        os.dup2(old_fd1, 1)
        os.close(old_fd1)
        os.dup2(old_fd2, 2)
        os.close(old_fd2)


def _lazy_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer
    except Exception as e:
        raise ImportError(
            "sentence-transformers not installed or failed to import. "
            "Run: pip install sentence-transformers"
        ) from e


def _lazy_import_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        return CrossEncoder
    except Exception as e:
        raise ImportError(
            "CrossEncoder not available from sentence-transformers. "
            "Run: pip install sentence-transformers"
        ) from e


def _lazy_import_hnswlib():
    try:
        import hnswlib  # type: ignore

        return hnswlib
    except Exception as e:
        raise ImportError(
            "hnswlib not installed or failed to import. Run: pip install hnswlib"
        ) from e


def _lazy_import_usearch_index():
    try:
        from usearch.index import Index  # type: ignore

        return Index
    except Exception as e:
        raise ImportError(
            "usearch not installed or failed to import. Run: pip install usearch"
        ) from e


def _lazy_import_chromadb():
    try:
        import chromadb  # type: ignore

        return chromadb
    except Exception as e:
        raise ImportError(
            "chromadb not installed or failed to import. Run: pip install chromadb"
        ) from e


def _lazy_import_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def _lazy_import_bitsandbytes_config():
    try:
        from transformers import BitsAndBytesConfig  # type: ignore

        return BitsAndBytesConfig
    except Exception:
        return None


def _reset_hf_hub_client() -> None:
    try:
        import huggingface_hub  # type: ignore

        close_session = getattr(huggingface_hub, "close_session", None)
        if callable(close_session):
            close_session()
            return
    except Exception:
        pass

    try:
        from huggingface_hub.utils import _http  # type: ignore

        close_session = getattr(_http, "close_session", None)
        if callable(close_session):
            close_session()
    except Exception:
        pass


def _resolve_model_load_kwargs(
    *,
    quant_mode_env: str,
    force_cuda_env: str,
) -> tuple[dict[str, Any], str]:
    """
    Build optional kwargs for SentenceTransformer/CrossEncoder constructors.
    Quantization is opt-in via env vars:
      - quant_mode_env: off|8bit|4bit
      - force_cuda_env: 1/true/yes to force cuda device
    """
    kwargs: dict[str, Any] = {}
    notes: list[str] = []

    torch_mod = _lazy_import_torch()
    has_cuda = bool(torch_mod is not None and torch_mod.cuda.is_available())
    force_cuda = os.getenv(force_cuda_env, "0").strip().lower() in ("1", "true", "yes")
    device = "cuda" if (has_cuda or force_cuda) else "cpu"
    kwargs["device"] = device
    notes.append(f"device={device}")

    qmode = os.getenv(quant_mode_env, "off").strip().lower()
    if qmode in ("8bit", "4bit"):
        BitsAndBytesConfig = _lazy_import_bitsandbytes_config()
        if BitsAndBytesConfig is not None:
            try:
                if qmode == "8bit":
                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    bnb_kwargs: dict[str, Any] = {"load_in_4bit": True}
                    if torch_mod is not None and hasattr(torch_mod, "float16"):
                        bnb_kwargs["bnb_4bit_compute_dtype"] = torch_mod.float16
                    kwargs["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)
                notes.append(f"quant={qmode}")
            except Exception:
                notes.append(f"quant={qmode}_build_failed")
        else:
            notes.append(f"quant={qmode}_transformers_missing")
    else:
        notes.append("quant=off")

    return kwargs, ", ".join(notes)


def _hf_cache_root() -> Path:
    hf_home = os.getenv("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_model_cache_dir(model_name: str) -> Path:
    repo_id = str(model_name or "").strip().replace("/", "--")
    return _hf_cache_root() / f"models--{repo_id}"


def _hf_model_cached(model_name: str) -> bool:
    cache_dir = _hf_model_cache_dir(model_name)
    snapshots = cache_dir / "snapshots"
    if not snapshots.exists():
        return False
    try:
        return any(child.is_dir() for child in snapshots.iterdir())
    except Exception:
        return False


def _embedding_candidates(name: str) -> list[str]:
    parts = [p.strip() for p in str(name).split("|")]
    return [p for p in parts if p]


def _prepare_embedding_input(model_name: str, text: str, *, for_query: bool) -> str:
    lower = model_name.lower()
    normalized = str(text or "").strip()
    if any(
        token in lower
        for token in (
            "bge-m3",
            "snowflake-arctic-embed",
            "qwen",
            "e5-mistral",
        )
    ):
        prefix = "Instruct: " if for_query else ""
        return prefix + normalized
    if "nomic-embed-text" in lower:
        prefix = "search_query: " if for_query else "search_document: "
        return prefix + normalized
    if "e5" in lower:
        prefix = "query: " if for_query else "passage: "
        return prefix + normalized
    return normalized


def _stable_collection_name(base: str, embedding_model_name: str) -> str:
    primary = _embedding_candidates(embedding_model_name)
    model_key = primary[0] if primary else str(embedding_model_name)
    model_slug = model_key.replace("/", "_").replace("-", "_")
    digest = hashlib.sha256(model_key.encode("utf-8")).hexdigest()[:12]
    return f"{base}__{model_slug}__{digest}"


def _resolve_vector_collection_dir(
    *,
    root_path: str,
    collection_name: str,
    embedding_model_name: str,
    backend: str,
) -> Path:
    root = Path(root_path)
    stable_name = _stable_collection_name(collection_name, embedding_model_name)
    backend_norm = str(backend or "").strip().lower() or "usearch"

    preferred = root / backend_norm / stable_name
    if preferred.exists():
        return preferred

    legacy = root / stable_name
    if not legacy.exists():
        return preferred

    state_file = legacy / "state.json"
    if not state_file.exists():
        return legacy

    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return legacy

    saved_backend = str(state.get("backend", "") or "").strip().lower()
    if not saved_backend or saved_backend == backend_norm:
        return legacy
    return preferred


# ============================================================================
# Embedding + reranking wrappers
# ============================================================================
class _EmbeddingRuntime:
    _GLOBAL_ENCODERS: dict[str, Any] = {}
    _GLOBAL_LOCK = threading.RLock()

    def __init__(self, embedding_model_name: str, log: Any) -> None:
        self._model_name_raw = embedding_model_name
        self._log = log
        self._encoder: Any | None = None
        self._resolved_model_name: str | None = None
        self._lock = threading.RLock()

    @property
    def resolved_model_name(self) -> str:
        return self._resolved_model_name or self._model_name_raw

    def prefer_candidate(self, candidate: str) -> None:
        candidate_s = str(candidate or "").strip()
        if not candidate_s or self._encoder is not None:
            return
        candidates = _embedding_candidates(self._model_name_raw)
        if candidate_s not in candidates:
            return
        ordered = [candidate_s] + [item for item in candidates if item != candidate_s]
        self._model_name_raw = "|".join(ordered)

    def _ensure_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder

        SentenceTransformer = _lazy_import_sentence_transformers()
        last_err: Exception | None = None
        for candidate in _embedding_candidates(self._model_name_raw):
            with self._GLOBAL_LOCK:
                cached = self._GLOBAL_ENCODERS.get(candidate)
            if cached is not None:
                self._encoder = cached
                self._resolved_model_name = candidate
                self._log.info("Reusing cached embedding model: %s", candidate)
                return cached

            try:
                # Lock around load to avoid duplicate model initialization
                # when multiple DB runtimes start concurrently.
                with self._GLOBAL_LOCK:
                    cached2 = self._GLOBAL_ENCODERS.get(candidate)
                    if cached2 is not None:
                        self._encoder = cached2
                        self._resolved_model_name = candidate
                        self._log.info("Reusing cached embedding model: %s", candidate)
                        return cached2

                    self._log.info("Loading embedding model: %s", candidate)
                    kwargs, mode_note = _resolve_model_load_kwargs(
                        quant_mode_env="AGENT_EMBED_QUANT",
                        force_cuda_env="AGENT_FORCE_CUDA",
                    )
                    if _hf_model_cached(candidate):
                        kwargs["local_files_only"] = True
                    self._log.info("Embedding load kwargs: %s", mode_note)
                    try:
                        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                        try:
                            from transformers.utils import logging as _tf_logging

                            _tf_logging.set_verbosity_error()
                            disable_pb = getattr(_tf_logging, "disable_progress_bar", None)
                            if callable(disable_pb):
                                disable_pb()
                        except Exception:
                            pass
                        _reset_hf_hub_client()
                        with _suppress_fd_output():
                            try:
                                enc = SentenceTransformer(candidate, **kwargs)
                            except RuntimeError as exc:
                                if "client has been closed" not in str(exc).lower():
                                    raise
                                self._log.warning(
                                    "Hugging Face hub client was closed; resetting and retrying model load."
                                )
                                _reset_hf_hub_client()
                                enc = SentenceTransformer(candidate, **kwargs)
                    except TypeError:
                        # Some sentence-transformers versions may not accept quantization_config.
                        if "quantization_config" in kwargs:
                            self._log.warning(
                                "Embedding quantization kwargs unsupported; retrying without quantization."
                            )
                            kwargs = {
                                k: v
                                for k, v in kwargs.items()
                                if k != "quantization_config"
                            }
                        with _suppress_fd_output():
                            enc = SentenceTransformer(candidate, **kwargs)
                    self._GLOBAL_ENCODERS[candidate] = enc

                self._encoder = enc
                self._resolved_model_name = candidate
                return enc
            except Exception as e:
                last_err = e
                self._log.warning("Embedding model failed %s: %s", candidate, e)

        if last_err is not None:
            raise last_err
        raise ValueError("No embedding model candidate provided.")

    def dim(self) -> int:
        with self._lock:
            enc = self._ensure_encoder()
            sample = _prepare_embedding_input(
                self.resolved_model_name, "hello", for_query=False
            )
            vec = enc.encode([sample], normalize_embeddings=True)[0]
            return int(np.asarray(vec).shape[0])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            enc = self._ensure_encoder()
            prepared = [
                _prepare_embedding_input(self.resolved_model_name, t, for_query=False)
                for t in texts
            ]
            vecs = np.asarray(enc.encode(prepared, normalize_embeddings=True))
            if vecs.dtype != np.float32:
                vecs = vecs.astype(np.float32, copy=False)
            return vecs

    def embed_query(self, query: str) -> np.ndarray:
        with self._lock:
            enc = self._ensure_encoder()
            q = _prepare_embedding_input(
                self.resolved_model_name, query, for_query=True
            )
            vec = np.asarray(enc.encode([q], normalize_embeddings=True)[0])
            if vec.dtype != np.float32:
                vec = vec.astype(np.float32, copy=False)
            return vec


class _RerankerRuntime:
    _GLOBAL_MODELS: dict[str, Any] = {}
    _GLOBAL_LOCK = threading.RLock()

    def __init__(self, model_name: str, enabled: bool, log: Any) -> None:
        self.model_name = model_name
        self._model_name_raw = model_name
        self.enabled = bool(enabled)
        self._log = log
        self._model: Any | None = None
        self._lock = threading.RLock()

    def _ensure(self) -> Any | None:
        if not self.enabled:
            return None
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            CrossEncoder = _lazy_import_cross_encoder()
            last_err: Exception | None = None
            for candidate in _embedding_candidates(self._model_name_raw):
                with self._GLOBAL_LOCK:
                    cached = self._GLOBAL_MODELS.get(candidate)
                if cached is not None:
                    self._model = cached
                    self.model_name = candidate
                    self._log.info("Reusing cached reranker model: %s", candidate)
                    return cached

                self._log.info("Loading reranker model: %s", candidate)
                kwargs, mode_note = _resolve_model_load_kwargs(
                    quant_mode_env="AGENT_RERANK_QUANT",
                    force_cuda_env="AGENT_FORCE_CUDA",
                )
                if _hf_model_cached(candidate):
                    kwargs["local_files_only"] = True
                self._log.info("Reranker load kwargs: %s", mode_note)
                try:
                    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                    try:
                        from transformers.utils import logging as _tf_logging

                        _tf_logging.set_verbosity_error()
                        disable_pb = getattr(_tf_logging, "disable_progress_bar", None)
                        if callable(disable_pb):
                            disable_pb()
                    except Exception:
                        pass
                    with _suppress_fd_output():
                        model = CrossEncoder(candidate, **kwargs)
                except TypeError:
                    if "quantization_config" in kwargs:
                        self._log.warning(
                            "Reranker quantization kwargs unsupported; retrying without quantization."
                        )
                        kwargs = {
                            k: v for k, v in kwargs.items() if k != "quantization_config"
                        }
                    try:
                        with _suppress_fd_output():
                            model = CrossEncoder(candidate, **kwargs)
                    except Exception as exc:
                        last_err = exc
                        self._log.warning(
                            "Reranker model failed %s: %s", candidate, exc
                        )
                        continue
                except Exception as exc:
                    last_err = exc
                    self._log.warning("Reranker model failed %s: %s", candidate, exc)
                    continue

                with self._GLOBAL_LOCK:
                    self._GLOBAL_MODELS[candidate] = model
                self._model = model
                self.model_name = candidate
                return model

            if last_err is not None:
                raise last_err
            raise ValueError("No reranker model candidate provided.")

    def rerank(
        self,
        query: str,
        rows: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not rows:
            return rows
        model = self._ensure()
        if model is None:
            return rows[:top_k]
        pairs = [(query, str(r.get("content", ""))) for r in rows]
        scores = model.predict(pairs)
        ranked = [
            {**row, "rerank_score": float(score)}
            for score, row in sorted(
                zip(scores, rows), key=lambda item: item[0], reverse=True
            )
        ]
        return ranked[:top_k]


__all__ = [
    "_suppress_fd_output",
    "_lazy_import_sentence_transformers",
    "_lazy_import_cross_encoder",
    "_lazy_import_hnswlib",
    "_lazy_import_usearch_index",
    "_lazy_import_torch",
    "_lazy_import_bitsandbytes_config",
    "_resolve_model_load_kwargs",
    "_embedding_candidates",
    "_prepare_embedding_input",
    "_stable_collection_name",
    "_EmbeddingRuntime",
    "_RerankerRuntime",
]
