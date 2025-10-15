"""Vec2Text processing utilities.

This module restores the two-stage vec2text pipeline described in
``docs/how_to_use_jxe_and_ielab.md`` by wiring the hypothesiser (inversion
model) and iterative corrector together.  Both the JXE and IELab wrappers reuse
this module so the decoding behaviour matches the documented baseline.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator

LOGGER = logging.getLogger(__name__)

DEFAULT_TEACHER = "sentence-transformers/gtr-t5-base"
DEFAULT_LOCAL_TEACHER = Path("data/teacher_models/gtr-t5-base")


@dataclass
class Vec2TextConfig:
    """Configuration for the vec2text processor."""

    teacher_model: str = DEFAULT_TEACHER
    device: Optional[str] = None
    random_seed: Optional[int] = None
    debug: bool = False
    max_length: int = 128
    min_length: int = 1


def _resolve_teacher_path(model_hint: str | Path) -> str:
    candidate = Path(model_hint)
    if candidate.exists():
        return str(candidate.resolve())
    # Fall back to default local cache if present
    if DEFAULT_LOCAL_TEACHER.exists():
        return str(DEFAULT_LOCAL_TEACHER.resolve())
    return str(model_hint)


def _resolve_device(device_hint: Optional[str]) -> torch.device:
    if device_hint:
        hint = device_hint.lower()
    else:
        hint = "cpu"

    if hint == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if hint == "mps" and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)


class Vec2TextProcessor:
    """High level API that encapsulates teacher embeddings + vec2text inversion."""

    def __init__(self, config: Vec2TextConfig):
        self.config = config
        self.device = _resolve_device(config.device)
        _seed_everything(config.random_seed)
        self._prepare_environment()
        teacher_id = _resolve_teacher_path(config.teacher_model)

        try:
            from vec2text.api import load_pretrained_corrector, invert_embeddings
        except Exception as exc:  # pragma: no cover - dependency issue
            raise RuntimeError("vec2text is required for decoding") from exc
        self._load_pretrained_corrector = load_pretrained_corrector
        self._invert_embeddings = invert_embeddings

        if config.debug:
            LOGGER.setLevel(logging.DEBUG)
            LOGGER.debug("Vec2TextProcessor initialising with teacher=%s device=%s", teacher_id, self.device)

        # Ensure the orchestrator uses the vec2text-compatible encoder implementation
        os.environ.setdefault("VEC2TEXT_FORCE_CPU", "1")
        os.environ.setdefault("LNSP_EMBED_MODEL_DIR", teacher_id)

        try:
            self.orchestrator = IsolatedVecTextVectOrchestrator(
                steps=1,
                debug=config.debug,
                vec2text_backend="isolated",
            )
        except Exception as exc:  # pragma: no cover - dependency issue
            raise RuntimeError(f"Failed to initialise vec2text encoder orchestrator: {exc}") from exc

        # Load vec2text hypothesiser + corrector pair
        try:
            # Use the original JXM model that worked before
            self.corrector = self._load_pretrained_corrector("gtr-base")
        except Exception as exc:  # pragma: no cover - dependency issue
            raise RuntimeError(f"Failed to load vec2text corrector: {exc}") from exc

        # Align generation limits with configuration so invert_embeddings honours them.
        gen_kwargs = dict(getattr(self.corrector, "gen_kwargs", {}))
        gen_kwargs["min_length"] = self.config.min_length
        gen_kwargs["max_length"] = self.config.max_length
        self.corrector.gen_kwargs = gen_kwargs

        # Align global vec2text device helpers with our selection.
        try:
            import vec2text.api as vec2text_api  # type: ignore
            from vec2text.models import model_utils as vec_model_utils  # type: ignore

            vec_model_utils.device = self.device
            vec2text_api.device = self.device
        except Exception as exc:  # pragma: no cover - best effort
            if self.config.debug:
                LOGGER.debug("Failed to patch vec2text devices: %s", exc)

        # Ensure models live on the requested device
        self._move_corrector_to_device()
        self.corrector.inversion_trainer.model.eval()
        self.corrector.model.eval()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _prepare_environment(self) -> None:
        """Ensure vec2text uses a stable cache location."""
        default_cache = Path(".hf_cache")
        if default_cache.exists():
            os.environ.setdefault("HF_HOME", str(default_cache.resolve()))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(default_cache.resolve()))
        vec_cache = Path(os.environ.get("VEC2TEXT_CACHE", "app_cache/vec2text"))
        vec_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("VEC2TEXT_CACHE", str(vec_cache.resolve()))
        if self.device.type == "cpu":
            os.environ.setdefault("PYTORCH_MPS_DISABLE", "1")
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            try:
                if hasattr(torch.backends, "mps"):
                    torch.backends.mps.is_available = lambda: False  # type: ignore
            except Exception:  # pragma: no cover - best effort
                pass
        elif self.device.type == "mps":
            os.environ.pop("PYTORCH_MPS_DISABLE", None)

    def _move_corrector_to_device(self) -> None:
        # Inversion model (initial hypothesis generator)
        try:
            self.corrector.inversion_trainer.model.to(self.device)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to place inversion model on %s: %s", self.device, exc)
        # Corrector encoders (iterative refinement)
        try:
            self.corrector.model.to(self.device)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to place corrector model on %s: %s", self.device, exc)
        # The embedder tokenizer/encoder are owned by inversion_trainer; it already handles device dispatch

    def _prepare_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        embedding = embedding.to(self.device).float()
        embedding = F.normalize(embedding, dim=-1)
        return embedding

    def _run_inversion(
        self,
        embedding: torch.Tensor,
        *,
        num_steps: int | None,
        beam_width: int,
    ) -> str:
        with torch.no_grad():
            prepared = embedding
            if prepared.dim() == 1:
                prepared = prepared.unsqueeze(0)
            prepared = prepared.to(self.device)
            beam = max(0, beam_width)
            previous_setting = getattr(self.corrector, "return_best_hypothesis", False)
            self.corrector.return_best_hypothesis = beam > 0
            outputs = self._invert_embeddings(
                embeddings=prepared,
                corrector=self.corrector,
                num_steps=num_steps,
                sequence_beam_width=beam,
            )
            self.corrector.return_best_hypothesis = previous_setting
        text = outputs[0] if outputs else ""
        return text.strip()

    def _embed_text(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            vec = self.orchestrator.encode_texts([text])
        vec = vec.to(self.device).float()
        vec = F.normalize(vec, dim=-1)
        return vec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_vector_from_source(self, text: str, vector_source: str = "teacher") -> torch.Tensor:
        if vector_source != "teacher":
            raise ValueError(f"Unsupported vector source '{vector_source}'")
        with torch.no_grad():
            vec = self.orchestrator.encode_texts([text])
        return vec.squeeze(0).to(self.device)

    def iterative_vec2text_process(
        self,
        input_text: str,
        *,
        vector_source: str = "teacher",
        num_iterations: int = 4,
        beam_width: int = 1,
        target_embedding: Optional[torch.Tensor] = None,
    ) -> dict:
        """Run the full vec2text pipeline and return diagnostic information."""
        if target_embedding is None:
            target_embedding = self.get_vector_from_source(input_text, vector_source)
        target_embedding = self._prepare_embedding(target_embedding)
        initial_text = self._run_inversion(
            target_embedding,
            num_steps=None,
            beam_width=0,
        )
        final_text = self._run_inversion(
            target_embedding,
            num_steps=max(1, num_iterations),
            beam_width=beam_width,
        )

        # Measure cosine similarity between the target embedding and initial/final reconstructions
        try:
            initial_vec = self._embed_text(initial_text)
            final_vec = self._embed_text(final_text)
            target_vec = target_embedding
            initial_cosine = float(F.cosine_similarity(target_vec, initial_vec, dim=-1).item())
            final_cosine = float(F.cosine_similarity(target_vec, final_vec, dim=-1).item())
        except Exception as exc:  # pragma: no cover - diagnostics only
            LOGGER.debug("Cosine computation failed: %s", exc)
            initial_cosine = float("nan")
            final_cosine = float("nan")

        return {
            "initial_text": initial_text,
            "final_text": final_text,
            "initial_cosine": initial_cosine,
            "final_cosine": final_cosine,
            "num_iterations": int(max(1, num_iterations)),
            "beam_width": int(max(0, beam_width)),
        }

    def decode_embeddings(
        self,
        embeddings: Sequence[torch.Tensor] | np.ndarray,
        *,
        num_iterations: int = 4,
        beam_width: int = 1,
        prompts: Optional[Sequence[str]] = None,
    ) -> List[dict]:
        """Decode a batch of embeddings."""
        if isinstance(embeddings, np.ndarray):
            tensor = torch.from_numpy(embeddings)
        else:
            tensor = torch.stack([torch.as_tensor(e) for e in embeddings])

        tensor = tensor.to(self.device).float()
        results: List[dict] = []
        for idx, embedding in enumerate(tensor):
            prompt = prompts[idx] if prompts and idx < len(prompts) else " "
            info = self.iterative_vec2text_process(
                prompt,
                vector_source="teacher",
                num_iterations=num_iterations,
                beam_width=beam_width,
                target_embedding=embedding,
            )
            results.append(info)
        return results


def create_vec2text_processor(
    *,
    teacher_model_name: str = DEFAULT_TEACHER,
    device: Optional[str] = None,
    random_seed: Optional[int] = None,
    debug: bool = False,
) -> Vec2TextProcessor:
    """Factory helper used by wrappers and tests."""
    config = Vec2TextConfig(
        teacher_model=teacher_model_name,
        device=device,
        random_seed=random_seed,
        debug=debug,
    )
    return Vec2TextProcessor(config)
