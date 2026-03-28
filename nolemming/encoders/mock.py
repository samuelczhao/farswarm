"""Mock brain encoder for testing and development.

Produces content-aware synthetic neural responses: different stimuli
produce different activation patterns, simulating how real brains
process different content via different neural circuits.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nolemming.core.types import (
    FSAVERAGE5_VERTICES,
    NeuralResponse,
    Stimulus,
    StimulusType,
)
from nolemming.encoders.base import BrainEncoder

TIMESTEPS_TEXT = 60
TIMESTEPS_AUDIO_DEFAULT = 120
TIMESTEPS_VIDEO = 300
DEFAULT_SEED = 42

# Region boundaries aligned with BrainAtlas
REGION_SLICES: dict[str, slice] = {
    "visual_cortex": slice(0, 2000),
    "auditory_cortex": slice(2000, 3000),
    "motor_cortex": slice(3000, 4000),
    "prefrontal_cortex": slice(4000, 6000),
    "amygdala_proxy": slice(6000, 6500),
    "insula_proxy": slice(6500, 7000),
    "temporal_pole": slice(7000, 7500),
    "reward_circuit_proxy": slice(7500, 8000),
    "acc": slice(8000, 8500),
    "somatosensory": slice(8500, 9500),
    "parietal": slice(9500, 11000),
    "language_areas": slice(11000, 12500),
}

# Content keywords that trigger specific brain regions
CONTENT_TRIGGERS: dict[str, list[str]] = {
    "amygdala_proxy": [
        "fear", "crash", "risk", "warning", "alarming", "panic",
        "threat", "decline", "loss", "miss", "disappointing", "weak",
    ],
    "reward_circuit_proxy": [
        "beat", "record", "growth", "strong", "surge", "profit",
        "bullish", "upgrade", "buy", "opportunity", "gains", "boom",
    ],
    "prefrontal_cortex": [
        "analysis", "estimate", "guidance", "revenue", "margins",
        "strategy", "valuation", "forecast", "outlook", "capex",
    ],
    "temporal_pole": [
        "CEO", "Musk", "Zuckerberg", "Cook", "Pichai", "Huang",
        "investors", "community", "public", "social", "users",
    ],
    "insula_proxy": [
        "spending", "expensive", "overvalued", "bubble", "concern",
        "skeptical", "uncertainty", "caution", "doubt", "risky",
    ],
    "acc": [
        "but", "however", "despite", "although", "mixed", "contrast",
        "debate", "controversial", "question", "challenge",
    ],
}


class MockEncoder(BrainEncoder):
    """Content-aware mock encoder for testing.

    Produces different neural activation patterns for different stimuli
    by analyzing content keywords to determine which brain regions activate.
    """

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self._seed = seed

    @property
    def name(self) -> str:
        return "mock"

    def encode(self, stimulus: Stimulus) -> NeuralResponse:
        n_timesteps = _resolve_timesteps(stimulus)
        content_seed = _content_seed(stimulus, self._seed)
        rng = np.random.default_rng(content_seed)
        base = _generate_base_noise(rng, n_timesteps)
        _add_modality_structure(rng, base, stimulus.stimulus_type)
        _add_content_structure(rng, base, stimulus)
        _add_temporal_dynamics(base, n_timesteps)
        return NeuralResponse(activations=base.astype(np.float32))


def _content_seed(stimulus: Stimulus, base_seed: int) -> int:
    """Deterministic seed from stimulus content + base seed."""
    content = _read_stimulus_text(stimulus)
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
    return base_seed + int(h, 16) % 100_000


def _read_stimulus_text(stimulus: Stimulus) -> str:
    """Read text content from stimulus for content-aware encoding."""
    if stimulus.stimulus_type == StimulusType.TEXT:
        if stimulus.path.exists():
            return stimulus.path.read_text(encoding="utf-8")[:2000]
    return stimulus.path.stem


def _resolve_timesteps(stimulus: Stimulus) -> int:
    if stimulus.stimulus_type == StimulusType.TEXT:
        return TIMESTEPS_TEXT
    if stimulus.stimulus_type == StimulusType.VIDEO:
        return TIMESTEPS_VIDEO
    return _audio_timesteps(stimulus.path)


def _audio_timesteps(path: Path) -> int:
    try:
        import soundfile as sf  # type: ignore[import-untyped]
        info = sf.info(str(path))
        return max(1, int(info.duration))
    except (ImportError, RuntimeError, FileNotFoundError):
        return TIMESTEPS_AUDIO_DEFAULT


def _generate_base_noise(
    rng: np.random.Generator, n_timesteps: int,
) -> NDArray[np.float64]:
    return rng.normal(loc=0.0, scale=0.1, size=(n_timesteps, FSAVERAGE5_VERTICES))


def _add_modality_structure(
    rng: np.random.Generator,
    activations: NDArray[np.float64],
    stimulus_type: StimulusType,
) -> None:
    """Base modality boost (language for text, visual for video, etc.)."""
    gains = _modality_gains(stimulus_type)
    for region_name, gain in gains:
        _boost_region(rng, activations, region_name, gain)


def _add_content_structure(
    rng: np.random.Generator,
    activations: NDArray[np.float64],
    stimulus: Stimulus,
) -> None:
    """Content-specific boosts based on keywords in stimulus text."""
    text = _read_stimulus_text(stimulus).lower()
    for region, keywords in CONTENT_TRIGGERS.items():
        count = sum(1 for kw in keywords if kw in text)
        if count > 0:
            gain = min(0.3 + count * 0.15, 1.2)
            _boost_region(rng, activations, region, gain)


def _boost_region(
    rng: np.random.Generator,
    activations: NDArray[np.float64],
    region_name: str,
    gain: float,
) -> None:
    """Add gaussian boost to a specific brain region."""
    region = REGION_SLICES[region_name]
    n_verts = region.stop - region.start
    activations[:, region] += rng.normal(
        loc=gain, scale=gain * 0.2,
        size=(activations.shape[0], n_verts),
    )


def _modality_gains(
    stimulus_type: StimulusType,
) -> list[tuple[str, float]]:
    """Base modality-level region gains."""
    if stimulus_type == StimulusType.TEXT:
        return [("language_areas", 0.5), ("prefrontal_cortex", 0.3)]
    if stimulus_type == StimulusType.AUDIO:
        return [("auditory_cortex", 0.7), ("language_areas", 0.4)]
    return [
        ("visual_cortex", 0.8),
        ("auditory_cortex", 0.4),
        ("language_areas", 0.3),
    ]


def _add_temporal_dynamics(
    activations: NDArray[np.float64], n_timesteps: int,
) -> None:
    t = np.linspace(0, 2 * np.pi, n_timesteps)
    envelope = 0.5 + 0.5 * np.sin(t)
    activations *= envelope[:, np.newaxis]
