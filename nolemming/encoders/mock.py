"""Mock brain encoder for testing and development."""

from __future__ import annotations

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

# Region boundaries aligned with BrainAtlas for consistent ROI extraction
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


class MockEncoder(BrainEncoder):
    """Generates reproducible synthetic neural responses for testing."""

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self._seed = seed

    @property
    def name(self) -> str:
        return "mock"

    def encode(self, stimulus: Stimulus) -> NeuralResponse:
        n_timesteps = _resolve_timesteps(stimulus)
        rng = np.random.default_rng(self._seed)
        base = _generate_base_noise(rng, n_timesteps)
        _add_region_structure(rng, base, stimulus.stimulus_type)
        _add_temporal_dynamics(base, n_timesteps)
        return NeuralResponse(activations=base.astype(np.float32))


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


def _add_region_structure(
    rng: np.random.Generator,
    activations: NDArray[np.float64],
    stimulus_type: StimulusType,
) -> None:
    """Boost brain regions based on stimulus modality."""
    gains = _region_gains_for(stimulus_type)
    for region_name, gain in gains:
        region = REGION_SLICES[region_name]
        n_verts = region.stop - region.start
        activations[:, region] += rng.normal(
            loc=gain, scale=gain * 0.2,
            size=(activations.shape[0], n_verts),
        )


def _region_gains_for(
    stimulus_type: StimulusType,
) -> list[tuple[str, float]]:
    """Map stimulus type to (region_name, activation gain) pairs."""
    if stimulus_type == StimulusType.TEXT:
        return [
            ("language_areas", 0.8),
            ("prefrontal_cortex", 0.6),
            ("auditory_cortex", 0.3),
            ("temporal_pole", 0.4),
        ]
    if stimulus_type == StimulusType.AUDIO:
        return [
            ("auditory_cortex", 0.9),
            ("language_areas", 0.5),
            ("temporal_pole", 0.3),
        ]
    return [
        ("visual_cortex", 1.0),
        ("auditory_cortex", 0.6),
        ("language_areas", 0.4),
        ("motor_cortex", 0.3),
    ]


def _add_temporal_dynamics(
    activations: NDArray[np.float64], n_timesteps: int,
) -> None:
    t = np.linspace(0, 2 * np.pi, n_timesteps)
    envelope = 0.5 + 0.5 * np.sin(t)
    activations *= envelope[:, np.newaxis]
