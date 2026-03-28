"""Generate a synthetic population of diverse neural responses.

Given a stimulus neural signature (from the brain encoder), generate
N individuals who each process the stimulus differently based on their
unique neural wiring. Some brains weight fear circuits more, others
reward circuits, etc.

This is the key bridge between a single brain encoder output and a
diverse agent population.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nolemming.core.types import NeuralResponse
from nolemming.mapping.brain_atlas import BrainAtlas

N_INDIVIDUALS_DEFAULT = 200


def generate_population_responses(
    stimulus_response: NeuralResponse,
    atlas: BrainAtlas,
    n_individuals: int = N_INDIVIDUALS_DEFAULT,
    seed: int = 42,
) -> NDArray[np.float32]:
    """Generate diverse individual neural profiles from a stimulus signature.

    Each individual has the same base stimulus response, but with region-specific
    amplification factors drawn from a log-normal distribution. This simulates
    inter-individual neural variability.

    Returns:
        shape (n_individuals, n_regions) — each row is one individual's ROI profile.
    """
    rng = np.random.default_rng(seed)
    base_rois = _extract_stimulus_rois(stimulus_response, atlas)
    n_regions = len(base_rois)

    individual_factors = _generate_individual_factors(
        rng, n_individuals, n_regions,
    )
    population = base_rois[np.newaxis, :] * individual_factors
    return population.astype(np.float32)


def _extract_stimulus_rois(
    response: NeuralResponse, atlas: BrainAtlas,
) -> NDArray[np.float64]:
    """Extract per-region mean activations from stimulus response."""
    rois = atlas.extract_all_rois(response.mean_activation())
    return np.array(list(rois.values()), dtype=np.float64)


def _generate_individual_factors(
    rng: np.random.Generator,
    n_individuals: int,
    n_regions: int,
) -> NDArray[np.float64]:
    """Generate per-region amplification factors for each individual.

    Uses log-normal distribution: most individuals are near 1.0 (average),
    but some have significantly amplified or dampened region responses.
    This matches real inter-individual variability in neural processing.
    """
    return rng.lognormal(mean=0.0, sigma=0.5, size=(n_individuals, n_regions))
