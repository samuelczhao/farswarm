"""Clustering a neural population into behavioral archetypes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from nolemming.core.types import NeuralArchetype
from nolemming.mapping.brain_atlas import BrainAtlas

REGION_BEHAVIOR: dict[str, str] = {
    "amygdala_proxy": "Responds to threats with urgency, likely to share alarming content",
    "reward_circuit_proxy": "Optimistic, attracted to positive narratives, early adopter of trends",
    "prefrontal_cortex": "Analytical, deliberative, prefers data-driven discussion",
    "temporal_pole": "Socially attuned, influenced by group dynamics",
    "insula_proxy": "Risk-averse, skeptical, quick to disengage from uncertain information",
    "acc": "Contrarian, notices inconsistencies, likely to challenge narratives",
    "visual_cortex": "Visually oriented, responds strongly to imagery and aesthetics",
    "auditory_cortex": "Attuned to tone and vocal cues, sensitive to rhetoric",
    "motor_cortex": "Action-oriented, inclined to participate and engage physically",
    "somatosensory": "Empathetic, bodily aware, responds to visceral content",
    "parietal": "Attention-driven, integrates multiple information streams",
    "language_areas": "Verbal, articulate, engages deeply with written arguments",
}

REGION_LABEL_PREFIX: dict[str, str] = {
    "amygdala_proxy": "fear-dominant",
    "reward_circuit_proxy": "reward-seeking",
    "prefrontal_cortex": "analytical",
    "temporal_pole": "social-attuned",
    "insula_proxy": "risk-averse",
    "acc": "contrarian",
    "visual_cortex": "visual-driven",
    "auditory_cortex": "auditory-tuned",
    "motor_cortex": "action-oriented",
    "somatosensory": "empathetic",
    "parietal": "attention-focused",
    "language_areas": "verbal-analytical",
}


class ArchetypeClusterer:
    """Groups a neural population into distinct behavioral archetypes."""

    def __init__(self, n_archetypes: int = 8) -> None:
        self.n_archetypes = n_archetypes

    def cluster_population(
        self,
        population: NDArray[np.float32],
        atlas: BrainAtlas,
    ) -> list[NeuralArchetype]:
        """Cluster a population matrix (n_individuals, n_regions) into archetypes.

        This is the primary method. Each row is one individual's ROI profile.
        """
        effective_k = min(self.n_archetypes, population.shape[0])
        kmeans = KMeans(n_clusters=effective_k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(population)
        region_names = list(atlas.REGIONS.keys())
        return [
            self._build_from_population(
                i, kmeans, labels, population, region_names,
            )
            for i in range(effective_k)
        ]

    def cluster(
        self,
        compressed_timesteps: NDArray[np.float32],
        atlas: BrainAtlas,
        original_activations: NDArray[np.float32],
    ) -> list[NeuralArchetype]:
        """Legacy method: cluster compressed timesteps.

        Prefer cluster_population() for better archetype diversity.
        Falls back to population-based clustering using the original
        activations' ROI profiles.
        """
        from nolemming.core.types import NeuralResponse
        from nolemming.mapping.population import generate_population_responses

        response = NeuralResponse(activations=original_activations)
        population = generate_population_responses(response, atlas)
        return self.cluster_population(population, atlas)

    def _build_from_population(
        self,
        cluster_idx: int,
        kmeans: KMeans,
        labels: NDArray[np.int32],
        population: NDArray[np.float32],
        region_names: list[str],
    ) -> NeuralArchetype:
        """Build archetype from population cluster."""
        mask = labels == cluster_idx
        pop_fraction = float(np.sum(mask)) / len(labels)
        centroid = kmeans.cluster_centers_[cluster_idx]

        dominant = _dominant_regions_from_centroid(centroid, region_names)
        return NeuralArchetype(
            archetype_id=cluster_idx,
            centroid=centroid.astype(np.float32),
            label=_make_label(dominant),
            description=_make_description(dominant),
            population_fraction=pop_fraction,
            dominant_regions=dominant,
        )


def _dominant_regions_from_centroid(
    centroid: NDArray[np.float32],
    region_names: list[str],
    top_k: int = 3,
) -> list[str]:
    """Find the top-k regions by centroid value."""
    indices = np.argsort(centroid)[::-1][:top_k]
    return [region_names[i] for i in indices if i < len(region_names)]


def _make_label(dominant_regions: list[str]) -> str:
    if not dominant_regions:
        return "mixed-profile"
    top = dominant_regions[0]
    return REGION_LABEL_PREFIX.get(top, top.replace("_", "-"))


def _make_description(dominant_regions: list[str]) -> str:
    traits = [
        REGION_BEHAVIOR[r] for r in dominant_regions if r in REGION_BEHAVIOR
    ]
    if not traits:
        return "Mixed neural profile with no single dominant behavioral tendency."
    return ". ".join(traits) + "."
