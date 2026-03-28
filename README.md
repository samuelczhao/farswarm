# Farswarm

**Brain-encoded swarm social simulation engine.**

Feed any stimulus (video, audio, text) into Farswarm. It predicts how human brains would process it, generates thousands of neurally-grounded agents, and simulates their collective social behavior to produce prediction reports.

The first project built on [Meta TRIBE v2](https://github.com/facebookresearch/tribev2) — combining brain encoding with multi-agent social simulation.

## How It Works

```
Stimulus (earnings call, news article, ad)
    |
    v
Brain Encoder (TRIBE v2 / pluggable)
    |
    v
20,484 cortical vertex activations
    |
    +---> PCA Compression (50-100 dims)
    |         |
    |         v
    |     Neural Archetypes (K-means clustering)
    |     "fear-dominant", "reward-seeking", "analytical", etc.
    |
    +---> Engagement Template
              (archetype x content_similarity -> engagement_prob)
                  |
                  v
          Agent Population (1000+ agents)
          Each grounded in a neural archetype
                  |
                  v
          Social Simulation (Twitter/Reddit-like)
          Agents post, share, debate, form coalitions
                  |
                  v
          Prediction Report
          Sentiment trajectory, dominant narratives,
          archetype dynamics, quantitative signals
```

## Why Neural Grounding Matters

Standard multi-agent simulations generate agent personalities from LLM heuristics. Farswarm generates them from **predicted brain activation patterns** — how a population of real brains would actually process the stimulus.

This matters because different stimuli activate different neural circuits:
- An alarming earnings miss activates **amygdala** (fear) -> more fear-driven agents -> faster panic spread
- A breakthrough product launch activates **reward circuits** -> more optimistic agents -> sustained positive engagement
- A complex policy document activates **prefrontal cortex** -> more analytical agents -> slower, deliberative discourse

The simulation captures these dynamics because agent behavior is grounded in the stimulus's neural signature, not generic personality templates.

## Architecture: Why Hybrid (Architecture C)

We evaluated three architectures and chose the one that maximizes signal preservation:

| Architecture | Signal Preserved | Compute Cost | Feasibility |
|---|---|---|---|
| A: Neural personality init only | ~3% (Big Five bottleneck destroys signal) | Low | High |
| B: Continuous neural scoring | ~100% theoretical, unreliable in practice | Prohibitive (1M forward passes) | Low |
| **C: Hybrid (neural init + engagement templates)** | **~15-25%** | **Low (one-time)** | **High** |

Architecture C runs the brain encoder **once** on the original stimulus (where it's reliable), skips the lossy Big Five personality mapping, and uses PCA + clustering to create **neural archetypes** directly from the activation data. A pre-computed engagement template modulates agent behavior throughout the simulation.

## Quick Start

```bash
pip install farswarm

# Run with mock encoder (no GPU needed)
farswarm run earnings_call.txt --agents 500 --encoder mock

# Run with TRIBE v2 brain encoder
pip install farswarm[tribe]
farswarm run earnings_call.mp4 --agents 1000 --encoder tribe_v2

# Use any LLM backend (Ollama, vLLM, Together, etc.)
farswarm run stimulus.txt --model llama3 --base-url http://localhost:11434/v1

# A/B test content variants
farswarm compare variant_a.txt variant_b.txt --agents 1000

# Run benchmark
farswarm benchmark aapl_q4_2025 --data-dir ./benchmarks/data
```

### Python API

```python
import asyncio
from farswarm.core.pipeline import FarswarmPipeline
from farswarm.core.llm import OpenAICompatibleBackend

# Works with any OpenAI-compatible API
llm = OpenAICompatibleBackend(
    model="llama3",
    base_url="http://localhost:11434/v1",  # Ollama
)

pipeline = FarswarmPipeline(llm=llm)
result = asyncio.run(pipeline.run("earnings_call.mp4", n_agents=1000))

report = result["analysis"]["report"]
print(report.to_markdown())
```

## Plug-and-Play Design

Every component is swappable:

### Brain Encoders
```python
from farswarm.encoders.registry import encoder_registry

# Built-in
encoder_registry.get("mock")      # Synthetic responses for testing
encoder_registry.get("tribe_v2")  # Meta TRIBE v2

# Register your own
from farswarm.encoders.base import BrainEncoder
class MyEncoder(BrainEncoder):
    def encode(self, stimulus):
        ...
    def name(self):
        return "my_encoder"

encoder_registry.register("my_encoder", MyEncoder)
```

### LLM Backends
```python
from farswarm.core.llm import OpenAICompatibleBackend

# OpenAI
llm = OpenAICompatibleBackend(model="gpt-4o")

# Ollama (local)
llm = OpenAICompatibleBackend(model="llama3", base_url="http://localhost:11434/v1")

# Together AI
llm = OpenAICompatibleBackend(model="meta-llama/Llama-3-70b", base_url="https://api.together.xyz/v1", api_key="...")

# vLLM (self-hosted)
llm = OpenAICompatibleBackend(model="my-model", base_url="http://localhost:8000/v1")

# Any OpenAI SDK-compatible endpoint
llm = OpenAICompatibleBackend(model="...", base_url="...", api_key="...")
```

### Environment Variables
```bash
export FARSWARM_LLM_MODEL=llama3
export FARSWARM_LLM_BASE_URL=http://localhost:11434/v1
export FARSWARM_ENCODER_NAME=mock
```

## Use Cases

### Market Sentiment Prediction
Feed an earnings call, predict the 72-hour social sentiment trajectory.
```bash
farswarm run earnings_call.mp4 --agents 1000 --rounds 72
```

### Content A/B Testing
Compare how different content variants spread through a neurally-diverse population.
```bash
farswarm compare headline_a.txt headline_b.txt --agents 2000
```

### Misinformation Spread Modeling
Simulate how misinformation propagates, identify which neural archetypes are most susceptible.
```bash
farswarm run misinfo_article.txt --agents 5000 --rounds 168
```

### Regime Change Detection
Track shifts in population-level neural response distributions across sequential stimuli.
```python
from farswarm.core.pipeline import FarswarmPipeline

pipeline = FarswarmPipeline()
for stimulus in sequential_stimuli:
    response = pipeline.encode_stimulus(stimulus)
    archetypes, _ = pipeline.build_archetypes(response)
    # Monitor archetype distribution shifts
```

## Benchmarking

Farswarm ships with a first-class benchmarking framework. Transparency about what works and what doesn't.

```bash
# 3-way comparison: neural vs vanilla vs random
farswarm benchmark aapl_q4_2025

# Results include:
# - Sentiment trajectory correlation (Pearson r)
# - Engagement distribution divergence (KL)
# - Narrative overlap (Jaccard similarity)
```

## Project Structure

```
farswarm/
├── core/           # Pipeline, types, LLM abstraction
├── encoders/       # Brain encoder plugins (mock, TRIBE v2, custom)
├── mapping/        # PCA compression, archetype clustering, engagement templates
├── agents/         # Agent factory, profile generation
├── simulation/     # OASIS-based social simulation engine
├── analysis/       # Sentiment, networks, signals, reports
├── benchmark/      # 3-way comparison framework
├── web/            # FastAPI server
└── cli.py          # CLI entry point
```

## Requirements

- Python >= 3.11
- Any OpenAI SDK-compatible LLM provider
- Optional: TRIBE v2 (requires GPU, CC BY-NC license)
- Optional: OASIS/CAMEL-AI (for full social simulation)

## License

MIT (Farswarm code). Brain encoder adapters carry their own licenses (TRIBE v2 is CC BY-NC).
