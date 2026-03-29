# NoLemming

**Don't follow the herd. Predict it.**

NoLemming simulates how crowds react to information — before they actually react. Feed it any stimulus (earnings call, news article, product launch), and it generates a population of cognitively diverse agents who debate, share, and form opinions on simulated social media.

The key insight: not all brains process the same information the same way. NoLemming clusters agents into **neural archetypes** — fear-dominant, reward-seeking, analytical, contrarian, etc. — based on how different brain regions would respond to the stimulus. The result is a simulation where different cognitive types interact, producing emergent sentiment dynamics that mirror real social media behavior.

Optionally powered by [Meta TRIBE v2](https://github.com/facebookresearch/tribev2) for real brain encoding. Works with any LLM backend.

---

## How It Works

```
Stimulus (text / audio / video)
        |
        v
  Brain Encoder (TRIBE v2 / mock / custom)
        |
        v
  Neural activation pattern
        |
   -----+-----
   |         |
   v         v
  Population    Engagement
  Clustering    Template
   |              |
   v              v
  Neural Archetypes + Engagement Map
  ("fear-dominant", "reward-seeking", "analytical"...)
        |
        v
  Agent Population (100-1000+)
  Each grounded in a neural archetype
        |
        v
  Social Simulation (LLM-powered or template-based)
  Agents post, debate, form coalitions
        |
        v
  Analysis + Visualization
  Sentiment trajectory | Archetype dynamics | Network graph
```

**What makes this different from vanilla multi-agent simulation:** Agent behavior is parameterized by cognitive archetype, not random personality. A fear-dominant agent and a reward-seeking agent will react to the same earnings call in fundamentally different ways — and the population mix determines the emergent social dynamics.

---

## Quick Start

```bash
pip install nolemming

# Interactive demo (no API keys needed)
nolemming demo

# Run on your own stimulus
nolemming run earnings_call.txt --agents 500

# Use any LLM for higher-quality simulation
nolemming run stimulus.txt --model llama3 --base-url http://localhost:11434/v1

# A/B test content variants
nolemming compare variant_a.txt variant_b.txt --agents 1000
```

### Python API

```python
import asyncio
from nolemming import NoLemmingPipeline, OpenAICompatibleBackend

# Works with any OpenAI-compatible API (Ollama, vLLM, Together, Groq...)
llm = OpenAICompatibleBackend(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key="your-groq-key",
)

pipeline = NoLemmingPipeline(llm=llm)
result = asyncio.run(pipeline.run("earnings_call.txt", n_agents=500))

report = result["analysis"]["report"]
print(report.to_markdown())
```

---

## Neural Archetypes

Different stimuli activate different brain regions, producing different population compositions:

| Stimulus Type | Dominant Archetypes | Predicted Dynamic |
|---|---|---|
| Alarming earnings miss | Fear-dominant, risk-averse | Fast negative cascade |
| Strong earnings beat | Reward-seeking, analytical | Sustained positive engagement |
| Controversial announcement | Contrarian, social-attuned | Polarized debate, high volatility |
| Complex policy document | Analytical, verbal-analytical | Slow, deliberative discourse |

These aren't random labels — they're derived from clustering predicted neural activation patterns across brain regions (amygdala, reward circuits, prefrontal cortex, etc.).

---

## Brain Encoders

NoLemming supports pluggable brain encoders:

| Encoder | What it does | Requirements |
|---|---|---|
| `mock` (default) | Content-aware synthetic activations based on stimulus keywords | None |
| `tribe_v2` | Real brain encoding via Meta TRIBE v2 | GPU + `pip install nolemming[tribe]` |
| `precomputed` | Load pre-encoded .npy files | None (use with Colab notebook) |
| Custom | Implement `BrainEncoder` interface | Your encoder |

> **Note:** The default `mock` encoder generates synthetic neural data using keyword heuristics — it approximates but does not replicate real brain encoding. For research-grade results, use TRIBE v2 or precomputed responses from real brain models.

```python
from nolemming.encoders.registry import encoder_registry

# Register your own encoder
from nolemming.encoders.base import BrainEncoder
class MyEncoder(BrainEncoder):
    def encode(self, stimulus): ...
    @property
    def name(self): return "my_encoder"

encoder_registry.register("my_encoder", MyEncoder)
```

---

## LLM Backends

Any OpenAI SDK-compatible API works. Without an LLM, NoLemming uses archetype-aware post templates (still produces meaningful results, just less varied).

```python
from nolemming.core.llm import OpenAICompatibleBackend

# Groq (free, fast)
OpenAICompatibleBackend(model="llama-3.3-70b-versatile", base_url="https://api.groq.com/openai/v1")

# Ollama (local, private)
OpenAICompatibleBackend(model="qwen2.5:3b", base_url="http://localhost:11434/v1")

# OpenAI
OpenAICompatibleBackend(model="gpt-4o")
```

```bash
# Or via environment variables
export NOLEMMING_LLM_MODEL=llama-3.3-70b-versatile
export NOLEMMING_LLM_BASE_URL=https://api.groq.com/openai/v1
export NOLEMMING_LLM_API_KEY=your-key
```

---

## Visualization

`nolemming demo` generates an interactive neural network visualization — agents as neurons clustered in brain regions, with connections pulsing during interactions.

---

## Benchmark

3-way comparison across 5 Q4 2025 earnings calls (AAPL, TSLA, NVDA, META, GOOGL). Sentiment correlation with actual post-earnings social media dynamics:

| Condition | Description | Wins |
|-----------|-------------|------|
| **Neural** | **Archetype-grounded agents with engagement templates** | **3/5** |
| Vanilla | Same archetypes, uniform activity (no engagement modulation) | 1/5 |
| Random | Shuffled archetype assignments | 1/5 |

Run with Groq Llama 3.3 70B + mock encoder. Expanding to 20+ events with real TRIBE v2 brain encoding.

> **Limitations:** 5 events is not statistically significant. Ground truth sentiment is estimated. Results will improve with real brain encoding and more events.

```bash
python scripts/run_phase0.py
```

---

## Architecture

NoLemming uses **Architecture C (Hybrid)** — the brain encoder runs once on the original stimulus, then a population of 200 synthetic individuals with inter-individual neural variability is generated and clustered into archetypes. An engagement template modulates agent activation throughout the simulation based on archetype-content alignment.

This avoids the two failure modes: (A) compressing brain data to Big Five traits (destroys 95% of signal) and (B) running the encoder per-interaction (computationally infeasible).

---

## Project Structure

```
nolemming/
├── core/           # Pipeline, types, LLM abstraction
├── encoders/       # Brain encoder plugins (mock, TRIBE v2, precomputed, custom)
├── mapping/        # Population generation, archetype clustering, engagement templates
├── agents/         # Agent factory, profile generation with archetype templates
├── simulation/     # Social simulation engine (OASIS + template/LLM fallback)
├── analysis/       # Sentiment, networks, signals, reports
├── benchmark/      # 3-way comparison framework + 5 earnings call ground truth
├── viz/            # Interactive neural network visualization
├── web/            # FastAPI server
└── cli.py          # CLI (demo, run, compare, benchmark)
```

## Requirements

- Python >= 3.11
- Optional: LLM API (Groq free tier, Ollama, OpenAI, etc.)
- Optional: TRIBE v2 for real brain encoding (GPU + `pip install nolemming[tribe]`)

## Runtime Estimates

| Config | Time | Cost |
|--------|------|------|
| 120 agents, 20 rounds, mock encoder, no LLM | ~5 seconds | Free |
| 120 agents, 20 rounds, mock encoder, Groq 70B | ~5 minutes | Free (Groq free tier) |
| 500 agents, 168 rounds, TRIBE v2, Groq 70B | ~30 minutes | Free |

## License

MIT (NoLemming code). TRIBE v2 adapter uses CC BY-NC licensed model.
