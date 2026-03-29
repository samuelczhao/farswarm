"""Interactive HTML dashboard for NoLemming simulation results.

Generates a single self-contained HTML file with:
- Brain region activation radar chart
- Archetype population distribution
- Sentiment trajectory over time by archetype
- Sample posts from each archetype
- Summary prediction metrics

Usage:
    from nolemming.viz.dashboard import generate_dashboard
    html = generate_dashboard(sim_result, archetypes, trajectory, signals)
    Path("dashboard.html").write_text(html)
"""

from __future__ import annotations

import json
import sqlite3

from nolemming.analysis.sentiment import SentimentTrajectory
from nolemming.analysis.signals import PredictionSignals
from nolemming.core.types import NeuralArchetype, NeuralResponse, SimulationResult
from nolemming.mapping.brain_atlas import BrainAtlas


def generate_dashboard(
    result: SimulationResult,
    response: NeuralResponse | None = None,
    trajectory: SentimentTrajectory | None = None,
    signals: PredictionSignals | None = None,
) -> str:
    """Generate a complete interactive HTML dashboard."""
    brain_data = _brain_radar_data(response) if response else {}
    archetype_data = _archetype_pie_data(result.archetypes)
    sentiment_data = _sentiment_line_data(trajectory) if trajectory else {}
    posts_data = _sample_posts(result)
    per_archetype = _per_archetype_sentiment(result)
    summary = _summary_data(signals) if signals else {}

    return _render_html(
        brain_data=brain_data,
        archetype_data=archetype_data,
        sentiment_data=sentiment_data,
        per_archetype=per_archetype,
        posts_data=posts_data,
        summary=summary,
        archetypes=result.archetypes,
        stimulus_name=result.config.stimulus.path.name,
    )


def _brain_radar_data(response: NeuralResponse) -> dict:
    """Extract per-region activation for radar chart."""
    atlas = BrainAtlas()
    rois = atlas.extract_all_rois(response.mean_activation())
    labels = list(rois.keys())
    values = [float(v) for v in rois.values()]
    psych = [atlas.REGION_PSYCH_LABELS.get(r, r) for r in labels]
    return {"labels": labels, "values": values, "psych": psych}


def _archetype_pie_data(archetypes: list[NeuralArchetype]) -> dict:
    """Archetype distribution for pie chart."""
    labels = [a.label for a in archetypes]
    values = [a.population_fraction for a in archetypes]
    colors = _archetype_colors(archetypes)
    return {"labels": labels, "values": values, "colors": colors}


def _sentiment_line_data(trajectory: SentimentTrajectory) -> dict:
    return {
        "timestamps": trajectory.timestamps,
        "scores": trajectory.scores,
        "volumes": trajectory.volumes,
    }


def _per_archetype_sentiment(result: SimulationResult) -> dict:
    """Get sentiment per archetype per round from DB."""
    from nolemming.analysis.sentiment import SentimentAnalyzer
    analyzer = SentimentAnalyzer()
    per_arch = analyzer.extract_per_archetype(result)
    out = {}
    for arch_id, traj in per_arch.items():
        label = _find_archetype_label(arch_id, result.archetypes)
        out[label] = {
            "timestamps": traj.timestamps,
            "scores": traj.scores,
        }
    return out


def _sample_posts(result: SimulationResult, n: int = 15) -> list[dict]:
    """Load sample posts from the simulation DB."""
    posts: list[dict] = []
    try:
        with sqlite3.connect(str(result.db_path)) as conn:
            cursor = conn.execute(
                "SELECT p.round, p.content, u.username, u.archetype_id "
                "FROM posts p JOIN users u ON p.user_id = u.user_id "
                "ORDER BY p.round LIMIT ?",
                (n,),
            )
            for row in cursor:
                label = _find_archetype_label(row[3], result.archetypes)
                posts.append({
                    "round": row[0],
                    "content": row[1],
                    "username": row[2],
                    "archetype": label,
                })
    except (sqlite3.OperationalError, FileNotFoundError):
        pass
    return posts


def _summary_data(signals: PredictionSignals) -> dict:
    return {
        "sentiment": signals.sentiment_score,
        "momentum": signals.sentiment_momentum,
        "consensus": signals.consensus_strength,
        "volatility": signals.volatility_estimate,
        "dominant": signals.dominant_archetype,
        "keywords": signals.narrative_keywords[:8],
    }


def _find_archetype_label(
    arch_id: int, archetypes: list[NeuralArchetype],
) -> str:
    for a in archetypes:
        if a.archetype_id == arch_id:
            return a.label
    return f"archetype-{arch_id}"


ARCHETYPE_COLOR_MAP: dict[str, str] = {
    "fear-dominant": "#e74c3c",
    "reward-seeking": "#2ecc71",
    "analytical": "#3498db",
    "social-attuned": "#f39c12",
    "risk-averse": "#9b59b6",
    "contrarian": "#e67e22",
    "verbal-analytical": "#1abc9c",
    "visual-driven": "#e91e63",
    "empathetic": "#ff6b6b",
    "attention-focused": "#00bcd4",
}


def _archetype_colors(archetypes: list[NeuralArchetype]) -> list[str]:
    fallback = ["#95a5a6", "#7f8c8d", "#bdc3c7", "#34495e"]
    colors = []
    for i, a in enumerate(archetypes):
        c = ARCHETYPE_COLOR_MAP.get(a.label, fallback[i % len(fallback)])
        colors.append(c)
    return colors


def _render_html(
    brain_data: dict,
    archetype_data: dict,
    sentiment_data: dict,
    per_archetype: dict,
    posts_data: list[dict],
    summary: dict,
    archetypes: list[NeuralArchetype],
    stimulus_name: str,
) -> str:
    """Render the complete HTML dashboard."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NoLemming — {stimulus_name}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0f; color: #e0e0e0; }}
.header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); padding: 2rem; text-align: center; border-bottom: 2px solid #e94560; }}
.header h1 {{ font-size: 2rem; color: #fff; margin-bottom: 0.5rem; }}
.header .tagline {{ color: #e94560; font-size: 1.1rem; }}
.header .stimulus {{ color: #888; margin-top: 0.5rem; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; padding: 1.5rem; max-width: 1400px; margin: 0 auto; }}
.card {{ background: #1a1a2e; border-radius: 12px; padding: 1.5rem; border: 1px solid #2a2a4a; }}
.card h2 {{ color: #e94560; font-size: 1.1rem; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 1px; }}
.full-width {{ grid-column: 1 / -1; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; }}
.metric {{ text-align: center; padding: 1rem; background: #12121f; border-radius: 8px; }}
.metric .value {{ font-size: 1.8rem; font-weight: bold; color: #e94560; }}
.metric .label {{ font-size: 0.75rem; color: #888; margin-top: 0.3rem; text-transform: uppercase; }}
.keyword-tags {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; }}
.keyword-tag {{ background: #16213e; color: #3498db; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem; }}
.post {{ background: #12121f; border-radius: 8px; padding: 1rem; margin-bottom: 0.8rem; border-left: 3px solid; }}
.post .meta {{ display: flex; justify-content: space-between; margin-bottom: 0.5rem; }}
.post .username {{ color: #3498db; font-weight: bold; }}
.post .archetype-badge {{ font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 12px; color: #fff; }}
.post .content {{ color: #ccc; line-height: 1.5; }}
.post .round {{ color: #666; font-size: 0.8rem; }}
.chart {{ width: 100%; min-height: 350px; }}
</style>
</head>
<body>
<div class="header">
    <h1>NoLemming</h1>
    <div class="tagline">Brain-Encoded Swarm Social Simulation</div>
    <div class="stimulus">Stimulus: {stimulus_name}</div>
</div>

<div class="grid">
    <!-- Summary Metrics -->
    <div class="card full-width">
        <h2>Prediction Summary</h2>
        <div class="metric-grid">
            <div class="metric">
                <div class="value" style="color: {'#2ecc71' if summary.get('sentiment', 0) > 0 else '#e74c3c'}">{summary.get('sentiment', 0):+.2f}</div>
                <div class="label">Sentiment</div>
            </div>
            <div class="metric">
                <div class="value">{summary.get('consensus', 0):.0%}</div>
                <div class="label">Consensus</div>
            </div>
            <div class="metric">
                <div class="value" style="color: {'#2ecc71' if summary.get('momentum', 0) > 0 else '#e74c3c'}">{summary.get('momentum', 0):+.3f}</div>
                <div class="label">Momentum</div>
            </div>
            <div class="metric">
                <div class="value">{summary.get('volatility', 0):.3f}</div>
                <div class="label">Volatility</div>
            </div>
            <div class="metric">
                <div class="value" style="font-size: 1rem; color: #f39c12;">{summary.get('dominant', 'N/A')}</div>
                <div class="label">Dominant Archetype</div>
            </div>
        </div>
        <div class="keyword-tags">
            {''.join(f'<span class="keyword-tag">{kw}</span>' for kw in summary.get('keywords', []))}
        </div>
    </div>

    <!-- Brain Radar -->
    <div class="card">
        <h2>Brain Region Activation</h2>
        <div id="brain-radar" class="chart"></div>
    </div>

    <!-- Archetype Distribution -->
    <div class="card">
        <h2>Neural Archetype Population</h2>
        <div id="archetype-pie" class="chart"></div>
    </div>

    <!-- Sentiment Timeline -->
    <div class="card full-width">
        <h2>Sentiment by Archetype Over Time</h2>
        <div id="sentiment-timeline" class="chart"></div>
    </div>

    <!-- Sample Posts -->
    <div class="card full-width">
        <h2>Sample Agent Posts</h2>
        <div id="posts-container">
            {_render_posts_html(posts_data, archetypes)}
        </div>
    </div>
</div>

<script>
const brainData = {json.dumps(brain_data)};
const archetypeData = {json.dumps(archetype_data)};
const sentimentData = {json.dumps(sentiment_data)};
const perArchetype = {json.dumps(per_archetype)};

// Brain Radar Chart
if (brainData.labels) {{
    Plotly.newPlot('brain-radar', [{{
        type: 'scatterpolar',
        r: brainData.values,
        theta: brainData.psych,
        fill: 'toself',
        fillcolor: 'rgba(233, 69, 96, 0.2)',
        line: {{ color: '#e94560', width: 2 }},
        marker: {{ size: 6, color: '#e94560' }},
    }}], {{
        polar: {{
            radialaxis: {{ visible: true, color: '#555', gridcolor: '#2a2a4a' }},
            angularaxis: {{ color: '#888' }},
            bgcolor: 'transparent',
        }},
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: {{ t: 30, b: 30, l: 60, r: 60 }},
        font: {{ color: '#ccc' }},
        showlegend: false,
    }}, {{ responsive: true }});
}}

// Archetype Pie
if (archetypeData.labels) {{
    Plotly.newPlot('archetype-pie', [{{
        type: 'pie',
        labels: archetypeData.labels,
        values: archetypeData.values,
        marker: {{ colors: archetypeData.colors }},
        textinfo: 'label+percent',
        textfont: {{ color: '#fff', size: 12 }},
        hole: 0.4,
    }}], {{
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: {{ t: 20, b: 20, l: 20, r: 20 }},
        font: {{ color: '#ccc' }},
        showlegend: false,
    }}, {{ responsive: true }});
}}

// Sentiment Timeline
const archColors = {json.dumps(ARCHETYPE_COLOR_MAP)};
const defaultColors = ['#95a5a6', '#7f8c8d', '#bdc3c7', '#34495e', '#e74c3c', '#2ecc71', '#3498db'];
const sentTraces = [];
let ci = 0;
for (const [label, data] of Object.entries(perArchetype)) {{
    sentTraces.push({{
        x: data.timestamps,
        y: data.scores,
        name: label,
        type: 'scatter',
        mode: 'lines+markers',
        line: {{ width: 2, color: archColors[label] || defaultColors[ci % defaultColors.length] }},
        marker: {{ size: 5 }},
    }});
    ci++;
}}
if (sentimentData.timestamps) {{
    sentTraces.push({{
        x: sentimentData.timestamps,
        y: sentimentData.scores,
        name: 'Aggregate',
        type: 'scatter',
        mode: 'lines',
        line: {{ width: 3, color: '#fff', dash: 'dash' }},
    }});
}}
if (sentTraces.length > 0) {{
    Plotly.newPlot('sentiment-timeline', sentTraces, {{
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: {{ title: 'Hours', color: '#888', gridcolor: '#2a2a4a' }},
        yaxis: {{ title: 'Sentiment', color: '#888', gridcolor: '#2a2a4a', range: [-1, 1] }},
        margin: {{ t: 20, b: 50, l: 60, r: 20 }},
        font: {{ color: '#ccc' }},
        legend: {{ x: 0, y: 1, font: {{ size: 11 }} }},
    }}, {{ responsive: true }});
}}
</script>
</body>
</html>"""


def _render_posts_html(
    posts: list[dict], archetypes: list[NeuralArchetype],
) -> str:
    """Render sample posts as HTML cards."""
    if not posts:
        return '<p style="color: #666;">No posts generated yet.</p>'
    html_parts = []
    for post in posts:
        color = ARCHETYPE_COLOR_MAP.get(post["archetype"], "#95a5a6")
        html_parts.append(
            f'<div class="post" style="border-left-color: {color};">'
            f'<div class="meta">'
            f'<span class="username">@{post["username"]}</span>'
            f'<span class="archetype-badge" style="background: {color};">'
            f'{post["archetype"]}</span>'
            f'</div>'
            f'<div class="content">{post["content"]}</div>'
            f'<div class="round">Round {post["round"]}</div>'
            f'</div>'
        )
    return "\n".join(html_parts)
