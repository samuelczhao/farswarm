"""Animated swarm network visualization.

Generates an interactive HTML visualization showing:
- Agent nodes colored by neural archetype, sized by activity
- Pulsing connections showing information flow
- Animated particles flowing between interacting agents
- Real-time simulation playback with archetype clustering

Opens in any browser. Screenshot/record it for social media.

Usage:
    from nolemming.viz.swarm import generate_swarm_viz
    html = generate_swarm_viz(sim_result)
    Path("swarm.html").write_text(html)
"""

from __future__ import annotations

import json
import random
import sqlite3

from nolemming.core.types import NeuralArchetype, SimulationResult

ARCHETYPE_COLORS: dict[str, str] = {
    "fear-dominant": "#ff3b3b",
    "reward-seeking": "#00ff88",
    "analytical": "#4488ff",
    "social-attuned": "#ffaa00",
    "risk-averse": "#cc44ff",
    "contrarian": "#ff6600",
    "verbal-analytical": "#00ddcc",
    "visual-driven": "#ff44aa",
    "empathetic": "#ff8888",
    "attention-focused": "#00ccff",
}


def generate_swarm_viz(
    result: SimulationResult,
    title: str = "NoLemming Swarm",
) -> str:
    """Generate animated swarm network HTML."""
    nodes = _build_nodes(result)
    edges = _build_edges(result)
    posts = _load_posts_for_viz(result)
    return _render_swarm_html(nodes, edges, posts, title, result)


def _build_nodes(result: SimulationResult) -> list[dict]:
    """Create node data for each agent."""
    nodes = []
    for agent in result.agents:
        color = ARCHETYPE_COLORS.get(agent.archetype.label, "#888888")
        nodes.append({
            "id": agent.agent_id,
            "label": agent.username,
            "archetype": agent.archetype.label,
            "color": color,
            "size": 4 + agent.activity_level * 12,
            "activity": agent.activity_level,
        })
    return nodes


def _build_edges(result: SimulationResult) -> list[dict]:
    """Build edges from interaction data or simulate from posts."""
    edges: list[dict] = []
    try:
        with sqlite3.connect(str(result.db_path)) as conn:
            rows = conn.execute(
                "SELECT p1.user_id, p2.user_id, p1.round "
                "FROM posts p1, posts p2 "
                "WHERE p1.round = p2.round AND p1.user_id != p2.user_id "
                "AND p1.rowid < p2.rowid "
                "LIMIT 200"
            ).fetchall()
            for src, tgt, rnd in rows:
                edges.append({
                    "source": src, "target": tgt, "round": rnd,
                })
    except (sqlite3.OperationalError, FileNotFoundError):
        pass

    if not edges:
        edges = _generate_synthetic_edges(result)
    return edges


def _generate_synthetic_edges(result: SimulationResult) -> list[dict]:
    """Generate plausible edges when DB data is sparse."""
    rng = random.Random(42)
    edges = []
    agents = result.agents
    for rnd in range(result.config.n_rounds):
        active = rng.sample(agents, min(len(agents), 8))
        for i in range(len(active) - 1):
            if active[i].archetype.label == active[i + 1].archetype.label:
                edges.append({
                    "source": active[i].agent_id,
                    "target": active[i + 1].agent_id,
                    "round": rnd,
                })
            elif rng.random() < 0.3:
                edges.append({
                    "source": active[i].agent_id,
                    "target": active[i + 1].agent_id,
                    "round": rnd,
                })
    return edges


def _load_posts_for_viz(result: SimulationResult) -> list[dict]:
    """Load posts with timing for animation."""
    posts: list[dict] = []
    try:
        with sqlite3.connect(str(result.db_path)) as conn:
            rows = conn.execute(
                "SELECT p.user_id, p.round, p.content, u.archetype_id "
                "FROM posts p JOIN users u ON p.user_id = u.user_id "
                "ORDER BY p.round LIMIT 100"
            ).fetchall()
            for uid, rnd, content, arch_id in rows:
                label = _find_label(arch_id, result.archetypes)
                posts.append({
                    "user_id": uid,
                    "round": rnd,
                    "content": content[:120],
                    "archetype": label,
                })
    except (sqlite3.OperationalError, FileNotFoundError):
        pass
    return posts


def _find_label(arch_id: int, archetypes: list[NeuralArchetype]) -> str:
    for a in archetypes:
        if a.archetype_id == arch_id:
            return a.label
    return "unknown"


def _render_swarm_html(
    nodes: list[dict],
    edges: list[dict],
    posts: list[dict],
    title: str,
    result: SimulationResult,
) -> str:
    archetype_summary = {}
    for a in result.archetypes:
        archetype_summary[a.label] = {
            "fraction": round(a.population_fraction, 2),
            "color": ARCHETYPE_COLORS.get(a.label, "#888"),
            "description": a.description[:100],
        }

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #060612; overflow: hidden; font-family: -apple-system, sans-serif; }}
canvas {{ display: block; }}
#overlay {{ position: absolute; top: 0; left: 0; pointer-events: none; width: 100%; height: 100%; }}
#title {{ position: absolute; top: 20px; left: 50%; transform: translateX(-50%); text-align: center; pointer-events: none; }}
#title h1 {{ color: #fff; font-size: 1.6rem; text-shadow: 0 0 20px rgba(233,69,96,0.5); }}
#title p {{ color: #e94560; font-size: 0.9rem; margin-top: 4px; }}
#legend {{ position: absolute; bottom: 20px; left: 20px; pointer-events: none; }}
.legend-item {{ display: flex; align-items: center; margin-bottom: 6px; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; box-shadow: 0 0 8px currentColor; }}
.legend-label {{ color: #aaa; font-size: 0.8rem; }}
.legend-pct {{ color: #666; font-size: 0.75rem; margin-left: 6px; }}
#feed {{ position: absolute; top: 80px; right: 20px; width: 320px; max-height: 60vh; overflow: hidden; pointer-events: none; }}
.feed-post {{ background: rgba(20,20,40,0.85); border-radius: 8px; padding: 10px; margin-bottom: 8px; border-left: 3px solid; opacity: 0; animation: fadeIn 0.5s forwards; backdrop-filter: blur(4px); }}
@keyframes fadeIn {{ to {{ opacity: 1; }} }}
.feed-post .arch {{ font-size: 0.7rem; padding: 2px 6px; border-radius: 8px; color: #fff; display: inline-block; }}
.feed-post .text {{ color: #ccc; font-size: 0.8rem; margin-top: 6px; line-height: 1.4; }}
#stats {{ position: absolute; top: 80px; left: 20px; pointer-events: none; }}
.stat {{ color: #888; font-size: 0.75rem; margin-bottom: 4px; }}
.stat span {{ color: #e94560; font-weight: bold; }}
#round-indicator {{ position: absolute; bottom: 20px; right: 20px; pointer-events: none; }}
#round-indicator .round {{ color: #e94560; font-size: 2rem; font-weight: bold; text-shadow: 0 0 15px rgba(233,69,96,0.4); }}
#round-indicator .label {{ color: #666; font-size: 0.7rem; text-transform: uppercase; }}
</style>
</head>
<body>
<canvas id="canvas"></canvas>
<div id="overlay">
    <div id="title">
        <h1>NoLemming</h1>
        <p>{result.config.stimulus.path.stem}</p>
    </div>
    <div id="legend"></div>
    <div id="feed"></div>
    <div id="stats">
        <div class="stat">Agents: <span>{len(nodes)}</span></div>
        <div class="stat">Archetypes: <span>{len(result.archetypes)}</span></div>
        <div class="stat">Rounds: <span>{result.config.n_rounds}</span></div>
    </div>
    <div id="round-indicator">
        <div class="label">Simulation Round</div>
        <div class="round" id="round-num">0</div>
    </div>
</div>

<script>
const NODES = {json.dumps(nodes)};
const EDGES = {json.dumps(edges)};
const POSTS = {json.dumps(posts)};
const ARCHETYPES = {json.dumps(archetype_summary)};
const N_ROUNDS = {result.config.n_rounds};

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let W, H;

function resize() {{
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
}}
resize();
window.addEventListener('resize', resize);

// Initialize node positions with archetype clustering
const archGroups = {{}};
NODES.forEach(n => {{
    if (!archGroups[n.archetype]) archGroups[n.archetype] = [];
    archGroups[n.archetype].push(n);
}});
const groupKeys = Object.keys(archGroups);
groupKeys.forEach((key, gi) => {{
    const angle = (gi / groupKeys.length) * Math.PI * 2;
    const cx = W/2 + Math.cos(angle) * Math.min(W, H) * 0.25;
    const cy = H/2 + Math.sin(angle) * Math.min(W, H) * 0.25;
    archGroups[key].forEach((n, ni) => {{
        const spread = 40 + archGroups[key].length * 3;
        n.x = cx + (Math.random() - 0.5) * spread;
        n.y = cy + (Math.random() - 0.5) * spread;
        n.vx = 0; n.vy = 0;
        n.targetX = cx;
        n.targetY = cy;
        n.pulse = Math.random() * Math.PI * 2;
    }});
}});

const nodeMap = {{}};
NODES.forEach(n => nodeMap[n.id] = n);

// Particles for edge animations
let particles = [];

function spawnParticle(edge) {{
    const src = nodeMap[edge.source];
    const tgt = nodeMap[edge.target];
    if (!src || !tgt) return;
    particles.push({{
        x: src.x, y: src.y,
        tx: tgt.x, ty: tgt.y,
        progress: 0,
        speed: 0.01 + Math.random() * 0.02,
        color: src.color,
        size: 2 + Math.random() * 2,
    }});
}}

// Legend
const legendEl = document.getElementById('legend');
for (const [label, info] of Object.entries(ARCHETYPES)) {{
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<div class="legend-dot" style="background:${{info.color}};color:${{info.color}}"></div><span class="legend-label">${{label}}</span><span class="legend-pct">${{Math.round(info.fraction*100)}}%</span>`;
    legendEl.appendChild(item);
}}

// Simulation playback
let currentRound = 0;
let frameCount = 0;
const FRAMES_PER_ROUND = 120;
const feedEl = document.getElementById('feed');
const roundEl = document.getElementById('round-num');

function advanceRound() {{
    currentRound = (currentRound + 1) % N_ROUNDS;
    roundEl.textContent = currentRound;

    // Spawn particles for this round's edges
    EDGES.filter(e => e.round === currentRound).forEach(e => {{
        spawnParticle(e);
    }});

    // Show posts for this round
    const roundPosts = POSTS.filter(p => p.round === currentRound);
    roundPosts.forEach(p => {{
        const color = ARCHETYPES[p.archetype]?.color || '#888';
        const div = document.createElement('div');
        div.className = 'feed-post';
        div.style.borderLeftColor = color;
        div.innerHTML = `<span class="arch" style="background:${{color}}">${{p.archetype}}</span><div class="text">${{p.content}}</div>`;
        feedEl.prepend(div);
        if (feedEl.children.length > 8) feedEl.lastChild.remove();
    }});

    // Pulse active nodes
    NODES.forEach(n => {{
        const isActive = EDGES.some(e => (e.source === n.id || e.target === n.id) && e.round === currentRound);
        if (isActive) n.pulse = 0;
    }});
}}

function update() {{
    frameCount++;
    if (frameCount % FRAMES_PER_ROUND === 0) advanceRound();

    // Force-directed layout
    NODES.forEach(n => {{
        // Attract to cluster center
        n.vx += (n.targetX - n.x) * 0.001;
        n.vy += (n.targetY - n.y) * 0.001;

        // Repel from other nodes
        NODES.forEach(m => {{
            if (n === m) return;
            const dx = n.x - m.x;
            const dy = n.y - m.y;
            const dist = Math.sqrt(dx*dx + dy*dy) || 1;
            if (dist < 60) {{
                const force = 0.5 / dist;
                n.vx += dx * force;
                n.vy += dy * force;
            }}
        }});

        n.vx *= 0.95;
        n.vy *= 0.95;
        n.x += n.vx;
        n.y += n.vy;
        n.pulse += 0.03;
    }});

    // Update particles
    particles = particles.filter(p => {{
        p.progress += p.speed;
        p.x = p.x + (p.tx - p.x) * p.speed * 3;
        p.y = p.y + (p.ty - p.y) * p.speed * 3;
        return p.progress < 1;
    }});
}}

function draw() {{
    ctx.fillStyle = 'rgba(6, 6, 18, 0.15)';
    ctx.fillRect(0, 0, W, H);

    // Draw edges (faint)
    const roundEdges = EDGES.filter(e => e.round === currentRound);
    ctx.globalAlpha = 0.15;
    roundEdges.forEach(e => {{
        const src = nodeMap[e.source];
        const tgt = nodeMap[e.target];
        if (!src || !tgt) return;
        ctx.beginPath();
        ctx.moveTo(src.x, src.y);
        ctx.lineTo(tgt.x, tgt.y);
        ctx.strokeStyle = src.color;
        ctx.lineWidth = 0.5;
        ctx.stroke();
    }});
    ctx.globalAlpha = 1;

    // Draw particles
    particles.forEach(p => {{
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.globalAlpha = 1 - p.progress;
        ctx.fill();
    }});
    ctx.globalAlpha = 1;

    // Draw nodes
    NODES.forEach(n => {{
        const pulseSize = n.size + Math.sin(n.pulse) * 2;

        // Glow
        ctx.beginPath();
        ctx.arc(n.x, n.y, pulseSize + 6, 0, Math.PI * 2);
        const glow = ctx.createRadialGradient(n.x, n.y, pulseSize, n.x, n.y, pulseSize + 6);
        glow.addColorStop(0, n.color + '40');
        glow.addColorStop(1, 'transparent');
        ctx.fillStyle = glow;
        ctx.fill();

        // Core
        ctx.beginPath();
        ctx.arc(n.x, n.y, pulseSize, 0, Math.PI * 2);
        ctx.fillStyle = n.color;
        ctx.fill();
    }});
}}

function loop() {{
    update();
    draw();
    requestAnimationFrame(loop);
}}
loop();

// Start first round
setTimeout(() => advanceRound(), 500);
</script>
</body>
</html>"""
