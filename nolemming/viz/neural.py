"""Neural cortex visualization — society thinking.

Full-screen brain-shaped network with agents as neurons.
Minimal metrics bar at bottom. Hover for details.
Auto-plays simulation, subtle animation only on activation.

Usage:
    from nolemming.viz.neural import generate_neural_viz
    html = generate_neural_viz(result, response, trajectory, signals)
    Path("neural.html").write_text(html)
"""

from __future__ import annotations

import json
import sqlite3

from nolemming.analysis.sentiment import SentimentTrajectory
from nolemming.analysis.signals import PredictionSignals
from nolemming.core.types import NeuralArchetype, NeuralResponse, SimulationResult
from nolemming.mapping.brain_atlas import BrainAtlas

# Brain region positions (normalized 0-1, top-down cortex layout)
REGION_POSITIONS: dict[str, tuple[float, float]] = {
    "prefrontal_cortex": (0.42, 0.15),
    "acc": (0.58, 0.18),
    "motor_cortex": (0.25, 0.25),
    "parietal": (0.75, 0.25),
    "auditory_cortex": (0.18, 0.42),
    "visual_cortex": (0.82, 0.42),
    "temporal_pole": (0.22, 0.58),
    "somatosensory": (0.78, 0.55),
    "amygdala_proxy": (0.32, 0.68),
    "insula_proxy": (0.55, 0.65),
    "reward_circuit_proxy": (0.38, 0.78),
    "language_areas": (0.65, 0.75),
}

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

FALLBACK_COLOR = "#667788"


def generate_neural_viz(
    result: SimulationResult,
    response: NeuralResponse | None = None,
    trajectory: SentimentTrajectory | None = None,
    signals: PredictionSignals | None = None,
) -> str:
    nodes = _build_nodes(result)
    edges = _build_edges(result)
    posts = _load_posts(result)
    rois = _brain_rois(response) if response else {}
    metrics = _metrics(signals) if signals else {}

    return _render(
        nodes=nodes,
        edges=edges,
        posts=posts,
        rois=rois,
        metrics=metrics,
        archetypes=result.archetypes,
        n_rounds=result.config.n_rounds,
        stimulus=result.config.stimulus.path.stem,
    )


def _build_nodes(result: SimulationResult) -> list[dict]:
    nodes = []
    for agent in result.agents:
        region = _pick_region(agent.archetype, agent.agent_id)
        color = ARCHETYPE_COLORS.get(agent.archetype.label, FALLBACK_COLOR)
        nodes.append({
            "id": agent.agent_id,
            "name": agent.name,
            "user": agent.username,
            "arch": agent.archetype.label,
            "region": region,
            "color": color,
            "act": round(agent.activity_level, 3),
        })
    return nodes


def _pick_region(arch: NeuralArchetype, agent_id: int) -> str:
    """Spread agents across their top 2-3 dominant regions for visual distribution."""
    regions = arch.dominant_regions
    if not regions:
        all_regions = list(REGION_POSITIONS.keys())
        return all_regions[agent_id % len(all_regions)]
    return regions[agent_id % len(regions)]


def _build_edges(result: SimulationResult) -> list[dict]:
    """Build edges per round — sample up to 15 per round for performance."""
    edges: list[dict] = []
    try:
        with sqlite3.connect(str(result.db_path)) as conn:
            rounds = conn.execute(
                "SELECT DISTINCT round FROM posts ORDER BY round"
            ).fetchall()
            for (rnd,) in rounds:
                rows = conn.execute(
                    "SELECT p1.user_id, p2.user_id "
                    "FROM posts p1, posts p2 "
                    "WHERE p1.round = ? AND p2.round = ? "
                    "AND p1.user_id < p2.user_id LIMIT 15",
                    (rnd, rnd),
                ).fetchall()
                for s, t in rows:
                    edges.append({"s": s, "t": t, "r": rnd})
    except (sqlite3.OperationalError, FileNotFoundError):
        pass
    return edges


def _load_posts(result: SimulationResult) -> list[dict]:
    posts: list[dict] = []
    try:
        with sqlite3.connect(str(result.db_path)) as conn:
            rows = conn.execute(
                "SELECT p.user_id, p.round, p.content, u.username, u.archetype_id "
                "FROM posts p JOIN users u ON p.user_id = u.user_id "
                "ORDER BY p.round LIMIT 200"
            ).fetchall()
            for uid, rnd, content, uname, aid in rows:
                label = _find_label(aid, result.archetypes)
                posts.append({
                    "uid": uid, "r": rnd, "c": content[:100],
                    "u": uname, "a": label,
                })
    except (sqlite3.OperationalError, FileNotFoundError):
        pass
    return posts


def _brain_rois(response: NeuralResponse) -> dict[str, float]:
    atlas = BrainAtlas()
    return atlas.extract_all_rois(response.mean_activation())


def _metrics(signals: PredictionSignals) -> dict:
    return {
        "sent": round(signals.sentiment_score, 2),
        "cons": round(signals.consensus_strength, 2),
        "mom": round(signals.sentiment_momentum, 3),
        "dom": signals.dominant_archetype,
    }


def _find_label(aid: int, archetypes: list[NeuralArchetype]) -> str:
    for a in archetypes:
        if a.archetype_id == aid:
            return a.label
    return "unknown"


def _render(
    nodes: list[dict],
    edges: list[dict],
    posts: list[dict],
    rois: dict[str, float],
    metrics: dict,
    archetypes: list[NeuralArchetype],
    n_rounds: int,
    stimulus: str,
) -> str:
    arch_info = {
        a.label: {"frac": round(a.population_fraction, 2),
                   "color": ARCHETYPE_COLORS.get(a.label, FALLBACK_COLOR)}
        for a in archetypes
    }

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NoLemming — {stimulus}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#050510;overflow:hidden;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#aaa}}
canvas{{position:absolute;top:0;left:0}}

#hud{{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}}

#top-bar{{
  position:absolute;top:0;left:0;right:0;height:36px;
  display:flex;align-items:center;justify-content:space-between;
  padding:0 20px;
  background:linear-gradient(180deg,rgba(5,5,16,0.9) 0%,transparent 100%);
}}
#top-bar .title{{color:#667;font-size:12px;letter-spacing:2px;text-transform:uppercase}}
#top-bar .round{{color:#e94560;font-size:12px;font-weight:600}}
#top-bar .pause{{pointer-events:all;cursor:pointer;color:#445;font-size:11px;border:1px solid #223;border-radius:4px;padding:2px 10px;background:transparent}}
#top-bar .pause:hover{{color:#e94560;border-color:#e94560}}

#bottom-bar{{
  position:absolute;bottom:0;left:0;right:0;
  background:linear-gradient(0deg,rgba(5,5,16,0.95) 0%,transparent 100%);
  padding:12px 20px 10px;
}}
.metrics{{display:flex;gap:24px;justify-content:center;margin-bottom:8px}}
.metric{{text-align:center}}
.metric .val{{font-size:16px;font-weight:700;letter-spacing:-0.5px}}
.metric .lbl{{font-size:9px;color:#445;text-transform:uppercase;letter-spacing:1px;margin-top:2px}}
.legend{{display:flex;gap:14px;justify-content:center;flex-wrap:wrap}}
.legend-item{{display:flex;align-items:center;gap:5px;font-size:10px;color:#556}}
.legend-dot{{width:7px;height:7px;border-radius:50%}}

#tooltip{{
  position:absolute;display:none;
  background:rgba(10,10,25,0.92);backdrop-filter:blur(8px);
  border:1px solid rgba(100,100,150,0.2);border-radius:8px;
  padding:10px 14px;pointer-events:none;max-width:280px;
  box-shadow:0 4px 20px rgba(0,0,0,0.5);
}}
#tooltip .tt-name{{font-size:11px;font-weight:600;color:#ddd}}
#tooltip .tt-arch{{font-size:9px;padding:1px 6px;border-radius:8px;color:#fff;display:inline-block;margin-top:3px}}
#tooltip .tt-post{{font-size:10px;color:#889;margin-top:6px;line-height:1.4;font-style:italic}}

#region-label{{
  position:absolute;display:none;
  font-size:10px;color:rgba(255,255,255,0.3);
  text-transform:uppercase;letter-spacing:2px;
  pointer-events:none;
}}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="hud">
  <div id="top-bar">
    <span class="title">NoLemming &middot; {stimulus}</span>
    <span class="round" id="round-txt">Round 0 / {n_rounds}</span>
    <button class="pause" id="pause-btn" onclick="togglePause()">&#9646;&#9646;</button>
  </div>
  <div id="bottom-bar">
    <div class="metrics">
      <div class="metric"><div class="val" id="m-sent" style="color:{'#00ff88' if metrics.get('sent',0)>=0 else '#ff3b3b'}">{metrics.get('sent',0):+.2f}</div><div class="lbl">Sentiment</div></div>
      <div class="metric"><div class="val" style="color:#4488ff">{metrics.get('cons',0):.0%}</div><div class="lbl">Consensus</div></div>
      <div class="metric"><div class="val" style="color:#ffaa00">{metrics.get('mom',0):+.3f}</div><div class="lbl">Momentum</div></div>
      <div class="metric"><div class="val" style="color:#e94560;font-size:13px">{metrics.get('dom','—')}</div><div class="lbl">Dominant</div></div>
    </div>
    <div class="legend" id="legend"></div>
  </div>
  <div id="tooltip"><div class="tt-name"></div><div class="tt-arch"></div><div class="tt-post"></div></div>
  <div id="region-label"></div>
</div>

<script>
const NODES={json.dumps(nodes)};
const EDGES={json.dumps(edges)};
const POSTS={json.dumps(posts)};
const ARCHS={json.dumps(arch_info)};
const ROIS={json.dumps(rois)};
const REGIONS={json.dumps(REGION_POSITIONS)};
const N_ROUNDS={n_rounds};
const COLORS={json.dumps(ARCHETYPE_COLORS)};

const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');
let W,H,cx,cy,scale;
function resize(){{W=canvas.width=innerWidth;H=canvas.height=innerHeight;cx=W*0.5;cy=H*0.48;scale=Math.min(W,H)*0.42;}}
resize();
addEventListener('resize',resize);

// Position nodes in brain regions
const nodeMap={{}};
NODES.forEach(n=>{{
  const rp=REGIONS[n.region]||[0.5,0.5];
  const jitter=55+Math.random()*25;
  const angle=Math.random()*Math.PI*2;
  const dist=Math.random()*jitter;
  n.x=cx+(rp[0]-0.5)*scale*2+Math.cos(angle)*dist;
  n.y=cy+(rp[1]-0.5)*scale*2+Math.sin(angle)*dist;
  n.ox=n.x;n.oy=n.y;
  n.brightness=0.35+Math.random()*0.15;
  n.pulse=0;
  n.sz=3+n.act*4+Math.random()*2;
  nodeMap[n.id]=n;
}});

// Build post lookup by user
const postsByUser={{}};
POSTS.forEach(p=>{{if(!postsByUser[p.uid])postsByUser[p.uid]=[];postsByUser[p.uid].push(p);}});

// Legend
const leg=document.getElementById('legend');
for(const[label,info]of Object.entries(ARCHS)){{
  const d=document.createElement('div');d.className='legend-item';
  d.innerHTML=`<div class="legend-dot" style="background:${{info.color}};box-shadow:0 0 6px ${{info.color}}"></div>${{label}} ${{Math.round(info.frac*100)}}%`;
  leg.appendChild(d);
}}

// State
let currentRound=0;
let paused=false;
let frame=0;
const FPR=180; // frames per round (3s at 60fps)
let particles=[];
let mouseX=-1,mouseY=-1;
let hoveredNode=null;
let hoveredRegion=null;

function togglePause(){{
  paused=!paused;
  document.getElementById('pause-btn').innerHTML=paused?'&#9654;':'&#9646;&#9646;';
}}

// Advance round
function advanceRound(){{
  currentRound=(currentRound+1)%N_ROUNDS;
  document.getElementById('round-txt').textContent=`Round ${{currentRound}} / ${{N_ROUNDS}}`;

  // Activate agents in this round's edges
  const roundEdges=EDGES.filter(e=>e.r===currentRound);
  const activeIds=new Set();
  roundEdges.forEach(e=>{{activeIds.add(e.s);activeIds.add(e.t);}});

  NODES.forEach(n=>{{
    if(activeIds.has(n.id)){{
      n.brightness=1.0;
      n.pulse=1.0;
    }}
  }});

  // Spawn particles along active edges
  roundEdges.forEach(e=>{{
    const src=nodeMap[e.s],tgt=nodeMap[e.t];
    if(src&&tgt)particles.push({{
      x:src.x,y:src.y,tx:tgt.x,ty:tgt.y,
      p:0,spd:0.008+Math.random()*0.012,
      col:src.color,sz:1.5
    }});
  }});
}}

// Draw brain outline (organic shape with sulci)
function drawBrainOutline(){{
  const s=scale;
  ctx.save();
  ctx.translate(cx,cy);

  // Outer brain boundary — organic blob
  ctx.beginPath();
  const pts=24;
  for(let i=0;i<=pts;i++){{
    const a=(i/pts)*Math.PI*2;
    const wobble=1+Math.sin(a*3)*0.06+Math.cos(a*5)*0.03;
    const rx=s*0.95*wobble;
    const ry=s*1.05*wobble;
    const px=Math.cos(a)*rx;
    const py=Math.sin(a)*ry;
    if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py);
  }}
  ctx.closePath();
  ctx.strokeStyle='rgba(50,50,100,0.1)';
  ctx.lineWidth=1;
  ctx.stroke();

  // Central fissure (longitudinal)
  ctx.beginPath();
  ctx.moveTo(0,-s*0.85);
  ctx.bezierCurveTo(s*0.05,-s*0.3,-s*0.05,s*0.3,0,s*0.85);
  ctx.strokeStyle='rgba(50,50,100,0.07)';
  ctx.lineWidth=0.8;
  ctx.stroke();

  // Lateral sulcus (sylvian fissure)
  ctx.beginPath();
  ctx.moveTo(-s*0.7,-s*0.1);
  ctx.bezierCurveTo(-s*0.3,s*0.05,s*0.1,s*0.15,s*0.5,s*0.05);
  ctx.strokeStyle='rgba(50,50,100,0.06)';
  ctx.stroke();

  // Central sulcus
  ctx.beginPath();
  ctx.moveTo(-s*0.1,-s*0.8);
  ctx.bezierCurveTo(-s*0.2,-s*0.3,-s*0.15,s*0.1,-s*0.3,s*0.4);
  ctx.strokeStyle='rgba(50,50,100,0.05)';
  ctx.stroke();

  ctx.restore();
}}

// Draw connections (very faint, only current round)
function drawEdges(){{
  const re=EDGES.filter(e=>e.r===currentRound);
  re.forEach(e=>{{
    const s=nodeMap[e.s],t=nodeMap[e.t];
    if(!s||!t)return;
    ctx.beginPath();
    const edx=t.x-s.x,edy=t.y-s.y;
    const emx=(s.x+t.x)/2-edy*0.15;
    const emy=(s.y+t.y)/2+edx*0.15;
    ctx.moveTo(s.x,s.y);
    ctx.quadraticCurveTo(emx,emy,t.x,t.y);
    ctx.strokeStyle=toRGBA(s.color,0.04);
    ctx.lineWidth=0.4;
    ctx.stroke();
  }});
}}

function hexToRGB(hex){{
  const r=parseInt(hex.slice(1,3),16);
  const g=parseInt(hex.slice(3,5),16);
  const b=parseInt(hex.slice(5,7),16);
  return [r,g,b];
}}
function toRGBA(hex,a){{
  const[r,g,b]=hexToRGB(hex);
  return `rgba(${{r}},${{g}},${{b}},${{a}})`;
}}

// Draw particles
function drawParticles(){{
  particles.forEach(p=>{{
    const a=Math.max(0,(1-p.p)*0.8);
    ctx.beginPath();
    ctx.arc(p.x,p.y,p.sz,0,Math.PI*2);
    ctx.fillStyle=toRGBA(p.col,a);
    ctx.fill();
  }});
}}

// Draw nodes
function drawNodes(){{
  NODES.forEach(n=>{{
    const b=n.brightness;
    const r=n.sz;

    // Pulse ring (fades)
    if(n.pulse>0.01){{
      const pr=r+n.pulse*20;
      ctx.beginPath();
      ctx.arc(n.x,n.y,pr,0,Math.PI*2);
      ctx.strokeStyle=toRGBA(n.color,n.pulse*0.2);
      ctx.lineWidth=0.5;
      ctx.stroke();
    }}

    // Glow
    if(b>0.4){{
      const g=ctx.createRadialGradient(n.x,n.y,r,n.x,n.y,r+10*b);
      g.addColorStop(0,toRGBA(n.color,b*0.25));
      g.addColorStop(1,'transparent');
      ctx.beginPath();ctx.arc(n.x,n.y,r+10*b,0,Math.PI*2);
      ctx.fillStyle=g;ctx.fill();
    }}

    // Core dot
    ctx.beginPath();
    ctx.arc(n.x,n.y,r,0,Math.PI*2);
    ctx.fillStyle=toRGBA(n.color,b);
    ctx.fill();
  }});
}}

// Update
function update(){{
  frame++;
  if(!paused&&frame%FPR===0)advanceRound();

  // Decay brightness and pulse, drift nodes
  NODES.forEach(n=>{{
    n.brightness=Math.max(0.2,n.brightness*0.985);
    n.pulse*=0.95;
    // Drift back to rest position (gentle spring)
    n.x+=(n.ox-n.x)*0.015;
    n.y+=(n.oy-n.y)*0.015;
    // Subtle idle breathing
    n.x+=Math.sin(frame*0.008+n.id*1.7)*0.15;
    n.y+=Math.cos(frame*0.006+n.id*2.3)*0.12;
  }});

  // Active nodes nudge toward their interaction partners
  const re=EDGES.filter(e=>e.r===currentRound);
  re.forEach(e=>{{
    const s=nodeMap[e.s],t=nodeMap[e.t];
    if(!s||!t)return;
    const dx=t.x-s.x,dy=t.y-s.y;
    const d=Math.sqrt(dx*dx+dy*dy)||1;
    const pull=0.3/d;
    s.x+=dx*pull; s.y+=dy*pull;
    t.x-=dx*pull; t.y-=dy*pull;
  }});

  // Update particles
  particles=particles.filter(p=>{{
    p.p+=p.spd;
    const t=p.p;
    p.x=p.x+(p.tx-p.x)*p.spd*2;
    p.y=p.y+(p.ty-p.y)*p.spd*2;
    return p.p<1;
  }});

  // Hover detection
  hoveredNode=null;
  hoveredRegion=null;
  if(mouseX>=0){{
    let minD=20;
    NODES.forEach(n=>{{
      const d=Math.hypot(n.x-mouseX,n.y-mouseY);
      if(d<minD){{minD=d;hoveredNode=n;}}
    }});
    if(!hoveredNode){{
      for(const[name,[rx,ry]]of Object.entries(REGIONS)){{
        const px=cx+(rx-0.5)*scale*2;
        const py=cy+(ry-0.5)*scale*2;
        if(Math.hypot(px-mouseX,py-mouseY)<35){{hoveredRegion=name;break;}}
      }}
    }}
  }}
}}

// Tooltip
const ttEl=document.getElementById('tooltip');
const rlEl=document.getElementById('region-label');
function updateTooltip(){{
  if(hoveredNode){{
    const n=hoveredNode;
    ttEl.style.display='block';
    ttEl.style.left=(n.x+15)+'px';
    ttEl.style.top=(n.y-30)+'px';
    const col=COLORS[n.arch]||'#888';
    let postHtml='';
    const up=postsByUser[n.id];
    if(up&&up.length)postHtml=`<div class="tt-post">"${{up[up.length-1].c}}"</div>`;
    ttEl.innerHTML=`<div class="tt-name">@${{n.user}}</div><div class="tt-arch" style="background:${{col}}">${{n.arch}}</div>${{postHtml}}`;
    // Brighten on hover
    n.brightness=1;
  }}else{{
    ttEl.style.display='none';
  }}

  if(hoveredRegion&&!hoveredNode){{
    const[rx,ry]=REGIONS[hoveredRegion];
    rlEl.style.display='block';
    rlEl.style.left=(cx+(rx-0.5)*scale*2)+'px';
    rlEl.style.top=(cy+(ry-0.5)*scale*2-20)+'px';
    rlEl.textContent=hoveredRegion.replace(/_/g,' ');
    // Highlight all nodes in this region
    NODES.forEach(n=>{{if(n.region===hoveredRegion)n.brightness=0.9;}});
  }}else{{
    rlEl.style.display='none';
  }}
}}

// Draw
function draw(){{
  ctx.fillStyle='rgba(5,5,16,0.4)';
  ctx.fillRect(0,0,W,H);

  drawBrainOutline();
  drawEdges();
  drawParticles();
  drawNodes();
}}

function loop(){{
  update();
  draw();
  updateTooltip();
  requestAnimationFrame(loop);
}}

// Mouse tracking
canvas.addEventListener('mousemove',e=>{{mouseX=e.clientX;mouseY=e.clientY;}});
canvas.addEventListener('mouseleave',()=>{{mouseX=-1;mouseY=-1;}});
canvas.addEventListener('click',togglePause);

// Start
setTimeout(advanceRound,500);
loop();
</script>
</body>
</html>"""
