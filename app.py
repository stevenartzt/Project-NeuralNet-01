#!/usr/bin/env python3
"""
Neural Net Dashboard — Live Training Visualization
Port 5005 → artzt.cloud/neural
"""

import os
import sys
import json
import glob
import subprocess
import threading
from flask import Flask, jsonify, render_template_string, request
from datetime import datetime

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Track active training process
training_process = {"proc": None, "run_id": None}

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neural Net — Direction Engine</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a0f; color: #e0e0e0; font-family: 'JetBrains Mono', 'Fira Code', monospace; }

  .header {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a0a2e 100%);
    border-bottom: 1px solid #2a1a4a;
    padding: 20px 30px;
    display: flex; justify-content: space-between; align-items: center;
  }
  .header h1 { font-size: 1.4em; color: #a78bfa; }
  .header h1 span { color: #6366f1; }
  .header .status { font-size: 0.85em; padding: 4px 12px; border-radius: 12px; }
  .status-training { background: #1e3a2e; color: #4ade80; animation: pulse 2s infinite; }
  .status-complete { background: #1a2e4a; color: #60a5fa; }
  .status-idle { background: #2a2a2a; color: #888; }

  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.6; } }

  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    padding: 20px;
    max-width: 1400px;
    margin: 0 auto;
  }
  .card {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 16px;
  }
  .card h3 { color: #a78bfa; font-size: 0.9em; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
  .card.full { grid-column: 1 / -1; }

  .stats-row {
    display: flex; gap: 16px; padding: 0 20px; flex-wrap: wrap;
  }
  .stat-box {
    background: #12121a; border: 1px solid #1e1e2e; border-radius: 8px;
    padding: 12px 20px; text-align: center; flex: 1; min-width: 120px;
  }
  .stat-box .value { font-size: 1.6em; font-weight: bold; color: #a78bfa; }
  .stat-box .label { font-size: 0.75em; color: #888; margin-top: 2px; }

  .controls {
    padding: 16px 20px; display: flex; gap: 12px; align-items: center;
  }
  .btn {
    background: #6366f1; color: white; border: none; padding: 8px 20px;
    border-radius: 6px; cursor: pointer; font-family: inherit; font-size: 0.85em;
  }
  .btn:hover { background: #7c3aed; }
  .btn:disabled { background: #333; color: #666; cursor: not-allowed; }
  .btn-danger { background: #dc2626; }
  .btn-danger:hover { background: #ef4444; }

  .network-viz {
    display: flex; align-items: center; justify-content: center;
    min-height: 300px; position: relative;
  }
  .network-viz svg { width: 100%; max-height: 350px; }

  .data-table { width: 100%; border-collapse: collapse; font-size: 0.8em; }
  .data-table th, .data-table td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #1e1e2e; }
  .data-table th { color: #a78bfa; }

  .input-stream { max-height: 200px; overflow-y: auto; font-size: 0.75em; color: #888; }
  .input-row { padding: 3px 0; border-bottom: 1px solid #0f0f15; }
  .input-row .ticker { color: #60a5fa; font-weight: bold; }
  .input-row .val { color: #4ade80; }

  #confusion-matrix td { text-align: center; font-weight: bold; min-width: 60px; }
  .cm-high { background: rgba(99, 102, 241, 0.3); }
  .cm-low { background: rgba(220, 38, 38, 0.15); }
</style>
</head>
<body>

<div class="header">
  <h1><span>⚡</span> Neural Net — Direction Engine</h1>
  <div class="status status-idle" id="status">IDLE</div>
</div>

<div class="controls">
  <button class="btn" id="btn-train" onclick="startTraining()">🧠 Train Model</button>
  <button class="btn btn-danger" id="btn-stop" onclick="stopTraining()" disabled>⏹ Stop</button>
  <select id="run-select" onchange="loadRun(this.value)" style="background:#1e1e2e;color:#e0e0e0;border:1px solid #333;padding:6px;border-radius:4px;font-family:inherit;">
    <option value="">— Select Run —</option>
  </select>
  <span id="last-update" style="color:#555;font-size:0.8em;margin-left:auto;"></span>
</div>

<div class="stats-row" id="stats-row">
  <div class="stat-box"><div class="value" id="stat-epoch">—</div><div class="label">Epoch</div></div>
  <div class="stat-box"><div class="value" id="stat-train-acc">—</div><div class="label">Train Acc</div></div>
  <div class="stat-box"><div class="value" id="stat-test-acc">—</div><div class="label">Test Acc</div></div>
  <div class="stat-box"><div class="value" id="stat-loss">—</div><div class="label">Test Loss</div></div>
  <div class="stat-box"><div class="value" id="stat-samples">—</div><div class="label">Samples</div></div>
  <div class="stat-box"><div class="value" id="stat-params">—</div><div class="label">Parameters</div></div>
</div>

<div style="padding:0 20px;">
  <div style="background:#1e1e2e;border-radius:6px;height:16px;overflow:hidden;position:relative;" id="progress-bar-container">
    <div id="progress-bar" style="background:linear-gradient(90deg,#6366f1,#a78bfa);height:100%;width:0%;transition:width 0.5s;border-radius:6px;"></div>
    <div id="progress-text" style="position:absolute;top:0;left:0;right:0;text-align:center;font-size:0.7em;color:#fff;line-height:16px;">—</div>
  </div>
</div>

<div class="grid">
  <!-- Loss Curve -->
  <div class="card">
    <h3>📉 Loss Curve</h3>
    <div id="loss-chart" style="height:280px;"></div>
  </div>

  <!-- Accuracy -->
  <div class="card">
    <h3>📈 Accuracy</h3>
    <div id="accuracy-chart" style="height:280px;"></div>
  </div>

  <!-- Network Topology + Activations -->
  <div class="card full">
    <h3>🧠 Network Architecture — Neuron Activations <span id="training-badge" style="display:none;background:#1e3a2e;color:#4ade80;padding:2px 10px;border-radius:8px;font-size:0.8em;animation:pulse 1.5s infinite;">⚡ LIVE</span></h3>
    <div id="network-canvas" style="height:350px;overflow:hidden;background:#08080d;border-radius:8px;"></div>
  </div>

  <!-- Feature Importance -->
  <div class="card">
    <h3>🎯 Feature Importance (Emergent)</h3>
    <div id="importance-chart" style="height:300px;"></div>
  </div>

  <!-- Confusion Matrix -->
  <div class="card">
    <h3>🔢 Confusion Matrix</h3>
    <div id="confusion-div" style="height:300px;"></div>
  </div>

  <!-- Input Data Stream -->
  <div class="card">
    <h3>📊 Input Data Stream</h3>
    <div class="input-stream" id="input-stream">
      <span style="color:#555;">Waiting for training data...</span>
    </div>
  </div>

  <!-- Per-Class Accuracy -->
  <div class="card">
    <h3>📊 Per-Class Accuracy Over Time</h3>
    <div id="per-class-chart" style="height:280px;"></div>
  </div>

  <!-- Auto-Optimizer -->
  <div class="card full">
    <h3>🧬 Auto-Optimizer — Hyperparameter Search
      <button class="btn" style="margin-left:12px;font-size:0.75em;padding:4px 12px;" onclick="startOptimize(200)">🧬 Run 200 Trials</button>
      <button class="btn" style="font-size:0.75em;padding:4px 12px;background:#333;" onclick="startOptimize(50)">Quick 50</button>
      <span id="opt-status" style="color:#555;font-size:0.8em;margin-left:8px;"></span>
    </h3>
    <div style="display:flex;gap:16px;margin-top:10px;">
      <div style="flex:1;">
        <div style="background:#1e1e2e;border-radius:6px;height:12px;overflow:hidden;margin-bottom:8px;">
          <div id="opt-progress" style="background:linear-gradient(90deg,#8b5cf6,#a78bfa);height:100%;width:0%;transition:width 0.5s;"></div>
        </div>
        <div id="opt-chart" style="height:250px;"></div>
      </div>
      <div style="flex:0 0 250px;font-size:0.8em;" id="opt-info">
        <div style="color:#555;">Run optimizer to search for the best architecture</div>
        <div style="color:#444;font-size:0.85em;margin-top:8px;">
          Tests hundreds of combinations:<br>
          • 1-5 hidden layers<br>
          • 16-256 neurons per layer<br>
          • Learning rate, dropout, batch size<br>
          • Activation functions (ReLU, LeakyReLU, GELU)<br>
          <br>
          Uses Bayesian optimization — learns which<br>
          regions of the search space are promising<br>
          and focuses there. Smarter than random.
        </div>
      </div>
    </div>
  </div>

  <!-- Live Inference Visualizer -->
  <div class="card full">
    <h3>🔬 Live Inference — Watch a Prediction
      <button class="btn" style="margin-left:12px;font-size:0.75em;padding:4px 12px;" onclick="runInference()">▶ Run Sample</button>
      <button class="btn" style="font-size:0.75em;padding:4px 12px;background:#333;" onclick="autoInference()">🔄 Auto</button>
      <span id="inf-status" style="color:#555;font-size:0.8em;margin-left:8px;"></span>
    </h3>
    <div style="display:flex;gap:16px;margin-top:10px;">
      <!-- Input Panel -->
      <div style="flex:0 0 200px;max-height:400px;overflow-y:auto;" id="inf-inputs">
        <div style="color:#555;font-size:0.8em;">Run a sample to see inputs</div>
      </div>
      <!-- Network Flow Canvas -->
      <div style="flex:1;min-height:400px;background:#08080d;border-radius:8px;" id="inf-network"></div>
      <!-- Output Panel -->
      <div style="flex:0 0 160px;" id="inf-output">
        <div style="color:#555;font-size:0.8em;">Waiting for prediction...</div>
      </div>
    </div>
  </div>
</div>

<script>
let pollInterval = null;
let currentRunId = null;
const BASE = window.location.pathname.replace(/\/$/, '');

async function fetchRuns() {
  const res = await fetch(BASE + '/api/runs');
  const data = await res.json();
  const sel = document.getElementById('run-select');
  sel.innerHTML = '<option value="">— Select Run —</option>';
  data.runs.forEach(r => {
    const opt = document.createElement('option');
    opt.value = r.run_id;
    opt.textContent = r.run_id + (r.status === 'training' ? ' ⚡' : ' ✅');
    sel.appendChild(opt);
  });
  // Auto-select training or latest
  const training = data.runs.find(r => r.status === 'training');
  if (training) {
    sel.value = training.run_id;
    loadRun(training.run_id);
  } else if (data.runs.length > 0) {
    sel.value = data.runs[0].run_id;
    loadRun(data.runs[0].run_id);
  }
}

async function loadRun(runId) {
  if (!runId) return;
  currentRunId = runId;
  await updateDashboard();
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(updateDashboard, 2000);
}

async function updateDashboard() {
  if (!currentRunId) return;
  try {
    const res = await fetch(BASE + '/api/history/' + currentRunId);
    const h = await res.json();
    if (h.error) return;

    // Status badge
    const statusEl = document.getElementById('status');
    statusEl.textContent = h.status.toUpperCase();
    statusEl.className = 'status status-' + h.status;

    const epochs = h.epochs_data || [];
    if (epochs.length === 0) return;

    const latest = epochs[epochs.length - 1];

    // Stats
    document.getElementById('stat-epoch').textContent = latest.epoch;
    document.getElementById('stat-train-acc').textContent = (latest.train_accuracy * 100).toFixed(1) + '%';
    document.getElementById('stat-test-acc').textContent = (latest.test_accuracy * 100).toFixed(1) + '%';
    document.getElementById('stat-loss').textContent = latest.test_loss.toFixed(4);
    document.getElementById('stat-samples').textContent = (h.config?.train_size || 0).toLocaleString();
    document.getElementById('stat-params').textContent = (h.network_architecture?.total_params || 0).toLocaleString();
    document.getElementById('last-update').textContent = 'Updated: ' + new Date().toLocaleTimeString();

    // Progress bar
    const totalEpochs = h.config?.epochs || 100;
    const currentEpoch = latest.epoch;
    const pct = h.status === 'complete' ? 100 : Math.round((currentEpoch / totalEpochs) * 100);
    document.getElementById('progress-bar').style.width = pct + '%';
    document.getElementById('progress-bar').style.background = h.status === 'complete'
      ? 'linear-gradient(90deg, #4ade80, #22c55e)'
      : 'linear-gradient(90deg, #6366f1, #a78bfa)';
    const elapsed = h.started_at && latest.timestamp ? Math.round((new Date(latest.timestamp) - new Date(h.started_at)) / 1000) : 0;
    const perEpoch = currentEpoch > 0 ? elapsed / currentEpoch : 0;
    const remaining = h.status === 'complete' ? 0 : Math.round(perEpoch * (totalEpochs - currentEpoch));
    document.getElementById('progress-text').textContent = h.status === 'complete'
      ? `✅ Complete — ${currentEpoch} epochs in ${elapsed}s`
      : `Epoch ${currentEpoch}/${totalEpochs} (${pct}%) — ~${remaining}s remaining`;

    // Buttons + training badge
    document.getElementById('btn-train').disabled = h.status === 'training';
    document.getElementById('btn-stop').disabled = h.status !== 'training';
    document.getElementById('training-badge').style.display = h.status === 'training' ? 'inline' : 'none';

    // Loss chart
    const ep = epochs.map(e => e.epoch);
    Plotly.react('loss-chart', [
      { x: ep, y: epochs.map(e => e.train_loss), name: 'Train', line: { color: '#6366f1' } },
      { x: ep, y: epochs.map(e => e.test_loss), name: 'Test', line: { color: '#f97316', dash: 'dash' } }
    ], {
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: { color: '#888', size: 10 },
      margin: { t: 10, b: 30, l: 50, r: 10 },
      xaxis: { title: 'Epoch', gridcolor: '#1e1e2e' },
      yaxis: { title: 'Loss', gridcolor: '#1e1e2e' },
      legend: { x: 0.7, y: 1 },
      showlegend: true
    }, { responsive: true });

    // Accuracy chart
    Plotly.react('accuracy-chart', [
      { x: ep, y: epochs.map(e => e.train_accuracy * 100), name: 'Train', line: { color: '#4ade80' } },
      { x: ep, y: epochs.map(e => e.test_accuracy * 100), name: 'Test', line: { color: '#f97316', dash: 'dash' } }
    ], {
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: { color: '#888', size: 10 },
      margin: { t: 10, b: 30, l: 50, r: 10 },
      xaxis: { title: 'Epoch', gridcolor: '#1e1e2e' },
      yaxis: { title: 'Accuracy %', gridcolor: '#1e1e2e' },
      legend: { x: 0.7, y: 0.3 },
      showlegend: true
    }, { responsive: true });

    // Per-class accuracy
    if (latest.per_class_accuracy) {
      Plotly.react('per-class-chart', [
        { x: ep, y: epochs.map(e => (e.per_class_accuracy?.buy || 0) * 100), name: 'BUY', line: { color: '#4ade80' } },
        { x: ep, y: epochs.map(e => (e.per_class_accuracy?.sell || 0) * 100), name: 'SELL', line: { color: '#ef4444' } },
        { x: ep, y: epochs.map(e => (e.per_class_accuracy?.neutral || 0) * 100), name: 'NEUTRAL', line: { color: '#888' } }
      ], {
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#888', size: 10 },
        margin: { t: 10, b: 30, l: 50, r: 10 },
        xaxis: { title: 'Epoch', gridcolor: '#1e1e2e' },
        yaxis: { title: 'Accuracy %', gridcolor: '#1e1e2e' },
        legend: { x: 0.7, y: 0.3 }
      }, { responsive: true });
    }

    // Feature importance
    if (h.feature_importance) {
      const fi = Object.entries(h.feature_importance).slice(0, 15);
      Plotly.react('importance-chart', [{
        y: fi.map(f => f[0]),
        x: fi.map(f => f[1]),
        type: 'bar', orientation: 'h',
        marker: { color: fi.map((_, i) => `hsl(${260 - i * 8}, 70%, ${55 + i}%)`) }
      }], {
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#888', size: 10 },
        margin: { t: 10, b: 30, l: 100, r: 10 },
        xaxis: { title: 'Gradient Importance', gridcolor: '#1e1e2e' },
        yaxis: { autorange: 'reversed' }
      }, { responsive: true });
    }

    // Confusion matrix
    if (latest.confusion_matrix) {
      const cm = latest.confusion_matrix;
      const labels = ['SELL', 'NEUTRAL', 'BUY'];
      Plotly.react('confusion-div', [{
        z: cm, x: labels, y: labels, type: 'heatmap',
        colorscale: [[0, '#0a0a0f'], [0.5, '#2a1a4a'], [1, '#6366f1']],
        showscale: false,
        text: cm.map(row => row.map(v => v.toString())),
        texttemplate: '%{text}',
        textfont: { color: '#fff', size: 16 }
      }], {
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#888', size: 10 },
        margin: { t: 10, b: 40, l: 70, r: 10 },
        xaxis: { title: 'Predicted', side: 'bottom' },
        yaxis: { title: 'Actual', autorange: 'reversed' }
      }, { responsive: true });
    }

    // Network architecture visualization
    drawNetwork(h.network_architecture, latest.activation_stats, h.status === 'training');

    // Input data stream
    if (h.config?.features) {
      const streamDiv = document.getElementById('input-stream');
      streamDiv.innerHTML = h.config.features.map(f =>
        `<div class="input-row"><span class="ticker">${f}</span></div>`
      ).join('') +
      `<div style="margin-top:8px;color:#4ade80;">${h.config.train_size.toLocaleString()} samples × ${h.config.features.length} features</div>` +
      `<div style="color:#60a5fa;">Class split: Buy ${h.config.class_distribution?.train?.buy?.toLocaleString() || '?'} | Neutral ${h.config.class_distribution?.train?.neutral?.toLocaleString() || '?'} | Sell ${h.config.class_distribution?.train?.sell?.toLocaleString() || '?'}</div>`;
    }

    // Adjust polling speed based on training status
    if (h.status === 'complete' && pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    } else if (h.status === 'training' && pollInterval) {
      // Poll faster during active training
      clearInterval(pollInterval);
      pollInterval = setInterval(updateDashboard, 1000);
    }
  } catch (err) {
    console.error('Dashboard update error:', err);
  }
}

let lastEpochDrawn = 0;
let networkAnimFrame = null;
let particles = [];

function drawNetwork(arch, activations, isTraining) {
  if (!arch || !arch.layers) return;
  const canvas = document.getElementById('network-canvas');

  const linearLayers = arch.layers.filter(l => l.type === 'linear');
  if (linearLayers.length === 0) return;

  const sizes = [linearLayers[0].in];
  linearLayers.forEach(l => sizes.push(l.out));

  const W = canvas.clientWidth || 800;
  const H = 340;
  const maxNodes = 10;

  // Store layout globally for animation
  const layerX = sizes.map((_, i) => 80 + (W - 160) * i / (sizes.length - 1));
  const nodeMap = [];

  sizes.forEach((size, li) => {
    const displaySize = Math.min(size, maxNodes);
    const spacing = Math.min(28, (H - 80) / displaySize);
    const startY = (H - spacing * (displaySize - 1)) / 2;
    const nodes = [];
    const actData = activations && activations[li] ? activations[li] : null;

    for (let ni = 0; ni < displaySize; ni++) {
      const x = layerX[li];
      const y = startY + ni * spacing;
      let activation = 0;
      if (actData && actData.per_neuron_mean && actData.per_neuron_mean[ni] !== undefined) {
        activation = Math.min(actData.per_neuron_mean[ni] / (actData.max || 1), 1);
      }
      nodes.push({ x, y, activation, neuronIdx: ni });
    }
    nodeMap.push({ nodes, totalSize: size, displaySize });
  });

  // Use canvas element for smooth animation
  let cvs = canvas.querySelector('canvas');
  if (!cvs) {
    canvas.innerHTML = '';
    cvs = document.createElement('canvas');
    cvs.width = W * 2; cvs.height = H * 2;
    cvs.style.width = W + 'px'; cvs.style.height = H + 'px';
    canvas.appendChild(cvs);
  }
  const ctx = cvs.getContext('2d');
  ctx.setTransform(2, 0, 0, 2, 0, 0); // retina

  // Spawn particles during training
  if (isTraining && Math.random() < 0.3) {
    for (let li = 0; li < nodeMap.length - 1; li++) {
      const fromNodes = nodeMap[li].nodes;
      const toNodes = nodeMap[li + 1].nodes;
      const fi = Math.floor(Math.random() * fromNodes.length);
      const ti = Math.floor(Math.random() * toNodes.length);
      particles.push({
        x: fromNodes[fi].x, y: fromNodes[fi].y,
        tx: toNodes[ti].x, ty: toNodes[ti].y,
        progress: 0, speed: 0.02 + Math.random() * 0.03,
        layer: li
      });
    }
  }

  // Cancel previous animation
  if (networkAnimFrame) cancelAnimationFrame(networkAnimFrame);

  function render() {
    ctx.clearRect(0, 0, W, H);

    // Draw connections
    for (let li = 0; li < nodeMap.length - 1; li++) {
      const from = nodeMap[li].nodes;
      const to = nodeMap[li + 1].nodes;
      for (let fi = 0; fi < from.length; fi++) {
        for (let ti = 0; ti < to.length; ti++) {
          // Connection brightness based on both node activations
          const strength = (from[fi].activation + to[ti].activation) / 2;
          ctx.beginPath();
          ctx.moveTo(from[fi].x, from[fi].y);
          ctx.lineTo(to[ti].x, to[ti].y);
          ctx.strokeStyle = `rgba(99, 102, 241, ${0.03 + strength * 0.2})`;
          ctx.lineWidth = 0.5 + strength * 1.5;
          ctx.stroke();
        }
      }
    }

    // Draw & update particles (data flowing through network)
    particles = particles.filter(p => p.progress < 1);
    for (const p of particles) {
      p.progress += p.speed;
      const cx = p.x + (p.tx - p.x) * p.progress;
      const cy = p.y + (p.ty - p.y) * p.progress;
      const alpha = Math.sin(p.progress * Math.PI);
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(74, 222, 128, ${alpha * 0.9})`;
      ctx.fill();
      // Glow
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(74, 222, 128, ${alpha * 0.2})`;
      ctx.fill();
    }

    // Draw nodes
    for (let li = 0; li < nodeMap.length; li++) {
      const layer = nodeMap[li];
      for (const node of layer.nodes) {
        const act = node.activation;
        const pulseScale = isTraining ? 1 + Math.sin(Date.now() / 300 + node.neuronIdx) * 0.15 * act : 1;
        const radius = (7 + act * 5) * pulseScale;

        // Outer glow for active neurons
        if (act > 0.3) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius + 6, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(99, 102, 241, ${act * 0.15})`;
          ctx.fill();
        }

        // Node
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        const r = Math.round(40 + act * 215);
        const g = Math.round(40 + act * 62);
        const b = Math.round(60 + act * 181);
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fill();
        ctx.strokeStyle = `rgba(163, 139, 250, ${0.3 + act * 0.5})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Layer label
      const lx = layerX[li];
      const label = li === 0 ? `Input (${layer.totalSize})` : li === nodeMap.length - 1 ? `Output (${layer.totalSize})` : `Hidden (${layer.totalSize})`;
      ctx.fillStyle = '#a78bfa';
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(label, lx, 18);

      if (layer.totalSize > maxNodes) {
        ctx.fillStyle = '#555';
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.fillText(`+${layer.totalSize - layer.displaySize} more`, lx, H - 8);
      }
    }

    // Training status indicator
    if (isTraining) {
      ctx.fillStyle = '#4ade80';
      ctx.font = 'bold 11px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.fillText('⚡ TRAINING LIVE', W - 20, 18);
    }

    if (isTraining || particles.length > 0) {
      networkAnimFrame = requestAnimationFrame(render);
    }
  }

  render();
}

async function startTraining() {
  document.getElementById('btn-train').disabled = true;
  const res = await fetch(BASE + '/api/train', { method: 'POST' });
  const data = await res.json();
  if (data.run_id) {
    currentRunId = data.run_id;
    await fetchRuns();
    document.getElementById('run-select').value = data.run_id;
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(updateDashboard, 2000);
  }
}

async function stopTraining() {
  await fetch(BASE + '/api/stop', { method: 'POST' });
  document.getElementById('btn-stop').disabled = true;
}

// ── Auto-Optimizer ─────────────────────────────────────────
let optPollInterval = null;

async function startOptimize(n) {
  const res = await fetch(BASE + '/api/optimize?trials=' + n, { method: 'POST' });
  const data = await res.json();
  if (data.error) {
    document.getElementById('opt-status').textContent = data.error;
    return;
  }
  document.getElementById('opt-status').textContent = '⚡ Started — ' + n + ' trials';
  if (optPollInterval) clearInterval(optPollInterval);
  optPollInterval = setInterval(pollOptimizer, 2000);
}

async function pollOptimizer() {
  try {
    const res = await fetch(BASE + '/api/optimize/status');
    const s = await res.json();
    if (!s.completed_trials) return;

    const pct = Math.round((s.completed_trials / s.total_trials) * 100);
    document.getElementById('opt-progress').style.width = pct + '%';
    document.getElementById('opt-progress').style.background = s.status === 'complete'
      ? 'linear-gradient(90deg, #4ade80, #22c55e)'
      : 'linear-gradient(90deg, #8b5cf6, #a78bfa)';

    document.getElementById('opt-status').textContent = s.status === 'complete'
      ? `✅ Complete — Best: ${(s.best_accuracy * 100).toFixed(1)}%`
      : `Trial ${s.completed_trials}/${s.total_trials} | Best: ${(s.best_accuracy * 100).toFixed(1)}% | ~${Math.round(s.est_remaining_seconds)}s left`;

    // Accuracy over trials chart
    if (s.accuracy_over_time && s.accuracy_over_time.length > 0) {
      const trials = s.accuracy_over_time.map((_, i) => i + 1);
      const bestSoFar = s.accuracy_over_time.map((_, i) =>
        Math.max(...s.accuracy_over_time.slice(0, i + 1))
      );
      Plotly.react('opt-chart', [
        { x: trials, y: s.accuracy_over_time.map(a => a * 100), name: 'Trial Accuracy', mode: 'markers', marker: { color: '#6366f1', size: 4, opacity: 0.5 } },
        { x: trials, y: bestSoFar.map(a => a * 100), name: 'Best So Far', line: { color: '#4ade80', width: 2 } }
      ], {
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#888', size: 10 },
        margin: { t: 10, b: 30, l: 50, r: 10 },
        xaxis: { title: 'Trial #', gridcolor: '#1e1e2e' },
        yaxis: { title: 'Accuracy %', gridcolor: '#1e1e2e' },
        legend: { x: 0.6, y: 0.1 }
      }, { responsive: true });
    }

    // Best params info
    if (s.best_params) {
      const bp = s.best_params;
      const layers = [];
      for (let i = 0; i < (bp.n_layers || 0); i++) {
        if (bp['hidden_' + i]) layers.push(bp['hidden_' + i]);
      }
      document.getElementById('opt-info').innerHTML = `
        <div style="color:#4ade80;font-weight:bold;font-size:1.1em;margin-bottom:8px;">
          Best: ${(s.best_accuracy * 100).toFixed(1)}%
        </div>
        <div style="color:#a78bfa;margin-bottom:4px;">🏆 Best Architecture</div>
        <div style="color:#ccc;">Layers: ${layers.join(' → ') || '?'} → 3</div>
        <div style="color:#ccc;">Dropout: ${((bp.dropout || 0) * 100).toFixed(0)}%</div>
        <div style="color:#ccc;">LR: ${bp.lr ? bp.lr.toExponential(2) : '?'}</div>
        <div style="color:#ccc;">Batch: ${bp.batch_size || '?'}</div>
        <div style="color:#ccc;">Activation: ${bp.activation || '?'}</div>
        <div style="color:#ccc;">Weight Decay: ${bp.weight_decay ? bp.weight_decay.toExponential(2) : '?'}</div>
        <div style="margin-top:8px;color:#888;font-size:0.85em;">
          ${s.completed_trials} complete · ${s.pruned_trials || 0} pruned<br>
          Avg ${s.avg_trial_seconds}s/trial · ${Math.round(s.elapsed_seconds)}s total
        </div>
      `;
    }

    // Tree baselines
    if (s.tree_baselines) {
      const tb = s.tree_baselines;
      const infoDiv = document.getElementById('opt-info');
      let treeHtml = '<div style="margin-top:12px;border-top:1px solid #1e1e2e;padding-top:8px;">';
      treeHtml += '<div style="color:#f97316;margin-bottom:4px;">🌲 Tree Baselines (instant)</div>';
      for (const [name, acc] of Object.entries(tb)) {
        treeHtml += `<div style="color:#ccc;">${name}: <span style="color:#4ade80;font-weight:bold;">${(acc * 100).toFixed(1)}%</span></div>`;
      }
      treeHtml += '</div>';
      infoDiv.innerHTML += treeHtml;
    }

    if (s.status === 'complete') {
      clearInterval(optPollInterval);
      optPollInterval = null;
    }
  } catch (err) {
    console.error('Optimizer poll error:', err);
  }
}

// Check if optimizer is already running on page load
setTimeout(async () => {
  try {
    const res = await fetch(BASE + '/api/optimize/status');
    const s = await res.json();
    if (s.status === 'running') {
      if (!optPollInterval) optPollInterval = setInterval(pollOptimizer, 2000);
      pollOptimizer();
    } else if (s.status === 'complete') {
      pollOptimizer(); // Show final results
    }
  } catch(e) {}
}, 1000);

// ── Live Inference ──────────────────────────────────────────
let autoInfInterval = null;
let infAnimFrame = null;

async function runInference() {
  // Find the latest complete run
  const runsRes = await fetch(BASE + '/api/runs');
  const runsData = await runsRes.json();
  const completeRun = runsData.runs.find(r => r.status === 'complete');
  if (!completeRun) {
    document.getElementById('inf-status').textContent = 'No trained model yet';
    return;
  }

  const res = await fetch(BASE + '/api/inference/' + completeRun.run_id);
  const data = await res.json();
  if (data.error) {
    document.getElementById('inf-status').textContent = data.error;
    return;
  }

  document.getElementById('inf-status').textContent = data.ticker + ' — ' + (data.correct ? '✅ Correct' : '❌ Wrong');

  // Render input panel
  const inputDiv = document.getElementById('inf-inputs');
  const entries = Object.entries(data.inputs);
  inputDiv.innerHTML = `<div style="color:#60a5fa;font-weight:bold;margin-bottom:6px;font-size:0.9em;">${data.ticker}</div>` +
    entries.map(([k, v]) => {
      const scaled = data.scaled_inputs[k] || 0;
      const intensity = Math.min(Math.abs(scaled) / 3, 1);
      const color = scaled > 0 ? `rgba(74,222,128,${0.2 + intensity * 0.6})` : `rgba(239,68,68,${0.2 + intensity * 0.6})`;
      return `<div style="display:flex;justify-content:space-between;padding:2px 4px;font-size:0.7em;background:${color};margin:1px 0;border-radius:3px;">
        <span style="color:#ccc;">${k}</span>
        <span style="color:#fff;font-weight:bold;">${v}</span>
      </div>`;
    }).join('');

  // Render output panel
  const outDiv = document.getElementById('inf-output');
  const probs = data.output_probabilities;
  const predColor = data.prediction === 'BUY' ? '#4ade80' : data.prediction === 'SELL' ? '#ef4444' : '#888';
  outDiv.innerHTML = `
    <div style="text-align:center;margin-bottom:16px;">
      <div style="font-size:2em;font-weight:bold;color:${predColor};">${data.prediction}</div>
      <div style="font-size:0.75em;color:#888;">Predicted</div>
      <div style="font-size:0.8em;color:${data.correct ? '#4ade80' : '#ef4444'};margin-top:4px;">
        Actual: ${data.true_label} ${data.correct ? '✅' : '❌'}
      </div>
    </div>
    <div style="margin-top:12px;">
      ${['BUY', 'NEUTRAL', 'SELL'].map(label => {
        const pct = probs[label] || 0;
        const barColor = label === 'BUY' ? '#4ade80' : label === 'SELL' ? '#ef4444' : '#666';
        return `<div style="margin:6px 0;">
          <div style="display:flex;justify-content:space-between;font-size:0.75em;">
            <span>${label}</span><span>${pct.toFixed(1)}%</span>
          </div>
          <div style="background:#1e1e2e;border-radius:4px;height:8px;overflow:hidden;">
            <div style="background:${barColor};height:100%;width:${pct}%;transition:width 0.5s;"></div>
          </div>
        </div>`;
      }).join('')}
    </div>
    <div style="margin-top:16px;font-size:0.7em;color:#555;">
      ${data.layer_activations.map(l =>
        `Layer ${l.layer + 1}: ${l.firing}/${l.total} neurons firing`
      ).join('<br>')}
    </div>
  `;

  // Animate the inference through the network
  animateInference(data);
}

function animateInference(data) {
  const container = document.getElementById('inf-network');
  let cvs = container.querySelector('canvas');
  const W = container.clientWidth || 600;
  const H = 400;
  if (!cvs) {
    container.innerHTML = '';
    cvs = document.createElement('canvas');
    cvs.width = W * 2; cvs.height = H * 2;
    cvs.style.width = W + 'px'; cvs.style.height = H + 'px';
    container.appendChild(cvs);
  }
  const ctx = cvs.getContext('2d');
  ctx.setTransform(2, 0, 0, 2, 0, 0);

  // Build layers: input features → hidden layers → output
  const inputSize = Object.keys(data.inputs).length;
  const layers = [
    { size: inputSize, activations: Object.values(data.scaled_inputs).map(v => Math.min(Math.abs(v) / 3, 1)) }
  ];
  data.layer_activations.forEach(l => {
    const maxAct = l.max || 1;
    layers.push({
      size: l.total,
      activations: l.activations.map(a => Math.min(a / maxAct, 1))
    });
  });
  // Output layer
  const outProbs = [data.output_probabilities.SELL || 0, data.output_probabilities.NEUTRAL || 0, data.output_probabilities.BUY || 0];
  layers.push({ size: 3, activations: outProbs.map(p => p / 100) });

  const maxDisplay = 12;
  const layerX = layers.map((_, i) => 50 + (W - 100) * i / (layers.length - 1));

  // Precompute node positions
  const nodeMap = layers.map((layer, li) => {
    const display = Math.min(layer.size, maxDisplay);
    const spacing = Math.min(28, (H - 60) / display);
    const startY = (H - spacing * (display - 1)) / 2;
    const nodes = [];
    for (let ni = 0; ni < display; ni++) {
      nodes.push({
        x: layerX[li], y: startY + ni * spacing,
        act: layer.activations[ni] || 0
      });
    }
    return { nodes, totalSize: layer.size };
  });

  // Particles for animation
  let flowParticles = [];
  let frame = 0;
  const totalFrames = 180; // 3 seconds at 60fps

  if (infAnimFrame) cancelAnimationFrame(infAnimFrame);

  function render() {
    ctx.clearRect(0, 0, W, H);
    frame++;

    // Which layer boundary is currently "active" (data flowing through)
    const activeLayer = Math.floor((frame / totalFrames) * (layers.length - 1));

    // Spawn particles at the active boundary
    if (frame < totalFrames && activeLayer < nodeMap.length - 1) {
      for (let i = 0; i < 3; i++) {
        const from = nodeMap[activeLayer].nodes;
        const to = nodeMap[activeLayer + 1].nodes;
        const fi = Math.floor(Math.random() * from.length);
        const ti = Math.floor(Math.random() * to.length);
        flowParticles.push({
          x: from[fi].x, y: from[fi].y,
          tx: to[ti].x, ty: to[ti].y,
          progress: 0, speed: 0.03 + Math.random() * 0.02,
          act: to[ti].act
        });
      }
    }

    // Draw connections
    for (let li = 0; li < nodeMap.length - 1; li++) {
      const from = nodeMap[li].nodes;
      const to = nodeMap[li + 1].nodes;
      const layerActive = li <= activeLayer;
      for (let fi = 0; fi < from.length; fi++) {
        for (let ti = 0; ti < to.length; ti++) {
          const strength = (from[fi].act + to[ti].act) / 2;
          ctx.beginPath();
          ctx.moveTo(from[fi].x, from[fi].y);
          ctx.lineTo(to[ti].x, to[ti].y);
          const baseAlpha = layerActive ? 0.05 + strength * 0.25 : 0.02;
          ctx.strokeStyle = `rgba(99, 102, 241, ${baseAlpha})`;
          ctx.lineWidth = layerActive ? 0.5 + strength * 2 : 0.3;
          ctx.stroke();
        }
      }
    }

    // Draw particles
    flowParticles = flowParticles.filter(p => p.progress < 1);
    for (const p of flowParticles) {
      p.progress += p.speed;
      const cx = p.x + (p.tx - p.x) * p.progress;
      const cy = p.y + (p.ty - p.y) * p.progress;
      const alpha = Math.sin(p.progress * Math.PI);
      ctx.beginPath();
      ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(74, 222, 128, ${alpha * 0.9})`;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(cx, cy, 8, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(74, 222, 128, ${alpha * 0.15})`;
      ctx.fill();
    }

    // Draw nodes
    for (let li = 0; li < nodeMap.length; li++) {
      const layer = nodeMap[li];
      const layerLit = li <= activeLayer;

      for (const node of layer.nodes) {
        const act = layerLit ? node.act : 0;
        const radius = 6 + act * 6;

        // Glow
        if (act > 0.2 && layerLit) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius + 8, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(99, 102, 241, ${act * 0.2})`;
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        const r = Math.round(40 + act * 215);
        const g = Math.round(40 + act * 62);
        const b = Math.round(60 + act * 181);
        ctx.fillStyle = layerLit ? `rgb(${r}, ${g}, ${b})` : '#1a1a2e';
        ctx.fill();
        ctx.strokeStyle = layerLit ? `rgba(163, 139, 250, ${0.4 + act * 0.5})` : '#222';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Labels
      const labels = ['Input', ...data.layer_activations.map((_, i) => `Hidden ${i + 1}`), 'Output'];
      ctx.fillStyle = li <= activeLayer ? '#a78bfa' : '#333';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(labels[li] || '', layerX[li], 15);

      // Output labels
      if (li === nodeMap.length - 1 && layerLit) {
        const outLabels = ['SELL', 'NEUTRAL', 'BUY'];
        const outColors = ['#ef4444', '#888', '#4ade80'];
        layer.nodes.forEach((node, ni) => {
          ctx.fillStyle = outColors[ni];
          ctx.font = 'bold 10px JetBrains Mono, monospace';
          ctx.textAlign = 'left';
          ctx.fillText(`${outLabels[ni]} ${(outProbs[ni]).toFixed(0)}%`, node.x + 14, node.y + 4);
        });
      }
    }

    // Progress bar at bottom
    const progress = Math.min(frame / totalFrames, 1);
    ctx.fillStyle = '#1e1e2e';
    ctx.fillRect(30, H - 12, W - 60, 4);
    ctx.fillStyle = '#6366f1';
    ctx.fillRect(30, H - 12, (W - 60) * progress, 4);

    if (frame < totalFrames + 60 || flowParticles.length > 0) {
      infAnimFrame = requestAnimationFrame(render);
    }
  }

  render();
}

let autoRunning = false;
function autoInference() {
  if (autoRunning) {
    clearInterval(autoInfInterval);
    autoRunning = false;
    document.getElementById('inf-status').textContent = 'Auto stopped';
    return;
  }
  autoRunning = true;
  document.getElementById('inf-status').textContent = 'Auto mode ⚡';
  runInference();
  autoInfInterval = setInterval(runInference, 5000);
}

// Init
fetchRuns();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/runs")
def api_runs():
    """List all training runs."""
    runs = []
    for f in sorted(glob.glob(os.path.join(MODELS_DIR, "history_*.json")), reverse=True):
        try:
            with open(f) as fh:
                h = json.load(fh)
            runs.append({
                "run_id": h["run_id"],
                "status": h["status"],
                "started_at": h.get("started_at"),
                "best_accuracy": h.get("best_test_accuracy"),
                "epochs": len(h.get("epochs_data", []))
            })
        except:
            pass
    return jsonify({"runs": runs})


@app.route("/api/history/<run_id>")
def api_history(run_id):
    """Get training history for a run."""
    path = os.path.join(MODELS_DIR, f"history_{run_id}.json")
    if not os.path.exists(path):
        return jsonify({"error": "Run not found"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/train", methods=["POST"])
def api_train():
    """Start a new training run."""
    global training_process

    if training_process["proc"] and training_process["proc"].poll() is None:
        return jsonify({"error": "Training already in progress"}), 409

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check if data exists, fetch if not
    data_path = os.path.join(DATA_DIR, "training_data.csv")
    if not os.path.exists(data_path):
        # Fetch data first
        fetch_proc = subprocess.Popen(
            [sys.executable, os.path.join(BASE_DIR, "fetch_data.py")],
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        fetch_proc.wait()

    # Start training in background — more epochs, patience for live viewing
    train_env = os.environ.copy()
    train_env["RUN_ID"] = run_id
    train_env["EPOCHS"] = "50"
    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.join(BASE_DIR, "train.py")],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=train_env
    )

    training_process = {"proc": proc, "run_id": run_id}

    # Log subprocess output to a file for debugging
    def log_output(process, rid):
        log_path = os.path.join(MODELS_DIR, f"log_{rid}.txt")
        with open(log_path, "w") as lf:
            for line in process.stdout:
                lf.write(line.decode("utf-8", errors="replace"))
                lf.flush()
            process.wait()
            lf.write(f"\n--- Exit code: {process.returncode} ---\n")
    threading.Thread(target=log_output, args=(proc, run_id), daemon=True).start()

    return jsonify({"run_id": run_id, "status": "started"})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Stop current training."""
    global training_process
    if training_process["proc"] and training_process["proc"].poll() is None:
        training_process["proc"].terminate()
        return jsonify({"status": "stopped"})
    return jsonify({"status": "no training running"})


@app.route("/api/data-status")
def api_data_status():
    """Check training data status."""
    meta_path = os.path.join(DATA_DIR, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return jsonify(json.load(f))
    return jsonify({"status": "no data", "message": "Run fetch_data.py first"})


@app.route("/api/log/<run_id>")
def api_log(run_id):
    """Get subprocess log for debugging."""
    log_path = os.path.join(MODELS_DIR, f"log_{run_id}.txt")
    if not os.path.exists(log_path):
        return jsonify({"error": "No log found", "hint": "Training may not have started. Check that all dependencies are installed: pip install -r requirements.txt"})
    with open(log_path) as f:
        return jsonify({"log": f.read()})


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    """Start hyperparameter optimization."""
    global training_process
    if training_process["proc"] and training_process["proc"].poll() is None:
        return jsonify({"error": "Training/optimization already in progress"}), 409

    study_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_trials = request.args.get("trials", 200, type=int)

    opt_env = os.environ.copy()
    opt_env["STUDY_ID"] = study_id

    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.join(BASE_DIR, "auto_optimize.py"), str(n_trials)],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=opt_env
    )
    training_process = {"proc": proc, "run_id": f"opt_{study_id}"}
    return jsonify({"study_id": study_id, "n_trials": n_trials, "status": "started"})


@app.route("/api/optimize/status")
def api_optimize_status():
    """Get latest optimization status."""
    opt_dir = os.path.join(BASE_DIR, "optimize")
    files = sorted(glob.glob(os.path.join(opt_dir, "status_*.json")), reverse=True)
    if not files:
        return jsonify({"status": "no optimization runs"})
    with open(files[0]) as f:
        return jsonify(json.load(f))


@app.route("/api/inference/<run_id>")
def api_inference(run_id):
    """Run inference on a random sample and return full layer activations."""
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    import pickle

    # Load model
    model_path = os.path.join(MODELS_DIR, f"model_{run_id}.pt")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{run_id}.pkl")
    data_path = os.path.join(DATA_DIR, "training_data.csv")

    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found"}), 404

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    features = checkpoint["features"]
    config = checkpoint["config"]

    # Rebuild model
    from train import DirectionNet
    model = DirectionNet(
        input_size=config["input_size"],
        hidden_sizes=config["hidden_sizes"],
        dropout=config["dropout"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Load scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Load data and pick random sample
    df = pd.read_csv(data_path)
    sample_idx = request.args.get("idx", None)
    if sample_idx is not None:
        sample_idx = int(sample_idx)
    else:
        valid = df.dropna(subset=features + ["label", "ticker"])
        sample_idx = valid.sample(1).index[0]

    row = df.iloc[sample_idx]
    ticker = row.get("ticker", "???")
    true_label = int(row.get("label", 0))

    # Get input values
    raw_inputs = {}
    for f in features:
        val = row.get(f, 0)
        raw_inputs[f] = round(float(val), 4) if pd.notna(val) else 0.0

    # Scale and run inference
    X = np.array([[raw_inputs[f] for f in features]])
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)

    # Get activations at each layer
    layer_activations = []
    current = X_tensor
    layer_idx = 0
    with torch.no_grad():
        for child in model.net:
            current = child(current)
            if isinstance(child, nn.ReLU):
                acts = current.squeeze().numpy()
                layer_activations.append({
                    "layer": layer_idx,
                    "size": len(acts),
                    "activations": [round(float(a), 4) for a in acts],
                    "max": round(float(acts.max()), 4),
                    "mean": round(float(acts.mean()), 4),
                    "firing": int((acts > 0).sum()),
                    "total": len(acts)
                })
                layer_idx += 1

        # Final output (softmax)
        output_raw = current.squeeze().numpy()
        probs = np.exp(output_raw) / np.exp(output_raw).sum()

    labels = ["SELL", "NEUTRAL", "BUY"]
    prediction = labels[int(probs.argmax())]
    true_label_name = labels[true_label + 1]  # -1,0,1 → 0,1,2

    return jsonify({
        "ticker": ticker,
        "sample_idx": int(sample_idx),
        "inputs": raw_inputs,
        "scaled_inputs": {f: round(float(v), 4) for f, v in zip(features, X_scaled[0])},
        "layer_activations": layer_activations,
        "output_probabilities": {l: round(float(p) * 100, 1) for l, p in zip(labels, probs)},
        "prediction": prediction,
        "true_label": true_label_name,
        "correct": prediction == true_label_name
    })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="Host (use 0.0.0.0 for LAN access)")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"🧠 Neural Net Dashboard → http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
