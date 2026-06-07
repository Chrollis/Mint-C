<template>
  <canvas ref="canvasRef" class="bg-canvas"></canvas>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from "vue";

const props = defineProps({
  color: { type: String, default: "#ffffff" },
});

const canvasRef = ref(null);
let ctx = null;
let animationId = null;
let particles = [];
let resizeTimeout;

let cachedLayout = [];
let cachedFontSize = 0;
let cachedCanvasWidth = 0;
let cachedCanvasHeight = 0;

const asciiArt = String.raw`
__/\\\\____________/\\\\________________________________________________________/\\\\\\\\\_
 _\/\\\\\\________/\\\\\\_____________________________________________________/\\\////////__
  _\/\\\//\\\____/\\\//\\\__/\\\___________________/\\\______________________/\\\/___________
   _\/\\\\///\\\/\\\/_\/\\\_\///___/\\/\\\\\\____/\\\\\\\\\\\__/\\\\\\\\\\\__/\\\_____________
    _\/\\\__\///\\\/___\/\\\__/\\\_\/\\\////\\\__\////\\\////__\///////////__\/\\\_____________
     _\/\\\____\///_____\/\\\_\/\\\_\/\\\__\//\\\____\/\\\____________________\//\\\____________
      _\/\\\_____________\/\\\_\/\\\_\/\\\___\/\\\____\/\\\_/\\_________________\///\\\__________
       _\/\\\_____________\/\\\_\/\\\_\/\\\___\/\\\____\//\\\\\____________________\////\\\\\\\\\_
        _\///______________\///__\///__\///____\///______\/////________________________\/////////__
`;

const asciiLines = asciiArt.split("\n").filter(line => line.trim().length > 0);

let currentAsciiFontSize = 16;
let targetAsciiFontSize = 16;
const ASCII_BASE_FONT_SIZE = 16;
const ASCII_HOVER_FONT_SIZE = 24;
const WAVE_AMPLITUDE = 2;
const WAVE_FREQUENCY = 2.5;

const PARTICLE_COUNT = 280;
const ATTRACT_RADIUS = 150;
const ATTRACT_FORCE = 0.006;
const RANDOM_FORCE = 0.25;
const DAMP = 0.98;
const MAX_SPEED = 1.2;
const MAX_OFFSET = 80;

class Particle {
  constructor(width, height) {
    this.homeX = Math.random() * width;
    this.homeY = Math.random() * height;
    this.x = this.homeX;
    this.y = this.homeY;
    this.vx = (Math.random() - 0.5) * 1.2;
    this.vy = (Math.random() - 0.5) * 1.2;
    this.size = Math.random() * 2.5 + 1.2;
  }

  update(targetX, targetY, width, height) {
    if (targetX !== null && targetY !== null) {
      const dx = targetX - this.x;
      const dy = targetY - this.y;
      const dist = Math.hypot(dx, dy);
      if (dist < ATTRACT_RADIUS) {
        const strength = (1 - dist / ATTRACT_RADIUS) * ATTRACT_FORCE;
        this.vx += dx * strength;
        this.vy += dy * strength;
      }
    }
    this.vx += (Math.random() - 0.5) * RANDOM_FORCE;
    this.vy += (Math.random() - 0.5) * RANDOM_FORCE;
    this.vx *= DAMP;
    this.vy *= DAMP;
    let speed = Math.hypot(this.vx, this.vy);
    if (speed > MAX_SPEED) {
      this.vx = (this.vx / speed) * MAX_SPEED;
      this.vy = (this.vy / speed) * MAX_SPEED;
    }
    let newX = this.x + this.vx;
    let newY = this.y + this.vy;
    const dxHome = newX - this.homeX;
    const dyHome = newY - this.homeY;
    const offset = Math.hypot(dxHome, dyHome);
    if (offset > MAX_OFFSET) {
      const ratio = MAX_OFFSET / offset;
      newX = this.homeX + dxHome * ratio;
      newY = this.homeY + dyHome * ratio;
      const normX = dxHome / offset;
      const normY = dyHome / offset;
      const vDotN = this.vx * normX + this.vy * normY;
      if (vDotN > 0) {
        this.vx -= vDotN * normX * 0.5;
        this.vy -= vDotN * normY * 0.5;
      }
    }
    this.x = Math.min(Math.max(newX, 0), width);
    this.y = Math.min(Math.max(newY, 0), height);
  }

  draw(ctx, color) {
    const gradient = ctx.createRadialGradient(this.x, this.y, this.size * 0.2, this.x, this.y, this.size);
    gradient.addColorStop(0, color + "A0");
    gradient.addColorStop(1, color + "20");

    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
    ctx.fillStyle = gradient;
    ctx.fill();
  }
}

function initParticles(width, height) {
  particles = [];
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    particles.push(new Particle(width, height));
  }
}

function buildAsciiLayout(fontSize, canvasWidth, canvasHeight) {
  if (!ctx) return [];
  const fontFamily = '"Fira Code", "LXGW WenKai Mono Screen", monospace';
  ctx.save();
  ctx.font = `${fontSize}px ${fontFamily}`;
  const lineHeight = fontSize * 1.2;
  const totalHeight = asciiLines.length * lineHeight;
  const startY = (canvasHeight - totalHeight) / 2;
  const layout = [];

  for (let lineIdx = 0; lineIdx < asciiLines.length; lineIdx++) {
    const line = asciiLines[lineIdx];
    const chars = line.split('');
    let charWidths = [];
    let totalWidth = 0;
    for (let i = 0; i < chars.length; i++) {
      const w = ctx.measureText(chars[i]).width;
      charWidths.push(w);
      totalWidth += w;
    }
    let startX = (canvasWidth - totalWidth) / 2;
    let currentX = startX;
    const baseY = startY + lineIdx * lineHeight;

    for (let i = 0; i < chars.length; i++) {
      layout.push({
        char: chars[i],
        x: currentX,
        baseY: baseY,
        width: charWidths[i]
      });
      currentX += charWidths[i];
    }
  }
  ctx.restore();
  return layout;
}

function resizeCanvas() {
  if (!canvasRef.value) return;
  const canvas = canvasRef.value;
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  if (ctx) initParticles(canvas.width, canvas.height);
}

function handleResize() {
  if (resizeTimeout) clearTimeout(resizeTimeout);
  resizeTimeout = setTimeout(() => {
    if (!canvasRef.value) return;
    resizeCanvas();
    cachedLayout = [];
    cachedFontSize = 0;
  }, 150);
}

function drawAsciiArt() {
  if (!ctx || !canvasRef.value) return;
  const canvas = canvasRef.value;
  const w = canvas.width;
  const h = canvas.height;

  currentAsciiFontSize = currentAsciiFontSize * 0.92 + targetAsciiFontSize * 0.08;
  const fontSize = Math.round(currentAsciiFontSize);

  if (cachedLayout.length === 0 || cachedFontSize !== fontSize || cachedCanvasWidth !== w || cachedCanvasHeight !== h) {
    cachedLayout = buildAsciiLayout(fontSize, w, h);
    cachedFontSize = fontSize;
    cachedCanvasWidth = w;
    cachedCanvasHeight = h;
  }

  ctx.save();
  const fontFamily = '"Fira Code", "LXGW WenKai Mono Screen", monospace';
  ctx.font = `${fontSize}px ${fontFamily}`;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillStyle = props.color + "80";
  ctx.shadowBlur = 0;

  const now = Date.now() / 1000;
  const amplitude = WAVE_AMPLITUDE;

  for (const info of cachedLayout) {
    const phase = (info.x * 0.02) + now * WAVE_FREQUENCY;
    const offsetY = Math.sin(phase) * amplitude;
    ctx.fillText(info.char, info.x, info.baseY + offsetY);
  }
  ctx.restore();
}

function animate() {
  if (!ctx || !canvasRef.value) return;
  const canvas = canvasRef.value;
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  drawAsciiArt();

  for (let p of particles) {
    p.update(null, null, width, height);
    p.draw(ctx, props.color);
  }

  animationId = requestAnimationFrame(animate);
}

onMounted(() => {
  if (!canvasRef.value) return;
  ctx = canvasRef.value.getContext("2d");
  resizeCanvas();
  animate();
  window.addEventListener("resize", handleResize);
});

onUnmounted(() => {
  if (animationId) cancelAnimationFrame(animationId);
  window.removeEventListener("resize", handleResize);
});
</script>

<style scoped>
.bg-canvas {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 0;
}
</style>