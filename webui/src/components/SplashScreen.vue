<!--
  Mint-C - Web-based digit recognition tool with C++ CNN backend
  Copyright (C) 2026 Chrollis

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
-->

<template>
  <Transition name="fade" appear>
    <div v-if="visible" class="splash-overlay">
      <canvas ref="canvasRef" class="splash-canvas"></canvas>
    </div>
  </Transition>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from "vue";

const props = defineProps({
  visible: Boolean,
  color: { type: String, default: "#ffffff" },
});

const emit = defineEmits(["close", "updatePolygon"]);

const canvasRef = ref(null);
let ctx = null;
let animationId = null;
let mouseX = null,
  mouseY = null;
let latestProjectedPoints = [];
let resizeTimeout;

let angleRad = 0;
let isHoverCube = false;
let reverseDir = false;
let hoverStartTime = 0;
let currentScale = 1.0;
let targetScale = 1.0;

const orbits = [
  {
    name: "earth",
    radiusRatio: 0.286,
    speed: 0.012,
    dash: [2, 2],
    dotRadius: 2,
    color: null,
  },
  {
    name: "mars",
    radiusRatio: 0.429,
    speed: 0.009,
    dash: [],
    dotRadius: 2,
    color: null,
  },
  {
    name: "asteroid",
    radiusRatio: 1.0,
    speed: 0.006,
    dash: [2, 2],
    dotRadius: 0,
    color: null,
  },
  {
    name: "jupiter",
    radiusRatio: 1.428,
    speed: 0.004,
    dash: [],
    dotRadius: 4,
    color: null,
  },
];

const baseRadius = 50;
let precessionAngle = 0;
let tiltAngle = 1.25;
let orbitAngles = [0, 0, 0, 0];
let currentOrbitScale = 1.0;
let targetOrbitScale = 1.0;
const HOVER_ORBIT_SCALE = 4.2;

function resizeCanvas() {
  if (!canvasRef.value) return;
  const canvas = canvasRef.value;
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}

function handleResize() {
  if (resizeTimeout) clearTimeout(resizeTimeout);
  resizeTimeout = setTimeout(() => {
    if (!canvasRef.value) return;
    const canvas = canvasRef.value;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    if (ctx) resizeCanvas();
  }, 150);
}

function pointInPolygon(px, py, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x,
      yi = polygon[i].y;
    const xj = polygon[j].x,
      yj = polygon[j].y;
    const intersect =
      yi > py != yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function getConvexPolygon(points) {
  if (!points.length) return [];
  const center = points.reduce(
    (sum, p) => ({ x: sum.x + p.x, y: sum.y + p.y }),
    { x: 0, y: 0 },
  );
  center.x /= points.length;
  center.y /= points.length;
  return points
    .slice()
    .sort(
      (a, b) =>
        Math.atan2(a.y - center.y, a.x - center.x) -
        Math.atan2(b.y - center.y, b.x - center.x),
    );
}

function projectOrbitPoint(radius, angle, centerX, centerY) {
  const localX = Math.cos(angle) * radius;
  const localY = Math.sin(angle) * radius;
  const localZ = 0;
  const cosT = Math.cos(tiltAngle);
  const sinT = Math.sin(tiltAngle);
  let y1 = localY * cosT - localZ * sinT;
  let z1 = localY * sinT + localZ * cosT;
  let x1 = localX;
  const cosP = Math.cos(precessionAngle);
  const sinP = Math.sin(precessionAngle);
  const xRot = x1 * cosP + z1 * sinP;
  const zRot = -x1 * sinP + z1 * cosP;
  const yRot = y1;
  const cameraDistance = 230;
  const perspective = cameraDistance / (cameraDistance + zRot);
  const screenX = centerX + xRot * perspective;
  const screenY = centerY + yRot * perspective;
  return { x: screenX, y: screenY };
}

function drawOrbits() {
  if (!ctx) return;
  const w = canvasRef.value.width;
  const h = canvasRef.value.height;
  const centerX = w / 2;
  const centerY = h / 2 + 4;

  ctx.save();
  for (let idx = 0; idx < orbits.length; idx++) {
    const orbit = orbits[idx];
    const radius = baseRadius * orbit.radiusRatio * currentOrbitScale;
    const segments = 100;
    ctx.beginPath();
    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      const point = projectOrbitPoint(radius, angle, centerX, centerY);
      if (i === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    }
    ctx.closePath();
    ctx.strokeStyle = orbit.color || props.color + "80";
    ctx.lineWidth = 1.5;
    ctx.setLineDash(orbit.dash);
    ctx.stroke();

    if (orbit.dotRadius > 0) {
      const orbitAng = orbitAngles[idx];
      const dotPoint = projectOrbitPoint(radius, orbitAng, centerX, centerY);
      ctx.beginPath();
      ctx.arc(dotPoint.x, dotPoint.y, orbit.dotRadius, 0, Math.PI * 2);
      ctx.fillStyle = props.color + "cc";
      ctx.shadowBlur = 6;
      ctx.fill();
    }
    orbitAngles[idx] += orbit.speed;
    if (orbitAngles[idx] > Math.PI * 2) orbitAngles[idx] -= Math.PI * 2;
  }
  ctx.setLineDash([]);
  ctx.shadowBlur = 0;
  ctx.restore();
}

function drawRotatingCube() {
  if (!ctx) return [];
  const w = canvasRef.value.width;
  const h = canvasRef.value.height;
  const centerX = w / 2;
  const centerY = h / 2;
  const basePixelSize = 180;
  const pixelSize = basePixelSize * currentScale;

  const vertices = [
    { x: 0, y: 0, z: 0 },
    { x: 1, y: 0, z: 0 },
    { x: 1, y: 0, z: 1 },
    { x: 0, y: 0, z: 1 },
    { x: 0, y: 1, z: 0 },
    { x: 1, y: 1, z: 0 },
    { x: 1, y: 1, z: 1 },
    { x: 0, y: 1, z: 1 },
  ];
  const axis = { x: 1, y: 0, z: 1 };
  const len = Math.hypot(axis.x, axis.y, axis.z);
  const ax = axis.x / len,
    ay = axis.y / len,
    az = axis.z / len;
  const cosA = Math.cos(angleRad),
    sinA = Math.sin(angleRad);
  const oneMinusCos = 1 - cosA;
  const rot = {
    xx: cosA + ax * ax * oneMinusCos,
    xy: ax * ay * oneMinusCos - az * sinA,
    xz: ax * az * oneMinusCos + ay * sinA,
    yx: ay * ax * oneMinusCos + az * sinA,
    yy: cosA + ay * ay * oneMinusCos,
    yz: ay * az * oneMinusCos - ax * sinA,
    zx: az * ax * oneMinusCos - ay * sinA,
    zy: az * ay * oneMinusCos + ax * sinA,
    zz: cosA + az * az * oneMinusCos,
  };
  const projected = vertices.map((v) => {
    let x = v.x - 0.5,
      y = v.y - 0.5,
      z = v.z - 0.5;
    const rx = rot.xx * x + rot.xy * y + rot.xz * z;
    const ry = rot.yx * x + rot.yy * y + rot.yz * z;
    return { x: centerX + rx * pixelSize, y: centerY + ry * pixelSize };
  });

  const edges = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
  ];
  ctx.save();
  ctx.beginPath();
  ctx.lineWidth = 2;
  ctx.strokeStyle = isHoverCube ? props.color + "80" : props.color + "50";
  ctx.shadowBlur = 6;
  ctx.shadowColor = isHoverCube ? props.color + "50" : props.color + "20";
  for (let edge of edges) {
    ctx.moveTo(projected[edge[0]].x, projected[edge[0]].y);
    ctx.lineTo(projected[edge[1]].x, projected[edge[1]].y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.restore();
  return projected;
}

function animate() {
  if (!ctx || !canvasRef.value) return;
  const canvas = canvasRef.value;
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  const now = Date.now();
  const rotSpeed = isHoverCube ? 0.024 : 0.012;
  angleRad += reverseDir ? -rotSpeed : rotSpeed;

  if (isHoverCube) {
    if (hoverStartTime === 0) hoverStartTime = now;
    const t = ((now - hoverStartTime) / 1000) * 2;
    const sine = Math.sin(t);
    const amplitude = 2 + Math.cos(t / 2 + Math.PI / 4);
    targetScale = amplitude * (1 + sine) + 0.2;
  } else {
    hoverStartTime = 0;
    targetScale = 1.0;
  }
  currentScale = currentScale * 0.92 + targetScale * 0.08;

  precessionAngle = 1.3 * Math.sin(Date.now() * 0.0008);
  targetOrbitScale = isHoverCube ? HOVER_ORBIT_SCALE : 1.0;
  currentOrbitScale = currentOrbitScale * 0.92 + targetOrbitScale * 0.08;

  const projected = drawRotatingCube();
  latestProjectedPoints = projected;
  const polygon = getConvexPolygon(projected);
  emit("updatePolygon", polygon);
  drawOrbits();

  let hover = false;
  if (mouseX !== null && mouseY !== null && projected.length) {
    hover = pointInPolygon(mouseX, mouseY, polygon);
  }
  if (hover !== isHoverCube) {
    isHoverCube = hover;
    reverseDir = isHoverCube;
    if (isHoverCube) hoverStartTime = now;
  }

  animationId = requestAnimationFrame(animate);
}

function onMouseMove(e) {
  if (!canvasRef.value) return;
  const canvas = canvasRef.value;
  const rect = canvas.getBoundingClientRect();
  mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
  mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
}

function onClick(e) {
  if (!canvasRef.value || !latestProjectedPoints.length) return;
  const canvas = canvasRef.value;
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
  const my = (e.clientY - rect.top) * (canvas.height / rect.height);
  const poly = getConvexPolygon(latestProjectedPoints);
  if (pointInPolygon(mx, my, poly)) emit("close");
}

onMounted(() => {
  if (!canvasRef.value) return;
  ctx = canvasRef.value.getContext("2d");
  resizeCanvas();
  animate();
  canvasRef.value.addEventListener("mousemove", onMouseMove);
  canvasRef.value.addEventListener("click", onClick);
  window.addEventListener("resize", handleResize);
});

onUnmounted(() => {
  if (animationId) cancelAnimationFrame(animationId);
  window.removeEventListener("resize", handleResize);
});
</script>

<style scoped>
.splash-overlay {
  position: fixed;
  inset: 0;
  background-color: transparent;
  z-index: 9999;
  pointer-events: none;
}

.splash-canvas {
  position: absolute;
  inset: 0;
  display: block;
  pointer-events: auto;
}

.fade-enter-active,
.fade-leave-active {
  transition:
    opacity 1.6s cubic-bezier(0.2, 0.9, 0.4, 1.1),
    transform 1.6s,
    filter 1.6s;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: scale(0.96);
  filter: blur(4px);
}
</style>