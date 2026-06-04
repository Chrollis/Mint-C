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
  <canvas ref="cursorCanvas" class="global-cursor"></canvas>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from "vue";

const props = defineProps({
  color: { type: String, default: "#ffffff" },
  showSplash: { type: Boolean, default: true },
  splashPolygon: { type: Array, default: () => [] },
});

const cursorCanvas = ref(null);
let ctx = null;
let animationId = null;
let resizeTimeout;

const debouncedResize = () => {
  if (resizeTimeout) clearTimeout(resizeTimeout);
  resizeTimeout = setTimeout(() => {
    if (!cursorCanvas.value) return;
    const canvas = cursorCanvas.value;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }, 150);
};

let mouseX = 0,
  mouseY = 0;

let isHoverSplashCube = false;

let isHoverInteractive = false;

let frameRotation = 0;

let smallBoxRotation = 0;

let currentBoxSize = 28;

let targetBoxSize = 28;

const SIZE_LERP_SPEED = 0.15;

let breatheTime = 0;

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

function onGlobalMouseMove(e) {
  mouseX = e.clientX;
  mouseY = e.clientY;
  if (!props.showSplash) {
    const el = e.target;
    const isClickable =
      el.tagName === "A" ||
      el.tagName === "BUTTON" ||
      el.closest?.(
        "button, a, [role='button'], .v-btn, .v-list-item," +
        " .v-card-actions button, .v-btn--icon, .v-btn--fab," +
        " .canvas-container, .v-input,.brush-preview, .model-status",
      ) ||
      window.getComputedStyle(el).cursor === "pointer";
    isHoverInteractive = isClickable;
  } else {
    isHoverSplashCube =
      props.splashPolygon.length > 0 &&
      pointInPolygon(mouseX, mouseY, props.splashPolygon);
  }
}

function drawCursor(x, y, w, h) {
  if (!ctx) return;
  const breath = breatheTime * 2.5;
  const amp = 3.5 * Math.sin(breath);
  const gapFromBox = 4;
  const rayOffset = (gapFromBox + amp) * 1.3 + 10;
  const diagOffset = amp * 0.7;

  let hoverActive = false;
  if (props.showSplash) {
    hoverActive = isHoverSplashCube;
  } else {
    hoverActive = isHoverInteractive;
  }

  targetBoxSize = hoverActive ? 38 : 28;
  currentBoxSize =
    currentBoxSize * (1 - SIZE_LERP_SPEED) + targetBoxSize * SIZE_LERP_SPEED;
  const boxSize = currentBoxSize;
  const half = boxSize / 2;

  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = props.color + "D0";
  ctx.shadowBlur = 5;
  ctx.shadowColor = props.color + "80";

  const leftX = x - half - rayOffset;
  const rightX = x + half + rayOffset;
  const topY = y - half - rayOffset;
  const bottomY = y + half + rayOffset;
  ctx.beginPath();
  ctx.moveTo(leftX, y);
  ctx.lineTo(0, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(rightX, y);
  ctx.lineTo(w, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, topY);
  ctx.lineTo(x, 0);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, bottomY);
  ctx.lineTo(x, h);
  ctx.stroke();

  ctx.save();
  ctx.translate(x, y);
  ctx.rotate((frameRotation * Math.PI) / 180);
  const cornerLen = 12;
  const halfRot = boxSize / 2;
  ctx.beginPath();
  ctx.moveTo(-halfRot - diagOffset, -halfRot + cornerLen - diagOffset);
  ctx.lineTo(-halfRot - diagOffset, -halfRot - diagOffset);
  ctx.lineTo(-halfRot + cornerLen - diagOffset, -halfRot - diagOffset);
  ctx.moveTo(halfRot - cornerLen + diagOffset, -halfRot - diagOffset);
  ctx.lineTo(halfRot + diagOffset, -halfRot - diagOffset);
  ctx.lineTo(halfRot + diagOffset, -halfRot + cornerLen - diagOffset);
  ctx.moveTo(halfRot + diagOffset, halfRot - cornerLen + diagOffset);
  ctx.lineTo(halfRot + diagOffset, halfRot + diagOffset);
  ctx.lineTo(halfRot - cornerLen + diagOffset, halfRot + diagOffset);
  ctx.moveTo(-halfRot + cornerLen - diagOffset, halfRot + diagOffset);
  ctx.lineTo(-halfRot - diagOffset, halfRot + diagOffset);
  ctx.lineTo(-halfRot - diagOffset, halfRot - cornerLen + diagOffset);
  ctx.stroke();
  ctx.restore();

  if (hoverActive) {
    const smallSize = 12;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate((smallBoxRotation * Math.PI) / 180);
    ctx.strokeStyle = props.color + "A0";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(-smallSize / 2, -smallSize / 2, smallSize, smallSize);
    ctx.restore();
  }
  ctx.shadowBlur = 0;
  ctx.restore();
}

function resizeCanvasImmediate() {
  if (!cursorCanvas.value) return;
  const canvas = cursorCanvas.value;
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}

function animateCursor() {
  if (!ctx || !cursorCanvas.value) return;
  const canvas = cursorCanvas.value;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  breatheTime = Date.now() / 300;
  frameRotation = (frameRotation + 0.6) % 360;
  smallBoxRotation = (smallBoxRotation - 1.2) % 360;
  if (mouseX !== null && mouseY !== null) {
    drawCursor(mouseX, mouseY, canvas.width, canvas.height);
  }
  animationId = requestAnimationFrame(animateCursor);
}

onMounted(() => {
  if (!cursorCanvas.value) return;
  ctx = cursorCanvas.value.getContext("2d");
  resizeCanvasImmediate();
  animateCursor();
  window.addEventListener("resize", debouncedResize);
  document.addEventListener("mousemove", onGlobalMouseMove);
  document.body.style.cursor = "none";
});

onUnmounted(() => {
  if (animationId) cancelAnimationFrame(animationId);
  window.removeEventListener("resize", debouncedResize);
  document.removeEventListener("mousemove", onGlobalMouseMove);
  document.body.style.cursor = "";
});
</script>

<style scoped>
.global-cursor {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 99999;
}
</style>