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
  <v-row>
    <v-col cols="12" md="6">
      <v-card class="frosted-glass">
        <v-card-title>
          {{ $t("recognize.canvasTitle") }}
        </v-card-title>
        <v-divider></v-divider>
        <v-card-text class="text-center">
          <div class="canvas-container">
            <canvas ref="canvasRef" width="400" height="400" style="
                width: 100%;
                height: auto;
                border-radius: 16px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                touch-action: none;
              " @mousedown="startDrawing" @mousemove="draw" @mouseup="stopDrawing" @touchstart="startDrawing"
              @touchmove="draw" @touchend="stopDrawing"></canvas>
          </div>
        </v-card-text>
        <v-divider></v-divider>
        <v-card-actions>
          <div class="d-flex align-center ga-2">
            <v-btn icon="mdi-minus" variant="text" @click="decrementBrushSize"></v-btn>
            <v-menu offset-y>
              <template v-slot:activator="{ props }">
                <div v-bind="props" class="brush-preview" :style="{
                  width: brushSize + 'px',
                  height: brushSize + 'px',
                  borderRadius: '50%',
                  backgroundColor: brushColor,
                  border: '1px solid rgba(0,0,0,0.2)',
                  cursor: 'pointer',
                }"></div>
              </template>
              <v-card class="frosted-glass pa-2">
                <v-color-picker v-model="brushColor" mode="hexa" hide-mode-switch></v-color-picker>
              </v-card>
            </v-menu>
            <v-btn icon="mdi-plus" variant="text" @click="incrementBrushSize"></v-btn>
          </div>

          <v-spacer></v-spacer>

          <v-btn variant="text" @click="uploadImage" class="mr-2" prepend-icon="mdi-upload">
            {{ $t("common.upload") }}
          </v-btn>
          <v-btn variant="text" color="error" @click="clearCanvas" prepend-icon="mdi-brush-off" class="mr-2">
            {{ $t("common.clear") }}
          </v-btn>
          <v-btn variant="text" color="primary" @click="recognize" :loading="recognizing" prepend-icon="mdi-magnify">
            {{ $t("recognize.recognizeBtn") }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-col>

    <v-col cols="12" md="6">
      <v-card class="frosted-glass" style="height: 100%; display: flex; flex-direction: column">
        <v-card-title class="d-flex justify-space-between align-center">
          <span>{{ $t("recognize.resultTitle") }}</span>
          <div></div>
        </v-card-title>
        <v-divider></v-divider>
        <v-card-text class="pa-0" style="flex: 1; overflow-y: auto; max-height: 500px">
          <v-list v-if="recognizeStore.detections.length" class="transparent">
            <template v-for="(det, idx) in recognizeStore.detections" :key="idx">
              <v-list-item class="detection-item">
                <template v-slot:prepend>
                  <v-checkbox v-model="recognizeStore.selectedDetections" :value="det" hide-details
                    density="compact"></v-checkbox>
                </template>
                <div class="d-flex align-center w-100">
                  <div class="sample-image-wrapper ml-3 mr-3" style="width: 40px; height: 40px">
                    <img :src="det.mnist_base64" class="sample-image" />
                  </div>
                  <div class="flex-grow-1">
                    <div>
                      {{ $t("common.digit") }}
                      <strong>{{ det.digit }}</strong>
                      ({{ $t("recognize.confidenceLabel") }}
                      {{ (det.confidence * 100).toFixed(1) }}%)
                    </div>
                    <div class="text-grey" style="font-size: 75%">
                      {{
                        $t("recognize.positionLabel", {
                          x: det.bbox.x,
                          y: det.bbox.y,
                          w: det.bbox.w,
                          h: det.bbox.h,
                        })
                      }}
                    </div>
                  </div>
                  <div class="d-flex ga-2">
                    <v-btn size="x-small" variant="text" color="primary" icon="mdi-check" @click="confirmDetection(det)"
                      :title="$t('common.confirm')"></v-btn>
                    <v-btn size="x-small" variant="text" color="warning" icon="mdi-pencil"
                      @click="openRelabelDialog(det)" :title="$t('recognize.relabelTitle')"></v-btn>
                  </div>
                </div>
              </v-list-item>
              <v-divider v-if="idx < recognizeStore.detections.length - 1"></v-divider>
            </template>
          </v-list>
          <div v-else class="text-center pa-4 text-grey">
            {{ $t("recognize.noResultHint") }}
          </div>
        </v-card-text>
        <v-divider></v-divider>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn variant="text" prepend-icon="mdi-check-all" class="mr-2" @click="toggleSelectAll">
            {{ $t("recognize.selectAllBtn") }} </v-btn><v-btn variant="text" color="error"
            prepend-icon="mdi-delete-empty" @click="clearDetections">
            {{ $t("common.clear") }}
          </v-btn> </v-card-actions><v-divider></v-divider><v-card-actions><v-spacer></v-spacer>
          <v-btn color="primary" variant="text" prepend-icon="mdi-check"
            :disabled="recognizeStore.selectedDetections.length === 0" @click="batchAddToDataset" class="mr-2">
            {{ $t("recognize.addToPendingBtn") }}
          </v-btn>
          <v-btn color="warning" variant="text" prepend-icon="mdi-pencil"
            :disabled="recognizeStore.selectedDetections.length === 0" @click="openBatchRelabelDialog">
            {{ $t("recognize.batchRelabelBtn") }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-col>
  </v-row>

  <v-row class="mt-4">
    <v-col cols="12">
      <v-card class="frosted-glass">
        <v-card-title>
          {{ $t("recognize.pendingTitle") }}
        </v-card-title>
        <v-divider></v-divider>
        <v-card-text class="pa-0">
          <div v-if="recognizeStore.pendingSamples.length" class="sample-grid">
            <div v-for="(sample, idx) in recognizeStore.pendingSamples" :key="idx" class="sample-card">
              <div class="sample-image-wrapper" style="width: 80px; height: 80px">
                <img :src="sample.mnist_base64" class="sample-image" />
              </div>
              <div class="sample-info">
                <div class="sample-label-chip">
                  {{ $t("common.digit") }} {{ sample.label }}
                </div>
                <div class="sample-actions">
                  <v-btn icon="mdi-pencil" size="x-small" variant="text" color="warning"
                    @click="openSampleRelabelDialog(sample, idx)" />
                  <v-btn icon="mdi-delete" size="x-small" variant="text" color="error"
                    @click="openDeletePendingConfirm(idx)" />
                </div>
              </div>
            </div>
          </div>
          <div v-else class="text-center pa-4 text-grey">
            {{ $t("recognize.noPendingHint") }}
          </div>
        </v-card-text>
        <v-divider></v-divider>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn variant="text" color="error" :disabled="recognizeStore.pendingSamples.length === 0"
            @click="openClearPendingConfirm" prepend-icon="mdi-trash-can-outline" class="mr-2">
            {{ $t("common.clear") }}
          </v-btn>
          <v-btn variant="text" color="primary" :disabled="recognizeStore.pendingSamples.length === 0"
            @click="saveToDataset" prepend-icon="mdi-content-save">
            {{ $t("recognize.saveToDatasetBtn") }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-col>
  </v-row>

  <v-dialog v-model="relabelDialogVisible" max-width="400px">
    <v-card class="dialog-card">
      <v-card-title>{{ $t("recognize.relabelDialogTitle") }}</v-card-title>
      <v-divider />
      <v-card-text class="pt-4">
        <div class="mt-3">
          <NumericStepper v-model="newLabel" :label="$t('recognize.correctDigitLabel')" :min="0" :max="9" :step="1"
            readonly :hint="$t('recognize.currentDigitHint', { digit: relabelTarget?.digit })" persistent-hint />
        </div>
      </v-card-text>
      <v-card-actions>
        <v-spacer />
        <v-btn variant="text" @click="relabelDialogVisible = false">
          {{ $t("common.cancel") }}
        </v-btn>
        <v-btn color="primary" @click="applyRelabel">
          {{ $t("common.confirm") }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>

  <v-dialog v-model="batchRelabelDialogVisible" max-width="400px">
    <v-card class="dialog-card">
      <v-card-title>{{ $t("recognize.batchRelabelDialogTitle") }}</v-card-title>
      <v-divider />
      <v-card-text class="pt-4">
        <div class="mt-3">
          <NumericStepper v-model="batchNewLabel" :label="$t('recognize.correctDigitLabel')" :min="0" :max="9" :step="1"
            readonly :hint="$t('recognize.batchRelabelHint', { count: recognizeStore.selectedDetections.length })"
            persistent-hint />
        </div>
      </v-card-text>
      <v-card-actions>
        <v-spacer />
        <v-btn variant="text" @click="batchRelabelDialogVisible = false">
          {{ $t("common.cancel") }}
        </v-btn>
        <v-btn color="primary" @click="applyBatchRelabel">
          {{ $t("common.confirm") }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>

  <v-dialog v-model="saveDatasetDialogVisible" max-width="500px">
    <v-card class="dialog-card">
      <v-card-title>{{ $t("recognize.saveDatasetDialogTitle") }}</v-card-title>
      <v-divider />
      <v-card-text class="pt-4">
        <v-text-field v-model="datasetName" :label="$t('recognize.datasetNameLabel')"
          :placeholder="$t('recognize.datasetNamePlaceholder')" variant="outlined" density="compact" :hint="$t('recognize.saveDatasetHint', {
            count: recognizeStore.pendingSamples.length,
          })
            " persistent-hint />
      </v-card-text>
      <v-card-actions>
        <v-spacer />
        <v-btn variant="text" @click="saveDatasetDialogVisible = false">
          {{ $t("common.cancel") }}
        </v-btn>
        <v-btn color="primary" @click="confirmSaveDataset">
          {{ $t("common.save") }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>

  <v-dialog v-model="sampleRelabelDialogVisible" max-width="400px">
    <v-card class="dialog-card">
      <v-card-title>{{
        $t("recognize.sampleRelabelDialogTitle")
      }}</v-card-title>
      <v-divider />
      <v-card-text class="pt-4">
        <div class="mt-3">
          <NumericStepper v-model="tempSampleLabel" :label="$t('recognize.correctDigitLabel')" :min="0" :max="9"
            :step="1" readonly
            :hint="$t('recognize.currentSampleLabelHint', { label: recognizeStore.pendingSamples[currentSampleIndex]?.label })"
            persistent-hint />
        </div>
      </v-card-text>
      <v-card-actions>
        <v-spacer />
        <v-btn variant="text" @click="sampleRelabelDialogVisible = false">
          {{ $t("common.cancel") }}
        </v-btn>
        <v-btn color="primary" @click="applySampleRelabel">
          {{ $t("common.confirm") }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>

  <v-dialog v-model="deletePendingDialogVisible" max-width="400px">
    <v-card class="dialog-card">
      <v-card-title>{{
        $t("common.confirmDeletion")
      }}</v-card-title>
      <v-divider></v-divider>
      <v-card-text class="pt-4">
        {{ $t("recognize.confirmDeletePendingText") }}
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn variant="text" @click="deletePendingDialogVisible = false">
          {{ $t("common.cancel") }}
        </v-btn>
        <v-btn color="error" @click="confirmDeletePending">
          {{ $t("common.confirm") }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>

  <v-dialog v-model="clearPendingDialogVisible" max-width="400px">
    <v-card class="dialog-card">
      <v-card-title>{{
        $t("recognize.confirmClearPendingTitle")
      }}</v-card-title>
      <v-divider />
      <v-card-text class="pt-4">
        {{ $t("recognize.confirmClearPendingText") }}
      </v-card-text>
      <v-card-actions>
        <v-spacer />
        <v-btn variant="text" @click="clearPendingDialogVisible = false">
          {{ $t("common.cancel") }}
        </v-btn>
        <v-btn color="error" @click="confirmClearPending">
          {{ $t("common.confirm") }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>

</template>

<script setup>

import { ref, onMounted, watch } from "vue";
import { useI18n } from "vue-i18n";
import { useRecognizeStore } from "@/stores/recognizeStore";
import { useModelStore } from "@/stores/modelStore";
import { useSnackbarStore } from '@/stores/snackbarStore'
import NumericStepper from '@/components/NumericStepper.vue'

const { t } = useI18n();

const recognizeStore = useRecognizeStore();

const modelStore = useModelStore();

const snackbarStore = useSnackbarStore()

const canvasRef = ref(null);

let ctx = null;

let drawing = false;

const brushSize = ref(12);

const brushColor = ref("#000000");

const recognizing = ref(false);

const relabelDialogVisible = ref(false);

const relabelTarget = ref(null);

const newLabel = ref(0);

const batchRelabelDialogVisible = ref(false);

const batchNewLabel = ref(0);

const saveDatasetDialogVisible = ref(false);

const datasetName = ref("");

const sampleRelabelDialogVisible = ref(false);

const currentSampleIndex = ref(-1);

const tempSampleLabel = ref(0);

const deletePendingDialogVisible = ref(false);

let pendingDeleteIndex = -1;

const clearPendingDialogVisible = ref(false);

function initCanvas() {
  if (!canvasRef.value) return;
  ctx = canvasRef.value.getContext("2d");
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, canvasRef.value.width, canvasRef.value.height);
  ctx.strokeStyle = brushColor.value;
  ctx.lineWidth = brushSize.value;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}

function getCanvasCoordinates(e) {
  const canvas = canvasRef.value;
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  let clientX, clientY;
  if (e.touches) {
    if (e.touches.length === 0) return null;
    clientX = e.touches[0].clientX;
    clientY = e.touches[0].clientY;
  } else {
    clientX = e.clientX;
    clientY = e.clientY;
  }
  let x = (clientX - rect.left) * scaleX;
  let y = (clientY - rect.top) * scaleY;
  x = Math.min(Math.max(0, x), canvas.width);
  y = Math.min(Math.max(0, y), canvas.height);
  return { x, y };
}

function startDrawing(e) {
  drawing = true;
  const point = getCanvasCoordinates(e);
  if (ctx && point) {
    ctx.beginPath();
    ctx.moveTo(point.x, point.y);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
  }
}

function draw(e) {
  if (!drawing) return;
  const point = getCanvasCoordinates(e);
  if (ctx && point) {
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
  }
}

function stopDrawing() {
  drawing = false;
  if (ctx) ctx.beginPath();
}

function clearCanvas() {
  if (ctx && canvasRef.value) {
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, canvasRef.value.width, canvasRef.value.height);
  }
}

function getCanvasBase64() {
  if (!canvasRef.value) return "";
  return canvasRef.value.toDataURL("image/png").split(",")[1];
}

function incrementBrushSize() {
  brushSize.value = Math.min(30, brushSize.value + 2);
}

function decrementBrushSize() {
  brushSize.value = Math.max(2, brushSize.value - 2);
}

watch([brushSize, brushColor], () => {
  if (ctx) {
    ctx.lineWidth = brushSize.value;
    ctx.strokeStyle = brushColor.value;
  }
});

async function recognizeFromBase64(base64) {
  if (!modelStore.ready) {
    snackbarStore.show(t("common.modelNotReady"), "error");
    return;
  }
  recognizing.value = true;
  try {
    await recognizeStore.recognize(base64);
  } catch (err) {
    console.error("Recognition failed", err);
  } finally {
    recognizing.value = false;
  }
}

async function recognize() {
  const base64 = getCanvasBase64();
  if (!base64) return;
  await recognizeFromBase64(base64);
}

function uploadImage() {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/png, image/jpeg, image/jpg";
  input.onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (event) => {
      const base64 = event.target.result.split(",")[1];
      if (!base64) return;
      await recognizeFromBase64(base64);
    };
    reader.readAsDataURL(file);
  };
  input.click();
}

function clearDetections() {
  recognizeStore.clearDetections();
}

function toggleSelectAll() {
  recognizeStore.toggleSelectAll();
}

function confirmDetection(det) {
  recognizeStore.confirmDetection(det);
}

function batchAddToDataset() {
  recognizeStore.batchAddToPending();
}

function openRelabelDialog(det) {
  relabelTarget.value = det;
  newLabel.value = det.digit;
  relabelDialogVisible.value = true;
}

function applyRelabel() {
  if (relabelTarget.value) {
    recognizeStore.relabelDetection(relabelTarget.value, newLabel.value);
  }
  relabelDialogVisible.value = false;
}

function openBatchRelabelDialog() {
  if (recognizeStore.selectedDetections.length === 0) return;
  batchNewLabel.value = 0;
  batchRelabelDialogVisible.value = true;
}

function applyBatchRelabel() {
  recognizeStore.batchRelabel(batchNewLabel.value);
  batchRelabelDialogVisible.value = false;
}

function clearPendingSamples() {
  recognizeStore.clearPendingSamples();
}

function removePendingSample(idx) {
  recognizeStore.removePendingSample(idx);
}

function openDeletePendingConfirm(idx) {
  pendingDeleteIndex = idx;
  deletePendingDialogVisible.value = true;
}

function confirmDeletePending() {
  if (pendingDeleteIndex !== -1) {
    recognizeStore.removePendingSample(pendingDeleteIndex);
    pendingDeleteIndex = -1;
  }
  deletePendingDialogVisible.value = false;
}

function openSampleRelabelDialog(sample, idx) {
  currentSampleIndex.value = idx;
  tempSampleLabel.value = sample.label;
  sampleRelabelDialogVisible.value = true;
}

function applySampleRelabel() {
  if (currentSampleIndex.value !== -1) {
    recognizeStore.updatePendingLabel(
      currentSampleIndex.value,
      tempSampleLabel.value,
    );
  }
  sampleRelabelDialogVisible.value = false;
  currentSampleIndex.value = -1;
}

function openClearPendingConfirm() {
  clearPendingDialogVisible.value = true;
}

function confirmClearPending() {
  recognizeStore.clearPendingSamples();
  clearPendingDialogVisible.value = false;
}

function saveToDataset() {
  if (recognizeStore.pendingSamples.length === 0) return;
  datasetName.value = "";
  saveDatasetDialogVisible.value = true;
}

async function confirmSaveDataset() {
  if (!datasetName.value.trim()) return;
  try {
    await recognizeStore.savePendingToDataset(datasetName.value);
    saveDatasetDialogVisible.value = false;
  } catch (err) {
    console.error("Failed to save to dataset", err);
  }
}

onMounted(() => {
  initCanvas();
});
</script>

<style scoped>
.canvas-container {
  display: flex;
  justify-content: center;
  width: 100%;
}

.detection-item {
  padding: 8px 0;
}

.brush-preview {
  flex-shrink: 0;
  transition: all 0.2s ease;
  cursor: pointer;
}

.brush-preview:hover {
  transform: scale(1.2);
  box-shadow: 0 0 6px rgba(0, 0, 0, 0.2);
}

.v-color-picker {
  background-color: transparent !important;
}

.sample-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  gap: 16px;
  padding: 16px;
}

.sample-card {
  display: flex;
  align-items: stretch;
  background: rgba(255, 255, 255, 0.02);
  justify-content: space-between;
  backdrop-filter: blur(4px);
  border-radius: 16px;
  padding: 12px;
  gap: 24px;
  transition:
    transform 0.2s ease,
    box-shadow 0.2s ease;
}

.sample-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.sample-info {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-top: 8px;
  border-radius: 8px;
}

.sample-label-chip {
  background-color: rgba(var(--v-theme-primary), 0.12);
  color: rgb(var(--v-theme-primary));
  padding: 4px 12px;
  border-radius: 24px;
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s;
}

.sample-actions {
  display: flex;
  gap: 8px;
  align-items: center;
}

.sample-image-wrapper {
  border-radius: 8px;
  overflow: hidden;
  background-color: #f5f5f5;
  flex-shrink: 0;
}

.sample-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
</style>
