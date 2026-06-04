// Mint-C - Web-based digit recognition tool with C++ CNN backend
// Copyright (C) 2026 Chrollis
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

import { defineStore } from "pinia";
import { ref } from "vue";
import { recognizeImage, saveSamplesToDataset } from "@/api";

export const useRecognizeStore = defineStore("recognize", () => {
  const detections = ref([]);

  const selectedDetections = ref([]);

  const pendingSamples = ref([]);

  async function recognize(base64Image) {
    try {
      const res = await recognizeImage(base64Image);
      detections.value = res.data.digits || [];
      selectedDetections.value = [];
      return res.data;
    } catch (err) {
      console.error("Failed to recognize:", err);
      throw err;
    }
  }

  function clearDetections() {
    detections.value = [];
    selectedDetections.value = [];
  }

  function toggleSelectAll() {
    if (selectedDetections.value.length === detections.value.length) {
      selectedDetections.value = [];
    } else {
      selectedDetections.value = [...detections.value];
    }
  }

  function confirmDetection(detection) {
    addToPending(detection, detection.digit);
    removeDetection(detection);
  }

  function removeDetection(detection) {
    const idx = detections.value.indexOf(detection);
    if (idx !== -1) detections.value.splice(idx, 1);
    selectedDetections.value = selectedDetections.value.filter(
      (d) => d !== detection,
    );
  }

  function batchRelabel(newLabel) {
    const toProcess = [...selectedDetections.value];
    for (const det of toProcess) {
      addToPending(det, newLabel);
      removeDetection(det);
    }
    selectedDetections.value = [];
  }

  function relabelDetection(detection, newLabel) {
    addToPending(detection, newLabel);
    removeDetection(detection);
  }

  function addToPending(detection, label) {
    pendingSamples.value.push({
      mnist_base64: detection.mnist_base64,
      label: label,
    });
  }

  function batchAddToPending() {
    for (const det of selectedDetections.value) {
      addToPending(det, det.digit);
      removeDetection(det);
    }
    selectedDetections.value = [];
  }

  function removePendingSample(index) {
    pendingSamples.value.splice(index, 1);
  }

  function clearPendingSamples() {
    pendingSamples.value = [];
  }

  function updatePendingLabel(index, newLabel) {
    if (pendingSamples.value[index]) {
      pendingSamples.value[index].label = newLabel;
    }
  }

  async function savePendingToDataset(datasetName, append = true) {
    if (!pendingSamples.value.length) return;
    const samples = pendingSamples.value.map((sample) => ({
      image_base64: sample.mnist_base64,
      label: sample.label,
    }));
    try {
      const res = await saveSamplesToDataset(datasetName, samples, append);
      if (res.data.success) {
        clearPendingSamples();
      }
      return res.data;
    } catch (err) {
      console.error("Failed to save samples:", err);
      throw err;
    }
  }

  return {
    detections,
    selectedDetections,
    pendingSamples,
    recognize,
    clearDetections,
    toggleSelectAll,
    confirmDetection,
    removeDetection,
    batchRelabel,
    relabelDetection,
    addToPending,
    batchAddToPending,
    removePendingSample,
    clearPendingSamples,
    updatePendingLabel,
    savePendingToDataset,
  };
});
