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
import { ref, computed } from "vue";
import {
  exportModelJson,
  loadModelBinary,
  downloadModelBinary,
  initModelFromJson,
  isModelReady,
  getLayerTypes,
  getParameterRange,
  computeOutputShape,
} from "@/api";
import { useAsyncOperation } from "@/composables/useAsyncOperation";

export const useModelStore = defineStore("model", () => {
  const architecture = ref({ input_shape: [], num_classes: 0, layers: [] });

  const ready = ref(false);

  const layerTypes = ref([]);

  const { execute: fetchModelJsonOp, loading: loadingFetch } =
    useAsyncOperation(
      async () => {
        const res = await exportModelJson();
        architecture.value = res.data;
      },
      { throwOnError: true },
    );

  const { execute: initModelFromJsonOp, loading: loadingInit } =
    useAsyncOperation(
      async (config) => {
        await initModelFromJson(config);
        await fetchModelJsonOp();
        await checkReady();
      },
      { throwOnError: true },
    );

  const { execute: loadModelBinaryOp, loading: loadingLoad } =
    useAsyncOperation(
      async (file) => {
        await loadModelBinary(file);
        await fetchModelJsonOp();
        await checkReady();
      },
      { throwOnError: true },
    );

  const { execute: downloadModelBinaryOp, loading: loadingDownload } =
    useAsyncOperation(
      async () => {
        const res = await downloadModelBinary();
        const blob = new Blob([res.data], { type: "application/octet-stream" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "model.bin";
        a.click();
        URL.revokeObjectURL(url);
      },
      { throwOnError: true },
    );

  const loading = computed(
    () =>
      loadingFetch.value ||
      loadingInit.value ||
      loadingLoad.value ||
      loadingDownload.value,
  );

  async function checkReady() {
    try {
      const res = await isModelReady();
      ready.value = res.data.ready;
    } catch (err) {
      console.error("Check model ready failed:", err);
      ready.value = false;
    }
  }

  async function fetchLayerTypes() {
    try {
      const res = await getLayerTypes();
      layerTypes.value = res.data.types;
    } catch (err) {
      console.error("Fetch layer types failed:", err);
      throw err;
    }
  }

  async function fetchParameterRange(type, inShape) {
    try {
      const res = await getParameterRange(type, inShape);
      return res.data;
    } catch (err) {
      console.error("Fetch parameter range failed:", err);
      throw err;
    }
  }

  async function fetchOutputShape(type, inShape, params) {
    try {
      const res = await computeOutputShape(type, inShape, params);
      return res.data.out_shape;
    } catch (err) {
      console.error("Compute output shape failed:", err);
      throw err;
    }
  }

  async function fetchModelBinaryBlob() {
    const res = await downloadModelBinary();
    return new Blob([res.data], { type: "application/octet-stream" });
  }

  return {
    architecture,
    ready,
    loading,
    layerTypes,

    fetchModelJson: fetchModelJsonOp,
    initModelFromJson: initModelFromJsonOp,
    loadModelBinary: loadModelBinaryOp,
    downloadModelBinary: downloadModelBinaryOp,

    checkReady,
    fetchLayerTypes,
    fetchParameterRange,
    fetchOutputShape,
    fetchModelBinaryBlob,
  };
});
