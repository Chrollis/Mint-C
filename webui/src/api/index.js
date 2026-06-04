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

import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error("API Error:", error);
    return Promise.reject(error);
  },
);

export const getHealth = () => api.get("/health");

export const getServerStatus = () => api.get("/status");

export const getConfig = () => api.get("/config");

export const updateConfig = (config) => api.post("/config", config);

export const listDatasets = () => api.get("/datasets");

export const uploadDataset = (datasetName, imagesFile, labelsFile) => {
  const formData = new FormData();
  formData.append("images_file", imagesFile);
  formData.append("labels_file", labelsFile);
  return api.post(
    `/datasets/upload?name=${encodeURIComponent(datasetName)}`,
    formData,
    {
      headers: { "Content-Type": "multipart/form-data" },
    },
  );
};

export const downloadDataset = (datasetName) =>
  api.get(`/datasets/download?name=${encodeURIComponent(datasetName)}`);

export const deleteDataset = (datasetName) =>
  api.post("/datasets/delete", { name: datasetName });

export const saveSamplesToDataset = (datasetName, samples, append = false) =>
  api.post("/datasets/save_samples", {
    dataset_name: datasetName,
    samples,
    append,
  });

export const mergeDatasets = (sourceDatasets, targetDataset) =>
  api.post("/datasets/merge", {
    source_datasets: sourceDatasets,
    target_dataset: targetDataset,
  });

export const getLogs = (count = 100, level = "info") =>
  api.get("/logs", { params: { count, level } });

export const recognizeImage = (base64) => api.post("/recognize", { base64 });

export const getLayerTypes = () => api.get("/layer/types");

export const getParameterRange = (type, inShape) =>
  api.post("/layer/parameter_range", { type, in_shape: inShape });

export const computeOutputShape = (type, inShape, params) =>
  api.post("/layer/compute_output_shape", { type, in_shape: inShape, params });

export const loadModelBinary = (modelFile) => {
  const formData = new FormData();
  formData.append("model_file", modelFile);
  return api.post("/model/load_bin", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export const initModelFromJson = (config) =>
  api.post("/model/init_json", config);

export const downloadModelBinary = () =>
  api.get("/model/save_bin", { responseType: "blob" });

export const exportModelJson = () => api.get("/model/export_json");

export const isModelReady = () => api.get("/model/ready");

export const getHyperparamsRange = () => api.get("/train/hyperparams_range");

export const startTraining = (
  datasetName,
  hyperparams = {},
  offset = 0,
  count = 0,
) =>
  api.post("/train/start", {
    dataset_name: datasetName,
    offset,
    count,
    hyperparams,
  });

export const cancelTraining = () => api.post("/train/cancel");

export const getTrainingStatus = () => api.get("/train/status");

export const acceptTrainingResult = (accept) =>
  api.post("/train/accept", { accept });

export const startEvaluation = (
  datasetName,
  offset = 0,
  count = 0,
  batchSize = 32,
) =>
  api.post("/evaluate/start", {
    dataset_name: datasetName,
    offset,
    count,
    batch_size: batchSize,
  });

export const cancelEvaluation = () => api.post("/evaluate/cancel");

export const getEvaluationStatus = () => api.get("/evaluate/status");

export default {
  getHealth,
  getServerStatus,
  getConfig,
  updateConfig,
  listDatasets,
  uploadDataset,
  downloadDataset,
  deleteDataset,
  saveSamplesToDataset,
  mergeDatasets,
  getLogs,
  recognizeImage,
  getLayerTypes,
  getParameterRange,
  computeOutputShape,
  loadModelBinary,
  initModelFromJson,
  downloadModelBinary,
  exportModelJson,
  isModelReady,
  getHyperparamsRange,
  startTraining,
  cancelTraining,
  getTrainingStatus,
  acceptTrainingResult,
  startEvaluation,
  cancelEvaluation,
  getEvaluationStatus,
};
