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
  getHyperparamsRange,
  startTraining,
  cancelTraining,
  getTrainingStatus,
  acceptTrainingResult,
} from "@/api";
import { toFixed, toPercent } from "@/utils/formatters";
import { usePersistedState } from "@/composables/usePersistedState";
import { usePolling } from "@/composables/usePolling";

export const useTrainingStore = defineStore("training", () => {
  const status = ref({
    state: "idle",
    current_epoch: 0,
    total_epochs: 0,
    processed_samples: 0,
    total_samples: 0,
    current_loss: 0,
    current_accuracy: 0,
    loss_history: [],
    accuracy_history: [],
    error_message: "",
  });

  const hyperparamsRange = ref({
    learning_rate: { min: 0.0001, max: 0.01, step: 0.0001 },
    batch_size: { min: 16, max: 256, step: 16 },
    epochs: { min: 1, max: 20, step: 1 },
  });

  const realtimeLossData = usePersistedState("mint_realtime_loss", []);

  const realtimeAccData = usePersistedState("mint_realtime_acc", []);

  const persistedLossData = usePersistedState("training_loss_history", []);

  const persistedAccData = usePersistedState("training_acc_history", []);

  const { start: startPolling, cancel: cancelPolling } = usePolling(
    fetchTrainingStatus,
    { interval: 1800 },
  );

  let hasAppendedHistory = true;

  const realtimeLabels = computed(() =>
    realtimeLossData.value.map((_, idx) => `${idx + 1}`),
  );

  const persistedEpochLabels = computed(() =>
    persistedLossData.value.map((_, idx) => `${idx + 1}`),
  );

  function clearRealtimeData() {
    realtimeLossData.value = [];
    realtimeAccData.value = [];
  }

  function appendTrainingHistory(lossArray, accArray) {
    if (!lossArray.length || !accArray.length) return;
    persistedLossData.value.push(...lossArray);
    persistedAccData.value.push(...accArray);
    if (persistedLossData.value.length > 100) {
      const excess = persistedLossData.value.length - 100;
      persistedLossData.value.splice(0, excess);
      persistedAccData.value.splice(0, excess);
    }
  }

  async function fetchHyperparamsRange() {
    try {
      const res = await getHyperparamsRange();
      hyperparamsRange.value = res.data;
    } catch (err) {
      console.error("Fetch hyperparams range failed:", err);
    }
  }

  async function startTrainingAction(
    datasetName,
    hyperparams,
    offset = 0,
    count = 0,
  ) {
    try {
      resetStatus();
      await startTraining(datasetName, hyperparams, offset, count);
      await fetchTrainingStatus();
    } catch (err) {
      console.error("Start training failed:", err);
      throw err;
    }
  }

  async function cancelTrainingAction() {
    try {
      await cancelTraining();
      await fetchTrainingStatus();
    } catch (err) {
      console.error("Cancel training failed:", err);
      throw err;
    }
  }

  async function fetchTrainingStatus() {
    try {
      const res = await getTrainingStatus();
      const newStatus = res.data;

      const formattedLoss = toFixed(newStatus.current_loss, 4);
      const formattedAcc = toPercent(newStatus.current_accuracy);
      const formattedLossHistory = newStatus.loss_history.map((v) =>
        toFixed(v, 4),
      );
      const formattedAccHistory = newStatus.accuracy_history.map((v) =>
        toPercent(v),
      );

      status.value = {
        ...newStatus,
        current_loss: formattedLoss,
        current_accuracy: formattedAcc,
        loss_history: formattedLossHistory,
        accuracy_history: formattedAccHistory,
      };

      if (
        status.value.state === "running" &&
        status.value.processed_samples > 0
      ) {
        realtimeLossData.value.push(formattedLoss);
        realtimeAccData.value.push(formattedAcc);
        if (realtimeLossData.value.length > 100) realtimeLossData.value.shift();
        if (realtimeAccData.value.length > 100) realtimeAccData.value.shift();
      }

      if (
        status.value.state === "finished" &&
        !hasAppendedHistory &&
        formattedLossHistory.length
      ) {
        appendTrainingHistory(formattedLossHistory, formattedAccHistory);
        hasAppendedHistory = true;
      }

      if (status.value.state === "running") {
        startPolling();
      } else if (
        ["finished", "cancelled", "error"].includes(status.value.state)
      ) {
        cancelPolling();
      }
    } catch (err) {
      console.error("Fetch training status failed:", err);
    }
  }

  async function acceptResult(accept) {
    try {
      await acceptTrainingResult(accept);
      await fetchTrainingStatus();
    } catch (err) {
      console.error("Accept training result failed:", err);
      throw err;
    }
  }

  function resetStatus() {
    status.value = {
      state: "idle",
      current_epoch: 0,
      total_epochs: 0,
      processed_samples: 0,
      total_samples: 0,
      current_loss: 0,
      current_accuracy: 0,
      loss_history: [],
      accuracy_history: [],
      error_message: "",
    };
    clearRealtimeData();
    hasAppendedHistory = false;
    cancelPolling();
  }

  function clearPersistedHistory() {
    persistedLossData.value = [];
    persistedAccData.value = [];
  }

  return {
    status,
    hyperparamsRange,
    realtimeLossData,
    realtimeAccData,
    persistedLossData,
    persistedAccData,
    realtimeLabels,
    persistedEpochLabels,
    fetchHyperparamsRange,
    startTraining: startTrainingAction,
    cancelTraining: cancelTrainingAction,
    fetchTrainingStatus,
    acceptResult,
    resetStatus,
    clearPersistedHistory,
  };
});
