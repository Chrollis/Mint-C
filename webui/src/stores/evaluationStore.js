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
import { startEvaluation, cancelEvaluation, getEvaluationStatus } from "@/api";
import { toPercent } from "@/utils/formatters";
import { usePersistedState } from "@/composables/usePersistedState";
import { usePolling } from "@/composables/usePolling";

export const useEvaluationStore = defineStore("evaluation", () => {
  const status = ref({
    state: "idle",
    processed_samples: 0,
    total_samples: 0,
    current_accuracy: 0,
    error_message: "",
    final_accuracy: 0,
    per_class_accuracy: {},
  });

  const realtimeAccData = usePersistedState("eval_realtime_acc", []);

  const historyFinalAccuracies = usePersistedState("eval_history_final", []);

  const historyPerClassAccuracies = usePersistedState(
    "eval_history_perclass",
    [],
  );

  const { start: startPolling, cancel: cancelPolling } = usePolling(
    fetchEvaluationStatus,
    { interval: 300 },
  );

  let hasAppendedCurrentHistory = true;

  const realtimeLabels = computed(() =>
    realtimeAccData.value.map((_, idx) => `${idx + 1}`),
  );

  const historyEpochLabels = computed(() =>
    historyFinalAccuracies.value.map((_, idx) => `${idx + 1}`),
  );

  const perClassCategories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

  function addHistoryRecord(finalAcc, perClassAcc) {
    historyFinalAccuracies.value.push(finalAcc);
    historyPerClassAccuracies.value.push(perClassAcc);
    if (historyFinalAccuracies.value.length > 100) {
      historyFinalAccuracies.value.shift();
      historyPerClassAccuracies.value.shift();
    }
    if (historyPerClassAccuracies.value.length > 10) {
      historyPerClassAccuracies.value.shift();
      historyFinalAccuracies.value.shift();
    }
  }

  function clearHistory() {
    historyFinalAccuracies.value = [];
    historyPerClassAccuracies.value = [];
  }

  function resetCurrent() {
    status.value = {
      state: "idle",
      processed_samples: 0,
      total_samples: 0,
      current_accuracy: 0,
      error_message: "",
      final_accuracy: 0,
      per_class_accuracy: {},
    };
    realtimeAccData.value = [];
    cancelPolling();
    hasAppendedCurrentHistory = false;
  }

  async function fetchEvaluationStatus() {
    try {
      const res = await getEvaluationStatus();
      const newStatus = res.data;

      status.value = {
        state: newStatus.state,
        processed_samples: newStatus.processed_samples,
        total_samples: newStatus.total_samples,
        current_accuracy: newStatus.current_accuracy,
        error_message: newStatus.error_message || "",
        final_accuracy: newStatus.final_accuracy || 0,
        per_class_accuracy: newStatus.per_class_accuracy || {},
      };

      if (
        status.value.state === "running" &&
        status.value.processed_samples > 0
      ) {
        const accPercent = toPercent(status.value.current_accuracy);
        realtimeAccData.value.push(accPercent);
        if (realtimeAccData.value.length > 100) realtimeAccData.value.shift();
      }

      const finalAccPercent = toPercent(status.value.final_accuracy);
      const perClassPercent = Object.fromEntries(
        Object.entries(status.value.per_class_accuracy).map(([k, v]) => [
          k,
          v !== null ? toPercent(v) : null,
        ]),
      );

      if (status.value.state === "finished" && !hasAppendedCurrentHistory) {
        addHistoryRecord(finalAccPercent, perClassPercent);
        hasAppendedCurrentHistory = true;
      }

      if (status.value.state === "running") {
        startPolling();
      } else if (
        ["finished", "cancelled", "error"].includes(status.value.state)
      ) {
        cancelPolling();
      }
    } catch (err) {
      console.error("Fetch evaluation status failed", err);
      cancelPolling();
      status.value.state = "error";
      status.value.error_message = err.message || "Failed to fetch status";
    }
  }

  async function startEvaluationAction(
    datasetName,
    offset = 0,
    count = 0,
    batchSize = 32,
  ) {
    try {
      resetCurrent();
      const res = await startEvaluation(datasetName, offset, count, batchSize);
      if (res.data.success) {
        await fetchEvaluationStatus();
        startPolling();
      } else {
        throw new Error(res.data.message || "Start evaluation failed");
      }
    } catch (err) {
      console.error("Start evaluation error", err);
      status.value.state = "error";
      status.value.error_message = err.message || "Failed to start evaluation";
    }
  }

  async function cancelEvaluationAction() {
    try {
      await cancelEvaluation();
      await fetchEvaluationStatus();
    } catch (err) {
      console.error("Cancel evaluation error", err);
    }
  }

  return {
    status,
    realtimeAccData,
    historyFinalAccuracies,
    historyPerClassAccuracies,
    realtimeLabels,
    historyEpochLabels,
    perClassCategories,
    startEvaluation: startEvaluationAction,
    cancelEvaluation: cancelEvaluationAction,
    fetchEvaluationStatus,
    clearHistory,
    resetCurrent,
  };
});
