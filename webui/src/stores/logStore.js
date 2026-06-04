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
import { getLogs } from "@/api";
import { useAsyncOperation } from "@/composables/useAsyncOperation";

export const useLogStore = defineStore("log", () => {
  const logs = ref([]);

  const latestLog = ref("");

  const currentLevel = ref("info");

  const currentCount = ref(100);

  const { execute: fetchLogs, loading } = useAsyncOperation(
    async (level = currentLevel.value, count = currentCount.value) => {
      const res = await getLogs(count, level);
      logs.value = res.data.logs || [];
    },
    { throwOnError: false },
  );

  async function fetchLatestLog() {
    try {
      const res = await getLogs(1, "debug");
      const logLines = res.data.logs || [];
      latestLog.value =
        logLines.length > 0
          ? logLines[0].slice(logLines[0].indexOf("]") + 1)
          : "";
    } catch (err) {
      console.error("Failed to fetch latest log:", err);
      latestLog.value = "";
    }
  }

  function setLevel(level) {
    currentLevel.value = level;
    fetchLogs(level, currentCount.value);
  }

  function setCount(count) {
    currentCount.value = count;
    fetchLogs(currentLevel.value, count);
  }

  return {
    logs,
    latestLog,
    loading,
    currentLevel,
    currentCount,
    fetchLogs,
    fetchLatestLog,
    setLevel,
    setCount,
  };
});
