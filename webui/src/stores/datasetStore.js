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
import {
  listDatasets,
  uploadDataset,
  downloadDataset,
  deleteDataset,
  mergeDatasets,
} from "@/api";
import { useAsyncOperation } from "@/composables/useAsyncOperation";

export const useDatasetStore = defineStore("dataset", () => {
  const datasets = ref([]);

  const { execute: fetchDatasets, loading } = useAsyncOperation(
    async () => {
      const res = await listDatasets();
      datasets.value = res.data.datasets;
    },
    { throwOnError: true },
  );

  const { execute: uploadDatasetAction, loading: uploading } =
    useAsyncOperation(
      async (name, imagesFile, labelsFile) => {
        const res = await uploadDataset(name, imagesFile, labelsFile);
        await fetchDatasets();
        return res.data;
      },
      { throwOnError: true },
    );

  const { execute: deleteDatasetAction, loading: deleting } = useAsyncOperation(
    async (name) => {
      await deleteDataset(name);
      await fetchDatasets();
    },
    { throwOnError: true },
  );

  const { execute: mergeDatasetsAction, loading: merging } = useAsyncOperation(
    async (sourceNames, targetName) => {
      await mergeDatasets(sourceNames, targetName);
      await fetchDatasets();
    },
    { throwOnError: true },
  );

  async function fetchDatasetData(name) {
    try {
      const res = await downloadDataset(name);
      return res.data;
    } catch (err) {
      console.error("Fetch dataset data failed:", err);
      throw err;
    }
  }

  return {
    datasets,
    loading,
    uploading,
    deleting,
    merging,
    fetchDatasets,
    uploadDataset: uploadDatasetAction,
    deleteDataset: deleteDatasetAction,
    mergeDatasets: mergeDatasetsAction,
    fetchDatasetData,
  };
});
