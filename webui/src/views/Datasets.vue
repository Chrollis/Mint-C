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
    <v-col cols="12">
      <v-card class="frosted-glass mb-4">
        <v-card-title>
          {{ $t("datasets.uploadTitle") }}
        </v-card-title>
        <v-divider></v-divider>
        <v-card-text class="pa-4">
          <v-row>
            <v-col cols="12" md="4" class="pb-2">
              <v-text-field v-model="newDatasetName" :label="$t('datasets.datasetNameLabel')"
                :placeholder="$t('datasets.datasetNamePlaceholder')" density="compact"
                variant="outlined"></v-text-field>
            </v-col>
            <v-col cols="12" md="4" class="pb-2">
              <v-file-input v-model="imagesFile" :label="$t('datasets.imagesFileLabel')" accept=".idx3-ubyte"
                density="compact" variant="outlined" prepend-icon=""></v-file-input>
            </v-col>
            <v-col cols="12" md="4" class="pb-2">
              <v-file-input v-model="labelsFile" :label="$t('datasets.labelsFileLabel')" accept=".idx1-ubyte"
                density="compact" variant="outlined" prepend-icon=""></v-file-input>
            </v-col>
          </v-row>
        </v-card-text>
        <v-divider></v-divider>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn variant="text" prepend-icon="mdi-upload" :loading="datasetStore.uploading" @click="uploadDataset"
            class="mr-2">
            {{ $t("common.upload") }}
          </v-btn>
          <v-btn color="primary" variant="text" prepend-icon="mdi-refresh" @click="refreshList"
            :loading="datasetStore.loading">
            {{ $t("common.refresh") }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-col>

    <v-col cols="12">
      <v-card class="frosted-glass">
        <v-card-title>
          {{ $t("datasets.listTitle") }}
        </v-card-title>
        <v-divider></v-divider>
        <v-card-text class="pa-0">
          <v-data-table :key="tableKey" v-model="selected" :headers="headers" :items="datasetStore.datasets"
            :loading="datasetStore.loading" item-key="name" show-select return-object hide-default-footer
            :items-per-page="-1" class="dataset-table">
            <template v-slot:item.name="{ item }">
              <div class="d-flex align-center">
                <span class="text-truncate" style="max-width: 200px; display: inline-block" :title="item.name">
                  {{ item.name }}
                </span>
              </div>
            </template>
            <template v-slot:item.sample_count="{ item }">
              <span :class="item.error ? 'text-error' : ''">
                {{ item.error ? item.error : item.sample_count }}
              </span>
            </template>
            <template v-slot:item.status="{ item }">
              <v-chip :color="item.error ? 'error' : 'primary'" size="small">
                {{
                  item.error
                    ? $t("datasets.statusError")
                    : $t("datasets.statusOk")
                }}
              </v-chip>
            </template>
            <template v-slot:item.actions="{ item }">
              <v-icon class="mr-2 action-icon" size="small" @click.stop="downloadSingle(item)">
                mdi-download
              </v-icon>
              <v-icon class="action-icon" color="error" size="small" @click.stop="openSingleDeleteConfirm(item)">
                mdi-delete
              </v-icon>
            </template>
          </v-data-table>
        </v-card-text>
        <v-divider></v-divider>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn variant="text" prepend-icon="mdi-download" :disabled="selected.length === 0 || batchDownloading"
            :loading="batchDownloading" class="mr-2" @click="openDownloadConfirm">
            {{ $t("common.download") }}
          </v-btn>
          <v-btn color="error" variant="text" prepend-icon="mdi-delete" :disabled="selected.length === 0" class="mr-2"
            @click="openDeleteConfirm">
            {{ $t("common.delete") }}
          </v-btn>
          <v-btn color="primary" variant="text" prepend-icon="mdi-merge" :disabled="selected.length < 2"
            @click="openMergeDialog">
            {{ $t("datasets.mergeBtn") }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-col>
  </v-row>

  <v-dialog v-model="deleteDialogVisible" max-width="400px">
    <v-card class="dialog-card">
      <v-card-title>{{ $t("common.confirmDeletion") }}</v-card-title>
      <v-divider></v-divider>
      <v-card-text class="pt-4">
        <p class="mb-2">
          {{ $t("datasets.deleteConfirmText", { count: selected.length }) }}
        </p>
        <div class="dataset-list-container" style="max-height: 200px; overflow-y: auto; margin: 8px 0">
          <div v-for="(item, idx) in selected" :key="item.name" class="list-item" style="
              display: flex;
              align-items: center;
              gap: 12px;
              padding: 6px 0;
            ">
            <div style="width: 70px" class="index-circle">
              Idx.{{ idx + 1 }}
            </div>
            <div style="
                flex: 1;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
              " :title="item.name">
              {{ item.name }}
            </div>
          </div>
        </div>
        <p class="mt-2 mb-0">{{ $t("datasets.deleteIrreversible") }}</p>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn variant="text" @click="deleteDialogVisible = false">
          {{ $t("common.cancel") }}
        </v-btn>
        <v-btn color="error" @click="confirmDelete">
          {{ $t("common.confirm") }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>

  <v-dialog v-model="downloadDialogVisible" max-width="400px">
    <v-card class="dialog-card">
      <v-card-title>{{ $t("datasets.downloadConfirmTitle") }}</v-card-title>
      <v-divider></v-divider>
      <v-card-text class="pt-4">
        <p class="mb-2">
          {{ $t("datasets.downloadConfirmText", { count: selected.length }) }}
        </p>
        <div class="dataset-list-container" style="max-height: 200px; overflow-y: auto; margin: 8px 0">
          <div v-for="(item, idx) in selected" :key="item.name" class="list-item" style="
              display: flex;
              align-items: center;
              gap: 12px;
              padding: 6px 0;
            ">
            <div style="width: 70px" class="index-circle">
              Idx.{{ idx + 1 }}
            </div>
            <div style="
                flex: 1;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
              " :title="item.name">
              {{ item.name }}
            </div>
          </div>
        </div>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn variant="text" @click="downloadDialogVisible = false">
          {{ $t("common.cancel") }}
        </v-btn>
        <v-btn color="primary" @click="confirmDownload">
          {{ $t("common.confirm") }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>

  <v-dialog v-model="singleDeleteDialogVisible" max-width="400px">
    <v-card class="dialog-card">
      <v-card-title>{{ $t("common.confirmDeletion") }}</v-card-title>
      <v-divider></v-divider>
      <v-card-text class="pt-4">
        {{
          $t("datasets.singleDeleteConfirmText", {
            name: singleDeleteTarget?.name,
          })
        }}
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn variant="text" @click="singleDeleteDialogVisible = false">
          {{ $t("common.cancel") }}
        </v-btn>
        <v-btn color="error" @click="confirmSingleDelete">
          {{ $t("common.confirm") }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>

  <v-dialog v-model="mergeDialogVisible" max-width="500px">
    <v-card class="dialog-card">
      <v-card-title>{{ $t("datasets.mergeDialogTitle") }}</v-card-title>
      <v-divider></v-divider>
      <v-card-text class="pt-4">
        <p>{{ $t("datasets.mergeDialogText", { count: selected.length }) }}</p>
        <div class="dataset-list-container" style="max-height: 200px; overflow-y: auto; margin: 8px 0">
          <div v-for="(item, idx) in selected" :key="item.name" class="list-item" style="
              display: flex;
              align-items: center;
              gap: 12px;
              padding: 6px 0;
            ">
            <div style="width: 70px" class="index-circle">
              Idx.{{ idx + 1 }}
            </div>
            <div style="
                flex: 1;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
              " :title="item.name">
              {{ item.name }}
            </div>
          </div>
        </div>
        <v-text-field v-model="mergeTargetName" :label="$t('datasets.mergeTargetLabel')"
          :placeholder="$t('datasets.mergeTargetPlaceholder')" density="compact" variant="outlined"></v-text-field>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn variant="text" @click="mergeDialogVisible = false">
          {{ $t("common.cancel") }}
        </v-btn>
        <v-btn color="primary" @click="confirmMerge" :disabled="!mergeTargetName">
          {{ $t("common.confirm") }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup>
import { ref, onMounted, computed } from "vue";
import { useI18n } from "vue-i18n";
import JSZip from "jszip";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSnackbarStore } from '@/stores/snackbarStore'

const { t } = useI18n();

const datasetStore = useDatasetStore();

const snackbarStore = useSnackbarStore()

const newDatasetName = ref("");

const imagesFile = ref(null);

const labelsFile = ref(null);

const selected = ref([]);

const tableKey = ref(0);

const batchDownloading = ref(false);

const deleteDialogVisible = ref(false);

const downloadDialogVisible = ref(false);

const singleDeleteDialogVisible = ref(false);

const mergeDialogVisible = ref(false);

const singleDeleteTarget = ref(null);

const mergeTargetName = ref("");

const headers = computed(() => [
  { title: t("datasets.headers.name"), key: "name", sortable: true },
  {
    title: t("datasets.headers.sampleCount"),
    key: "sample_count",
    sortable: true,
  },
  { title: t("datasets.headers.status"), key: "status", sortable: false },
  {
    title: t("datasets.headers.actions"),
    key: "actions",
    sortable: false,
    align: "center",
  },
]);

function base64ToArrayBuffer(base64) {
  const raw = base64.includes(",") ? base64.split(",")[1] : base64;
  const binaryString = atob(raw);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}

async function saveAsFile(blob, suggestedName) {
  if (window.showSaveFilePicker) {
    try {
      const fileHandle = await window.showSaveFilePicker({
        suggestedName: suggestedName,
        types: [
          {
            description: 'ZIP Archive',
            accept: { 'application/zip': ['.zip'] },
          },
        ],
      });
      const writable = await fileHandle.createWritable();
      await writable.write(blob);
      await writable.close();
      return true;
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Save file error', err);
      }
      return false;
    }
  } else {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = suggestedName;
    a.click();
    URL.revokeObjectURL(url);
    return true;
  }
}

async function refreshList() {
  try {
    await datasetStore.fetchDatasets();
    selected.value = [];
    tableKey.value++;
  } catch (err) {
    console.error("Failed to load dataset list", err);
    snackbarStore.show(t("datasets.errors.loadFailed"), "error");
  }
}

async function uploadDataset() {
  if (!newDatasetName.value) {
    snackbarStore.show(t("datasets.errors.missingName"), "error");
    return;
  }
  if (!imagesFile.value || !labelsFile.value) {
    snackbarStore.show(t("datasets.errors.missingFiles"), "error");
    return;
  }

  try {
    await datasetStore.uploadDataset(
      newDatasetName.value,
      imagesFile.value,
      labelsFile.value,
    );
    snackbarStore.show(t("datasets.uploadSuccess", { name: newDatasetName.value }));
    newDatasetName.value = "";
    imagesFile.value = null;
    labelsFile.value = null;
  } catch (err) {
    console.error("Upload failed", err);
    snackbarStore.show(t("datasets.errors.uploadFailed"), "error");
  }
}

async function confirmDelete() {
  const names = selected.value.map((item) => item.name);
  deleteDialogVisible.value = false;
  let hasError = false;
  for (const name of names) {
    try {
      await datasetStore.deleteDataset(name);
    } catch (err) {
      console.error(`Delete failed for ${name}`, err);
      hasError = true;
    }
  }
  if (hasError) {
    snackbarStore.show(t("datasets.errors.partialDeleteFailed"), "error");
  } else {
    snackbarStore.show(t("datasets.deleteSuccess", { count: names.length }));
  }
  await refreshList();
}

async function downloadSingleDataset(name, displayName = name) {
  try {
    const data = await datasetStore.fetchDatasetData(name);
    const { images_base64, labels_base64 } = data;
    const zip = new JSZip();
    zip.file("images.idx3-ubyte", base64ToArrayBuffer(images_base64));
    zip.file("labels.idx1-ubyte", base64ToArrayBuffer(labels_base64));
    const zipBlob = await zip.generateAsync({ type: "blob" });

    const saved = await saveAsFile(zipBlob, `${displayName}.zip`);
    if (saved) {
      snackbarStore.show(t("datasets.downloadStart", { name: displayName }));
    }
  } catch (err) {
    console.error(`Download failed for ${name}`, err);
    snackbarStore.show(
      t("datasets.errors.downloadFailed", { name: displayName }),
      "error",
    );
  }
}

async function downloadSingle(item) {
  await downloadSingleDataset(item.name, item.name);
}

function openDownloadConfirm() {
  downloadDialogVisible.value = true;
}

async function confirmDownload() {
  const selectedDatasets = [...selected.value];
  if (selectedDatasets.length === 0) return;

  downloadDialogVisible.value = false;
  batchDownloading.value = true;
  const totalZip = new JSZip();

  try {
    for (const ds of selectedDatasets) {
      const data = await datasetStore.fetchDatasetData(ds.name);
      const { images_base64, labels_base64 } = data;
      const folder = totalZip.folder(ds.name);
      folder.file("images.idx3-ubyte", base64ToArrayBuffer(images_base64));
      folder.file("labels.idx1-ubyte", base64ToArrayBuffer(labels_base64));
    }
    const zipBlob = await totalZip.generateAsync({ type: "blob" });
    const defaultName = `datasets_${new Date().toISOString().slice(0, 19)}.zip`;

    const saved = await saveAsFile(zipBlob, defaultName);
    if (saved) {
      snackbarStore.show(
        t("datasets.batchDownloadSuccess", { count: selectedDatasets.length }),
      );
    }
  } catch (err) {
    console.error("Batch download failed", err);
    snackbarStore.show(t("datasets.errors.batchDownloadFailed"), "error");
  } finally {
    batchDownloading.value = false;
  }
}

function openSingleDeleteConfirm(item) {
  singleDeleteTarget.value = item;
  singleDeleteDialogVisible.value = true;
}

async function confirmSingleDelete() {
  if (singleDeleteTarget.value) {
    const name = singleDeleteTarget.value.name;
    try {
      await datasetStore.deleteDataset(name);
      snackbarStore.show(t("datasets.singleDeleteSuccess", { name }));
      await refreshList();
    } catch (err) {
      console.error(`Delete failed for ${name}`, err);
      snackbarStore.show(t("datasets.errors.singleDeleteFailed", { name }), "error");
    }
    singleDeleteDialogVisible.value = false;
    singleDeleteTarget.value = null;
  }
}

function openDeleteConfirm() {
  deleteDialogVisible.value = true;
}

function openMergeDialog() {
  mergeTargetName.value = "";
  mergeDialogVisible.value = true;
}

async function confirmMerge() {
  if (!mergeTargetName.value) return;
  const sourceNames = selected.value.map((item) => item.name);
  try {
    await datasetStore.mergeDatasets(sourceNames, mergeTargetName.value);
    snackbarStore.show(t("datasets.mergeSuccess", { name: mergeTargetName.value }));
    mergeDialogVisible.value = false;
    await refreshList();
  } catch (err) {
    console.error("Merge failed", err);
    snackbarStore.show(t("datasets.errors.mergeFailed"), "error");
    try {
      await datasetStore.deleteDataset(mergeTargetName.value);
      console.log(`Cleaned up failed dataset: ${mergeTargetName.value}`);
    } catch (cleanErr) {
      console.error("Failed to clean up empty dataset", cleanErr);
    }
  }
}

onMounted(() => {
  refreshList();
});
</script>

<style scoped>
.dataset-table {
  background: transparent;
}

.action-icon {
  cursor: pointer;
  transition:
    transform 0.2s,
    filter 0.2s;
}

.action-icon:hover {
  transform: scale(1.1);
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.2));
}

.index-circle {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 24px;
  height: 24px;
  padding: 0 8px;
  border-radius: 12px;
  background-color: rgba(var(--v-theme-primary), 0.2);
  color: rgb(var(--v-theme-primary));
  font-size: 0.75rem;
  font-weight: 500;
  flex-shrink: 0;
}
</style>