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
    <v-col cols="12" lg="6">
      <v-card class="frosted-glass" style="height: 100%">
        <v-card-title> {{ $t("model.currentArchTitle") }} </v-card-title>
        <v-divider></v-divider>
        <v-card-text class="pa-0">
          <ModelLayerList :layers="currentModel.layers" :input-shape="currentModel.input_shape"
            :num-classes="currentModel.num_classes" :editable="false" />
        </v-card-text>
        <v-divider></v-divider>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn variant="text" @click="loadModelFromFile" prepend-icon="mdi-upload" class="mr-2"
            :loading="modelStore.loading">
            {{ $t("model.loadModelBtn") }}
          </v-btn>
          <v-btn variant="text" @click="downloadModelBin" class="mr-2" prepend-icon="mdi-download">
            {{ $t("common.save") }}
          </v-btn>
          <v-btn color="primary" variant="text" @click="saveArchitectureToClipboard" prepend-icon="mdi-content-copy">
            {{ $t("model.copyToClipboardBtn") }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-col>

    <v-col cols="12" lg="6">
      <v-card class="frosted-glass" style="height: 100%">
        <v-card-title>
          {{ $t("model.newArchTitle") }}
        </v-card-title>
        <v-divider></v-divider>
        <v-card-text class="pa-0">
          <ModelLayerList :layers="newModel.layers" :input-shape="newModel.input_shape"
            :num-classes="newModel.num_classes" :editable="true" @remove-layer="removeLayer" />
        </v-card-text>
        <v-divider></v-divider>
        <v-card-actions>
          <v-chip size="small" color="primary" variant="flat" class="mr-2">
            {{ $t("model.inputShapeLabel") }}: [{{
              newModel.input_shape.join(", ")
            }}]
          </v-chip>
          <v-spacer></v-spacer>
          <v-btn variant="text" class="mr-2" @click="openAddLayerDialog" prepend-icon="mdi-plus">
            {{ $t("model.addLayerBtn") }}
          </v-btn>
          <v-btn variant="text" color="error" @click="openClearNewModelConfirm" prepend-icon="mdi-delete-empty">
            {{ $t("common.clear") }}
          </v-btn>
        </v-card-actions>
        <v-divider></v-divider>
        <v-card-actions>
          <v-chip size="small" color="primary" variant="flat">
            {{ $t("model.outputShapeLabel") }}: [{{
              displayOutputShape.join(", ")
            }}]
          </v-chip>
          <v-spacer></v-spacer>
          <v-btn variant="text" @click="applyToCurrent" prepend-icon="mdi-check" class="mr-2"
            :loading="modelStore.loading">
            {{ $t("model.applyBtn") }}
          </v-btn>
          <v-btn variant="text" color="primary" @click="loadFromClipboard" prepend-icon="mdi-clipboard-arrow-down">
            {{ $t("model.loadFromClipboardBtn") }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-col>
  </v-row>

  <v-dialog v-model="addLayerDialogVisible" max-width="600px">
    <v-card class="dialog-card">
      <v-card-title>{{ $t("model.addLayerDialogTitle") }}</v-card-title>
      <v-divider />
      <v-card-text class="pt-4">
        <div class="mb-4">
          <v-field :label="$t('model.layerTypeLabel')" :dirty="!!newLayerType" variant="outlined" density="compact"
            class="param-field">
            <v-menu offset-y>
              <template v-slot:activator="{ props }">
                <v-btn variant="text" block v-bind="props" class="field-trigger-btn">
                  {{
                    getLayerDisplayName(newLayerType) ||
                    $t("model.selectLayerType")
                  }}
                </v-btn>
              </template>
              <v-list class="frosted-glass">
                <v-list-item v-for="type in modelStore.layerTypes" :key="type" @click="changeLayerType(type)">
                  <v-list-item-title>
                    {{ getLayerDisplayName(type) }}
                  </v-list-item-title>
                </v-list-item>
              </v-list>
            </v-menu>
          </v-field>
        </div>
        <div v-if="Object.keys(currentLayerParams).length">
          <div v-for="(param, name) in currentLayerParams" :key="name" class="mb-3">
            <div v-if="param.type === 'string'">
              <v-field :label="param.label || name" :dirty="!!param.value" variant="outlined" density="compact"
                class="param-field">
                <v-menu offset-y>
                  <template v-slot:activator="{ props }">
                    <v-btn variant="text" block v-bind="props" class="field-trigger-btn">
                      {{ param.value || $t("model.selectOption") }}
                    </v-btn>
                  </template>
                  <v-list class="frosted-glass">
                    <v-list-item v-for="opt in param.options" :key="opt" @click="param.value = opt">
                      <v-list-item-title>{{ opt }}</v-list-item-title>
                    </v-list-item>
                  </v-list>
                </v-menu>
              </v-field>
            </div>
            <div v-else-if="param.type === 'int' || param.type === 'double'">
              <NumericStepper v-model="param.value" :label="param.label || name" :min="param.min" :max="param.max"
                :step="param.step" />
            </div>
          </div>
        </div>
        <div v-else class="text-center text-grey pa-4">
          {{ $t("model.noParams") }}
        </div>
      </v-card-text>
      <v-card-actions>
        <v-spacer />
        <v-btn variant="text" @click="addLayerDialogVisible = false">
          {{ $t("common.cancel") }}
        </v-btn>
        <v-btn color="primary" @click="confirmAddLayer">
          {{ $t("model.addBtn") }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>

  <v-dialog v-model="clearNewModelDialogVisible" max-width="400px">
    <v-card class="dialog-card">
      <v-card-title>{{ $t("model.confirmClearNewModelTitle") }}</v-card-title>
      <v-divider></v-divider>
      <v-card-text class="pt-4">
        {{ $t("model.confirmClearNewModelText") }}
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn variant="text" @click="clearNewModelDialogVisible = false">
          {{ $t("common.cancel") }}
        </v-btn>
        <v-btn color="error" @click="confirmClearNewModel">
          {{ $t("common.confirm") }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup>
import { ref, computed, onMounted, watch } from "vue";
import { useI18n } from "vue-i18n";
import ModelLayerList from "@/components/ModelLayerList.vue";
import { useModelStore } from "@/stores/modelStore";
import { useSnackbarStore } from '@/stores/snackbarStore'
import NumericStepper from '@/components/NumericStepper.vue'

const { t } = useI18n();

const modelStore = useModelStore();

const snackbarStore = useSnackbarStore()

const currentModel = ref({
  input_shape: [1, 28, 28],
  num_classes: 10,
  layers: [],
});

const newModel = ref({ input_shape: [1, 28, 28], num_classes: 10, layers: [] });

const rawOutputShape = ref([0, 0, 0]);

const addLayerDialogVisible = ref(false);

const newLayerType = ref("convolve");

const currentLayerParams = ref({});

const clearNewModelDialogVisible = ref(false);

const displayOutputShape = computed(() => {
  if (!rawOutputShape.value || rawOutputShape.value.length !== 3)
    return [0, 0, 0];
  return [
    rawOutputShape.value[0],
    rawOutputShape.value[1],
    rawOutputShape.value[2],
  ];
});

function getLayerDisplayName(type) {
  return t(`layerTypes.${type}`, type);
}

function getParamLabel(key) {
  return t(`layerParams.${key}`, key);
}

async function saveAsBinaryFile(blob, suggestedName = 'model.bin') {
  if (window.showSaveFilePicker) {
    try {
      const fileHandle = await window.showSaveFilePicker({
        suggestedName: suggestedName,
        types: [
          {
            description: 'Model File',
            accept: { 'application/octet-stream': ['.bin'] },
          },
        ],
      });
      const writable = await fileHandle.createWritable();
      await writable.write(blob);
      await writable.close();
      return true;
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Save model file error', err);
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

function debounce(fn, delay) {
  let timer = null;
  return function (...args) {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      fn.apply(this, args);
    }, delay);
  };
}

async function updateOutputShape() {
  if (!newModel.value.layers.length) {
    rawOutputShape.value = [...newModel.value.input_shape];
    return;
  }
  let shape = newModel.value.input_shape;
  try {
    for (const layer of newModel.value.layers) {
      const { type, ...params } = layer;
      const res = await modelStore.fetchOutputShape(type, shape, params);
      shape = res;
    }
    rawOutputShape.value = shape;
  } catch (err) {
    console.error("Failed to compute output shape", err);
    rawOutputShape.value = [0, 0, 0];
  }
}

async function fetchParameterRangeForCurrentLayer() {
  let inShape = newModel.value.input_shape;
  if (newModel.value.layers.length > 0) {
    let shape = newModel.value.input_shape;
    try {
      for (const layer of newModel.value.layers) {
        const { type, ...params } = layer;
        const res = await modelStore.fetchOutputShape(type, shape, params);
        shape = res;
      }
      inShape = shape;
    } catch (err) {
      console.error(err);
    }
  }
  try {
    const paramDefs = await modelStore.fetchParameterRange(
      newLayerType.value,
      inShape,
    );
    const params = {};
    for (const [name, def] of Object.entries(paramDefs)) {
      params[name] = {
        type: def.type,
        value:
          def.default !== undefined
            ? def.default
            : def.type === "int"
              ? 0
              : 0.0,
        min: def.min,
        max: def.max,
        step: def.step || (def.type === "int" ? 1 : 0.001),
        options: def.enum || def.options,
        label: getParamLabel(name),
      };
    }
    currentLayerParams.value = params;
  } catch (err) {
    console.error("Failed to fetch parameter range", err);
    currentLayerParams.value = {};
  }
}

async function changeLayerType(type) {
  newLayerType.value = type;
  await fetchParameterRangeForCurrentLayer();
}

async function openAddLayerDialog() {
  newLayerType.value = "convolve";
  await fetchParameterRangeForCurrentLayer();
  addLayerDialogVisible.value = true;
}

function confirmAddLayer() {
  const newLayer = { type: newLayerType.value };
  for (const [name, p] of Object.entries(currentLayerParams.value)) {
    newLayer[name] = p.value;
  }
  newModel.value.layers.push(newLayer);
  addLayerDialogVisible.value = false;
  updateOutputShape();
}

function removeLayer(index) {
  if (index === newModel.value.layers.length - 1) {
    newModel.value.layers.splice(index, 1);
    updateOutputShape();
  }
}

function openClearNewModelConfirm() {
  clearNewModelDialogVisible.value = true;
}

function confirmClearNewModel() {
  clearNewModel();
  clearNewModelDialogVisible.value = false;
  snackbarStore.show(t("model.clearSuccess"), "primary");
}

async function applyToCurrent() {
  let finalLayers = JSON.parse(JSON.stringify(newModel.value.layers));
  if (
    finalLayers.length === 0 ||
    finalLayers[finalLayers.length - 1].type !== "softmax"
  ) {
    finalLayers.push({ type: "softmax" });
  }
  let shape = newModel.value.input_shape;
  try {
    for (const layer of finalLayers) {
      const { type, ...params } = layer;
      const res = await modelStore.fetchOutputShape(type, shape, params);
      shape = res;
    }
    const [c, h, w] = shape;
    if (h !== newModel.value.num_classes || c !== 1 || w !== 1) {
      snackbarStore.show(
        t("model.shapeValidationError", {
          expected: `[1, ${newModel.value.num_classes}, 1]`,
          actual: `[${c}, ${h}, ${w}]`,
        }),
        "error",
      );
      return;
    }
  } catch (err) {
    snackbarStore.show(t("model.shapeValidationFailed"), "error");
    return;
  }

  const config = {
    input_shape: newModel.value.input_shape,
    num_classes: newModel.value.num_classes,
    layers: finalLayers,
  };
  try {
    await modelStore.initModelFromJson(config);
    await loadCurrentModel();
    snackbarStore.show(t("model.applySuccess"), "primary");
  } catch (err) {
    snackbarStore.show(t("model.applyFailed"), "error");
  }
}

async function loadCurrentModel() {
  try {
    await modelStore.fetchModelJson();
    currentModel.value = JSON.parse(JSON.stringify(modelStore.architecture));
  } catch (err) {
    snackbarStore.show(t("model.loadCurrentFailed"), "error");
  }
}

async function saveArchitectureToClipboard() {
  if (!modelStore.ready) {
    snackbarStore.show(t("common.modelNotReady"), "error");
    return;
  }
  const data = {
    input_shape: currentModel.value.input_shape,
    num_classes: currentModel.value.num_classes,
    layers: currentModel.value.layers,
  };
  try {
    await navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    snackbarStore.show(t("model.copySuccess"), "primary");
  } catch (err) {
    snackbarStore.show(t("model.copyFailed"), "error");
  }
}

async function loadFromClipboard() {
  try {
    const text = await navigator.clipboard.readText();
    const data = JSON.parse(text);
    if (
      data.layers &&
      Array.isArray(data.layers) &&
      data.input_shape &&
      data.num_classes !== undefined
    ) {
      newModel.value = {
        input_shape: data.input_shape,
        num_classes: data.num_classes,
        layers: data.layers,
      };
      await updateOutputShape();
      snackbarStore.show(t("model.loadClipboardSuccess"), "primary");
    } else {
      snackbarStore.show(t("model.loadClipboardInvalid"), "error");
    }
  } catch (err) {
    snackbarStore.show(t("model.loadClipboardFailed"), "error");
  }
}

function clearNewModel() {
  newModel.value = { input_shape: [1, 28, 28], num_classes: 10, layers: [] };
  updateOutputShape();
}

async function loadModelFromFile() {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".bin";
  input.onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      await modelStore.loadModelBinary(file);
      await loadCurrentModel();
      snackbarStore.show(t("model.loadModelSuccess"), "primary");
    } catch (err) {
      snackbarStore.show(t("model.loadModelFailed"), "error");
    }
  };
  input.click();
}

async function downloadModelBin() {
  if (!modelStore.ready) {
    snackbarStore.show(t("common.modelNotReady"), "error");
    return;
  }
  try {
    const blob = await modelStore.fetchModelBinaryBlob();
    const saved = await saveAsBinaryFile(blob, "model.bin");
    if (saved) {
      snackbarStore.show(t("model.downloadStarted"), "primary");
    }
  } catch (err) {
    console.error("Download failed", err);
    snackbarStore.show(t("model.downloadFailed"), "error");
  }
}

watch(
  () => newModel.value.layers,
  () => updateOutputShape(),
  { deep: true },
);

watch(
  () => newModel.value.input_shape,
  () => updateOutputShape(),
  { deep: true },
);

onMounted(async () => {
  await modelStore.fetchLayerTypes();
  await loadCurrentModel();
  await updateOutputShape();
});
</script>

<style scoped>
.param-field {
  border-radius: 12px;
}

.param-field :deep(.v-field__input) {
  padding: 0;
}

.param-field :deep(.v-field__outline) {
  border-radius: inherit;
}

.field-trigger-btn {
  justify-content: flex-center;
  text-transform: none;
  font-weight: normal;
  letter-spacing: normal;
  min-height: 40px;
}
</style>