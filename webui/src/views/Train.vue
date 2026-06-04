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
    <v-col cols="12" md="12">
      <v-card class="frosted-glass">
        <v-card-title>{{ $t("train.configTitle") }}</v-card-title>
        <v-divider></v-divider>
        <v-card-text>
          <v-menu offset-y>
            <template v-slot:activator="{ props }">
              <v-text-field v-model="selectedDatasetName" :label="$t('train.selectDataset')" readonly variant="outlined"
                density="compact" :disabled="isTrainingActive" v-bind="props"
                append-inner-icon="mdi-menu-down"></v-text-field>
            </template>
            <v-list class="frosted-glass">
              <v-list-item v-for="ds in datasetNames" :key="ds" :value="ds" @click="
                selectedDataset = ds;
              selectedDatasetName = ds;
              ">
                <v-list-item-title>{{ ds }}</v-list-item-title>
              </v-list-item>
            </v-list>
          </v-menu>
          <v-row class="mb-3">
            <v-col cols="12" md="6">
              <NumericStepper v-model="offsetValue" :label="$t('common.offsetLabel')"
                :disabled="isTrainingActive || !selectedDataset" :min="0" :max="totalSamplesInDataset" :step="1" />
            </v-col>
            <v-col cols="12" md="6">
              <NumericStepper v-model="countValue" :label="$t('common.countLabel')"
                :disabled="isTrainingActive || !selectedDataset" :min="0" :max="maxCount" :step="1" />
            </v-col>
          </v-row>
          <v-divider></v-divider>
          <div v-if="Object.keys(hyperparamsDefs).length" class="mt-6">
            <div v-for="(param, name) in hyperparamsDefs" :key="name" class="mb-3">
              <div v-if="param.type === 'string'">
                <v-menu offset-y>
                  <template v-slot:activator="{ props }">
                    <v-btn variant="outlined" block v-bind="props">
                      {{ param.value }}
                    </v-btn>
                  </template>
                  <v-list class="frosted-glass">
                    <v-list-item v-for="opt in param.options" :key="opt" @click="param.value = opt">
                      <v-list-item-title>{{ opt }}</v-list-item-title>
                    </v-list-item>
                  </v-list>
                </v-menu>
              </div>
              <div v-else-if="param.type === 'int' || param.type === 'double'">
                <NumericStepper v-model="param.value" :label="getParamLabel(name)" :min="param.min" :max="param.max"
                  :step="param.step" :disabled="isTrainingActive" />
              </div>
            </div>
          </div>
          <div v-else class="text-center text-grey pa-4">
            {{ $t("train.loadingHyperparams") }}
          </div>
        </v-card-text>
        <v-divider></v-divider>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn color="primary" :disabled="isTrainingActive || !selectedDataset" class="mr-2" prepend-icon="mdi-play"
            @click="startTraining">
            {{ $t("common.start") }}
          </v-btn>
          <v-btn color="error" :disabled="!isTrainingActive" prepend-icon="mdi-stop" @click="cancelTraining">
            {{ $t("common.cancel") }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-col>

    <v-col cols="12" md="12">
      <v-card class="frosted-glass">
        <v-card-title>{{ $t("train.monitorTitle") }}</v-card-title>
        <v-divider></v-divider>
        <v-card-text>
          <div class="mb-4">
            <v-progress-linear :model-value="displayProgress" height="20" :color="isTrainingActive ? 'primary' : 'grey'"
              striped class="mb-2">
              <template v-slot:default="{ value }">
                {{ Math.round(value) }}% {{ $t("train.progressWhileEpoch") }}
                {{ currentEpoch }}/{{ totalEpochs }}
              </template>
            </v-progress-linear>
          </div>

          <v-divider class="my-3"></v-divider>

          <v-row>
            <v-col cols="12" md="6">
              <StatCard :title="t('common.realtimeLoss')" :series="lossSeries"
                :xaxisLabels="trainingStore.realtimeLabels" :color="primaryColor" :tickAmount="8" :smoothWindow="Math.max(trainingStore.realtimeLabels.length / 5, 3)
                  " />
            </v-col>
            <v-col cols="12" md="6">
              <StatCard :title="t('common.realtimeAccuracy')" :series="accSeries"
                :xaxisLabels="trainingStore.realtimeLabels" :color="primaryColor" :tickAmount="8" :smoothWindow="Math.max(trainingStore.realtimeLabels.length / 5, 3)
                  " />
            </v-col>
          </v-row>
        </v-card-text>
      </v-card>
    </v-col>
  </v-row>

  <v-dialog v-model="trainingFinishedDialog" max-width="400px">
    <v-card class="frosted-glass dialog-card">
      <v-card-title>{{ $t("train.finishDialogTitle") }}</v-card-title>
      <v-divider></v-divider>
      <v-card-text>
        {{ $t("train.finishDialogText") }}
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn variant="text" @click="discardModel" :loading="accepting">
          {{ $t("train.discardBtn") }}
        </v-btn>
        <v-btn color="primary" @click="acceptModel" :loading="accepting">
          {{ $t("train.acceptBtn") }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup>

import { ref, computed, onMounted, watch } from "vue";
import { useTheme } from "vuetify";
import { useI18n } from "vue-i18n";
import { useTrainingStore } from "@/stores/trainingStore";
import { useDatasetStore } from "@/stores/datasetStore";
import StatCard from "@/components/StatCard.vue";
import { useModelStore } from "@/stores/modelStore";
import { useSnackbarStore } from '@/stores/snackbarStore'
import NumericStepper from '@/components/NumericStepper.vue'

const theme = useTheme();

const trainingStore = useTrainingStore();

const datasetStore = useDatasetStore();

const modelStore = useModelStore();

const snackbarStore = useSnackbarStore();

const { t } = useI18n();

const selectedDataset = ref("");

const selectedDatasetName = ref("");

const hyperparamsDefs = ref({});

const offsetValue = ref(0);

const countValue = ref(0);

const trainingFinishedDialog = ref(false);

const snackbarVisible = ref(false);

const snackbarMessage = ref("");

const snackbarColor = ref("primary");

const accepting = ref(false);

const primaryColor = computed(() => theme.global.current.value.colors.primary);

const datasetNames = computed(() => datasetStore.datasets.map((ds) => ds.name));

const totalSamplesInDataset = computed(() => {
  const ds = datasetStore.datasets.find(
    (d) => d.name === selectedDataset.value,
  );
  return ds?.sample_count || 0;
});

const maxCount = computed(() => {
  const total = totalSamplesInDataset.value;
  const offset = offsetValue.value;
  if (offset > total) return 0;
  return total - offset;
});

const isTrainingActive = computed(
  () => trainingStore.status.state === "running",
);

const currentEpoch = computed(() => trainingStore.status.current_epoch);

const totalEpochs = computed(() => trainingStore.status.total_epochs);

const processedSamples = computed(() => trainingStore.status.processed_samples);

const totalSamples = computed(() => trainingStore.status.total_samples);

const displayProgress = computed(() => {
  const state = trainingStore.status.state;
  if (state === "finished" || state === "cancelled" || state === "error")
    return 0;
  if (totalSamples.value === 0) return 0;
  return (
    ((processedSamples.value % totalSamples.value) / totalSamples.value) * 100
  );
});

const lossSeries = computed(() => [
  { name: t("common.realtimeLoss"), data: trainingStore.realtimeLossData },
]);

const accSeries = computed(() => [
  { name: t("common.realtimeAccuracy"), data: trainingStore.realtimeAccData },
]);

const hyperparamsForApi = computed(() => {
  const result = {};
  for (const [name, param] of Object.entries(hyperparamsDefs.value)) {
    result[name] = param.value;
  }
  return result;
});

function getParamLabel(name) {
  return t(`train.hyperparams.${name}`);
}

async function loadHyperparamsRange() {
  await trainingStore.fetchHyperparamsRange();
  const range = trainingStore.hyperparamsRange;
  if (Object.keys(range).length) {
    const defs = {};
    for (const [name, def] of Object.entries(range)) {
      defs[name] = {
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
        options: def.options || [],
      };
    }
    hyperparamsDefs.value = defs;
  }
}

async function startTraining() {
  if (!selectedDataset.value) {
    snackbarStore.show(t("common.noDataset"), "error");
    return;
  }
  if (!modelStore.ready) {
    snackbarStore.show(t("common.modelNotReady"), "error");
    return;
  }
  try {
    await trainingStore.startTraining(
      selectedDataset.value,
      hyperparamsForApi.value,
      offsetValue.value,
      countValue.value,
    );
    snackbarStore.show(t("train.startSuccess"));
  } catch (err) {
    console.error(err);
    snackbarStore.show(t("train.errors.startFailed"), "error");
  }
}

async function cancelTraining() {
  try {
    await trainingStore.cancelTraining();
    snackbarStore.show(t("train.cancelSuccess"));
  } catch (err) {
    console.error(err);
    snackbarStore.show(t("train.errors.cancelFailed"), "error");
  }
}

async function acceptModel() {
  accepting.value = true;
  try {
    await trainingStore.acceptResult(true);
    trainingFinishedDialog.value = false;
    await trainingStore.fetchTrainingStatus();
  } catch (err) {
    console.error("Failed to accept the model", err);
  } finally {
    accepting.value = false;
  }
}

async function discardModel() {
  accepting.value = true;
  try {
    await trainingStore.acceptResult(false);
    trainingFinishedDialog.value = false;
    await trainingStore.fetchTrainingStatus();
  } catch (err) {
    console.error("Failed to discard the model", err);
  } finally {
    accepting.value = false;
  }
}

watch(selectedDataset, () => {
  offsetValue.value = 0;
  countValue.value = 0;
});

watch(totalSamplesInDataset, () => {
  let newOffset = offsetValue.value;
  const maxOffset = totalSamplesInDataset.value;
  if (newOffset > maxOffset) newOffset = maxOffset;
  if (newOffset < 0) newOffset = 0;
  offsetValue.value = newOffset;

  let newCount = countValue.value;
  const maxC = maxCount.value;
  if (newCount > maxC) newCount = maxC;
  if (newCount < 0) newCount = 0;
  countValue.value = newCount;
});

watch(
  () => trainingStore.status.state,
  (newState) => {
    trainingFinishedDialog.value = newState === "finished";
  },
  { immediate: true },
);

onMounted(async () => {
  await datasetStore.fetchDatasets();
  await loadHyperparamsRange();
  await trainingStore.fetchTrainingStatus();
});
</script>