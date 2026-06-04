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
        <v-card-title>{{ $t("evaluate.title") }}</v-card-title>
        <v-divider></v-divider>
        <v-card-text>
          <v-menu offset-y>
            <template v-slot:activator="{ props }">
              <v-text-field v-model="selectedDatasetName" :label="$t('evaluate.selectDataset')" readonly
                variant="outlined" density="compact" :disabled="isEvaluating" v-bind="props"
                append-inner-icon="mdi-menu-down"></v-text-field>
            </template>
            <v-list class="frosted-glass">
              <v-list-item v-for="ds in datasetNames" :key="ds" :value="ds" @click="selectDataset(ds)">
                <v-list-item-title>{{ ds }}</v-list-item-title>
              </v-list-item>
            </v-list>
          </v-menu>
          <v-row class="mb-3">
            <v-col cols="12" md="6">
              <NumericStepper v-model="offsetValue" :label="$t('common.offsetLabel')"
                :disabled="isEvaluating || !selectedDataset" :min="0" :max="totalSamplesInDataset" :step="1" />
            </v-col>
            <v-col cols="12" md="6">
              <NumericStepper v-model="countValue" :label="$t('common.countLabel')"
                :disabled="isEvaluating || !selectedDataset" :min="0" :max="maxCount" :step="1" />
            </v-col>
          </v-row>
          <v-divider></v-divider>
          <v-row class="mt-3 mb-0">
            <v-col cols="12" md="12">
              <NumericStepper v-model="batchSize" :label="$t('evaluate.batchSizeLabel')" :disabled="isEvaluating"
                :min="1" :max="256" :step="1" />
            </v-col> </v-row></v-card-text>
        <v-divider></v-divider>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn color="primary" :disabled="isEvaluating || !selectedDataset" class="mr-2" prepend-icon="mdi-play"
            @click="startEvaluation">
            {{ $t("common.start") }}
          </v-btn>
          <v-btn color="error" :disabled="!isEvaluating" prepend-icon="mdi-stop" @click="cancelEvaluation">
            {{ $t("common.cancel") }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-col>

    <v-col cols="12" md="12">
      <v-card class="frosted-glass">
        <v-card-title>
          {{ $t("evaluate.statisticsTitle") }}
        </v-card-title>
        <v-divider></v-divider>
        <v-card-text>
          <div class="mb-4">
            <v-progress-linear :model-value="progressPercent" height="20" :color="isEvaluating ? 'primary' : 'grey'"
              striped class="mb-2">
              <template v-slot:default="{ value }">
                {{ Math.round(value) }}% ({{
                  evaluationStore.status.processed_samples
                }}/{{ evaluationStore.status.total_samples }})
              </template>
            </v-progress-linear>
          </div>

          <v-divider class="my-3"></v-divider>

          <v-row>
            <v-col cols="12" md="6">
              <StatCard :title="$t('common.realtimeAccuracy')" :series="realtimeSeries"
                :xaxisLabels="evaluationStore.realtimeLabels" :color="primaryColor" :tickAmount="8" :smoothWindow="Math.max(evaluationStore.realtimeLabels.length / 5, 3)
                  " />
            </v-col>
            <v-col cols="12" md="6">
              <StatCard :title="$t('common.historyAccuracy')" :series="historyFinalSeries"
                :xaxisLabels="evaluationStore.historyEpochLabels" :color="primaryColor" :tickAmount="8" :smoothWindow="Math.max(evaluationStore.historyEpochLabels.length / 5, 3)
                  " />
            </v-col>
          </v-row>

          <v-row>
            <v-col cols="12">
              <v-card class="frosted-glass mt-2">
                <v-card-title>{{
                  $t("evaluate.perClassHistoryTitle")
                }}</v-card-title>
                <v-divider></v-divider>
                <v-card-text>
                  <VueApexCharts :key="perClassChartKey" :options="perClassChartOptions" :series="perClassSeries"
                    type="line" height="350" />
                </v-card-text>
              </v-card>
            </v-col>
          </v-row>
        </v-card-text>
        <v-divider></v-divider>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn variant="text" color="error" prepend-icon="mdi-delete" @click="openClearHistoryDialog">
            {{ $t("common.clearHistory") }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-col>
  </v-row>

  <v-dialog v-model="clearHistoryDialogVisible" max-width="400px">
    <v-card class="dialog-card">
      <v-card-title>{{ $t("common.clearHistory") }}</v-card-title>
      <v-divider></v-divider>
      <v-card-text class="pt-4">
        {{ $t("common.clearHistoryText") }}
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn variant="text" @click="clearHistoryDialogVisible = false">
          {{ $t("common.cancel") }}
        </v-btn>
        <v-btn color="error" @click="confirmClearHistory">
          {{ $t("common.confirm") }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup>
import { ref, computed, onMounted, watch } from "vue";
import { useTheme } from "vuetify";
import { useI18n } from "vue-i18n";
import { useEvaluationStore } from "@/stores/evaluationStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { useModelStore } from "@/stores/modelStore";
import StatCard from "@/components/StatCard.vue";
import VueApexCharts from "vue3-apexcharts";
import { useSnackbarStore } from '@/stores/snackbarStore'
import NumericStepper from '@/components/NumericStepper.vue'

const theme = useTheme();

const { t, locale } = useI18n();

const evaluationStore = useEvaluationStore();

const datasetStore = useDatasetStore();

const modelStore = useModelStore();

const snackbarStore = useSnackbarStore()

const perClassChartKey = ref(0);

const selectedDataset = ref("");

const selectedDatasetName = ref("");

const offsetValue = ref(0);

const countValue = ref(0);

const batchSize = ref(32);

const clearHistoryDialogVisible = ref(false);

const colorPalette = [
  "#ABE7E5",
  "#C2E5E9",
  "#CDE6E3",
  "#E4DFE6",
  "#FBD8E3",
  "#F7E9E8",
  "#E7E8ED",
  "#CDE3EE",
  "#BFE3EF",
  "#AEDCF4",
];

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

const isEvaluating = computed(() => evaluationStore.status.state === "running");

const progressPercent = computed(() => {
  const state = evaluationStore.status.state;
  if (state !== "running") return 0;
  const total = evaluationStore.status.total_samples;
  if (!total) return 0;
  return (evaluationStore.status.processed_samples / total) * 100;
});

const realtimeSeries = computed(() => [
  {
    name: t("common.realtimeAccuracy"),
    data: evaluationStore.realtimeAccData,
  },
]);

const historyFinalSeries = computed(() => [
  {
    name: t("common.historyAccuracy"),
    data: evaluationStore.historyFinalAccuracies,
  },
]);

const perClassSeries = computed(() => {
  const series = [];
  for (let i = 0; i < evaluationStore.historyPerClassAccuracies.length; i++) {
    const perClass = evaluationStore.historyPerClassAccuracies[i];
    if (!perClass) continue;
    const data = [];
    for (let digit = 0; digit <= 9; digit++) {
      const acc = perClass[digit.toString()];
      data.push(acc !== null && acc !== undefined ? acc : 0);
    }
    series.push({
      name: t("evaluate.perClassName", { index: i + 1 }),
      data: data,
    });
  }
  return series;
});

const perClassChartOptions = computed(() => ({
  chart: {
    type: "line",
    height: 300,
    background: "transparent",
    toolbar: { show: false },
    zoom: { enabled: true, autoScaleYaxis: true },
  },
  colors: colorPalette,
  stroke: { curve: "smooth", width: 2 },
  xaxis: {
    categories: evaluationStore.perClassCategories,
    labels: {
      hideOverlappingLabels: true,
      trim: true,
      maxHeight: 60,
      style: {
        fontSize: "10px",
        colors: "#666",
      },
    },
  },
  yaxis: {
    labels: { style: { colors: "#666" } },
  },
  tooltip: { theme: "light" },
  grid: { borderColor: "#e0e0e0" },
}));

function selectDataset(ds) {
  selectedDataset.value = ds;
  selectedDatasetName.value = ds;
  offsetValue.value = 0;
  countValue.value = 0;
}

async function startEvaluation() {
  if (!selectedDataset.value) {
    snackbarStore.show(t("common.noDataset"), "error");
    return;
  }
  if (!modelStore.ready) {
    snackbarStore.show(t("common.modelNotReady"), "error");
    return;
  }
  try {
    await evaluationStore.startEvaluation(
      selectedDataset.value,
      offsetValue.value,
      countValue.value,
      batchSize.value,
    );
    snackbarStore.show(t("evaluate.startSuccess"));
  } catch (err) {
    console.error(err);
    snackbarStore.show(t("evaluate.errors.startFailed"), "error");
  }
}

async function cancelEvaluation() {
  try {
    await evaluationStore.cancelEvaluation();
    snackbarStore.show(t("evaluate.cancelSuccess"));
  } catch (err) {
    console.error(err);
    snackbarStore.show(t("evaluate.errors.cancelFailed"), "error");
  }
}

function openClearHistoryDialog() {
  clearHistoryDialogVisible.value = true;
}

function confirmClearHistory() {
  evaluationStore.clearHistory();
  clearHistoryDialogVisible.value = false;
}

watch(locale, () => {
  perClassChartKey.value++;
});

watch(
  () => selectedDataset.value,
  () => {
    offsetValue.value = 0;
    countValue.value = 0;
  },
);

watch(totalSamplesInDataset, () => {
  let newOffset = offsetValue.value;
  if (newOffset > totalSamplesInDataset.value) newOffset = totalSamplesInDataset.value;
  if (newOffset < 0) newOffset = 0;
  offsetValue.value = newOffset;

  let newCount = countValue.value;
  const maxC = maxCount.value;
  if (newCount > maxC) newCount = maxC;
  if (newCount < 0) newCount = 0;
  countValue.value = newCount;
});

onMounted(async () => {
  await datasetStore.fetchDatasets();
  await modelStore.checkReady();
  await evaluationStore.fetchEvaluationStatus();
});
</script>