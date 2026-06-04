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
  <v-card class="frosted-glass stat-card">
    <v-card-title class="card-title">{{ title }}</v-card-title>
    <v-card-text>
      <VueApexCharts v-if="chartSeries && chartSeries.length" :key="`${color}-${refreshKey}`" :options="chartOptions"
        :series="chartSeries" type="line" height="300" />
      <div v-else class="text-center pa-4">{{ $t("common.noData") }}</div>
    </v-card-text>
  </v-card>
</template>

<script setup>
import { computed, watch, ref } from "vue";
import { useI18n } from "vue-i18n";

const { t, locale } = useI18n();

const refreshKey = ref(0);

watch(locale, () => {
  refreshKey.value++;
});

const props = defineProps({
  title: String,
  series: Array,
  xaxisLabels: Array,
  color: {
    type: String,
    default: "#666",
  },
  tickAmount: {
    type: Number,
    default: 8,
  },
  smoothWindow: {
    type: Number,
    default: 5,
  },
});

function getDecimalPlaces(num) {
  const str = num.toString();
  const match = str.match(/\.(\d+)$/);
  return match ? match[1].length : 0;
}

function computeMovingAverage(data, window, decimalPlaces) {
  if (!data || data.length === 0) return [];
  const half = Math.floor(window / 2);
  const result = [];
  for (let i = 0; i < data.length; i++) {
    let sum = 0;
    let count = 0;
    for (let j = i - half; j <= i + half; j++) {
      if (j >= 0 && j < data.length) {
        sum += data[j];
        count++;
      }
    }
    const rawAvg = sum / count;
    const rounded = parseFloat(rawAvg.toFixed(decimalPlaces));
    result.push(rounded);
  }
  return result;
}

const originalSeries = computed(() => {
  if (!props.series || props.series.length === 0) return null;
  return {
    name: props.series[0].name,
    data: props.series[0].data,
    type: "line",
  };
});

const originalDecimalPlaces = computed(() => {
  const data = originalSeries.value?.data;
  if (!data || data.length === 0) return 2;
  const firstValid = data.find((v) => v !== null && v !== undefined);
  if (firstValid === undefined) return 2;
  return getDecimalPlaces(firstValid);
});

const smoothedSeries = computed(() => {
  if (!originalSeries.value) return null;
  const originalData = originalSeries.value.data;
  if (!originalData || originalData.length < 2) return null;
  const smoothedData = computeMovingAverage(
    originalData,
    props.smoothWindow,
    originalDecimalPlaces.value,
  );
  return {
    name: `${t("common.average")}`,
    data: smoothedData,
    type: "area",
  };
});

const chartSeries = computed(() => {
  const seriesList = [];
  if (originalSeries.value) {
    seriesList.push(originalSeries.value);
  }
  if (smoothedSeries.value) {
    seriesList.push(smoothedSeries.value);
  }
  return seriesList;
});

const chartOptions = computed(() => ({
  chart: {
    height: 300,
    background: "transparent",
    toolbar: { show: false },
    zoom: { enabled: true, autoScaleYaxis: true },
  },
  stroke: {
    curve: "smooth",
    width: 2,
  },
  colors: [`${props.color}50`, `${props.color}90`],
  fill: {
    type: "gradient",
    gradient: {
      shadeIntensity: 1,
      opacityFrom: 0.3,
      opacityTo: 0.1,
      colorStops: [
        { offset: 0, color: props.color, opacity: 0.3 },
        { offset: 100, color: props.color, opacity: 0.1 },
      ],
    },
  },
  xaxis: {
    categories: props.xaxisLabels,
    labels: {
      hideOverlappingLabels: true,
      trim: true,
      maxHeight: 60,
      style: {
        fontSize: "10px",
        colors: "#666",
      },
    },
    tickAmount: props.tickAmount,
  },
  yaxis: {
    labels: { style: { colors: "#666" } },
  },
  tooltip: { theme: "light" },
  grid: { borderColor: "#e0e0e0" },
}));
</script>

<style scoped>
.stat-card {
  height: 100%;
}

.card-title {
  padding: 25px 25px 0 25px;
  font-size: 1.2rem;
  font-weight: 500;
}

:deep(.apexcharts-canvas) {
  background: transparent !important;
}
</style>