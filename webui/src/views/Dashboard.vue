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
    <v-col cols="12" md="6">
      <v-card class="frosted-glass" style="height: 100%">
        <v-card-title>{{ $t("dashboard.serverStatus") }}</v-card-title>
        <v-divider></v-divider>
        <v-card-text class="text-center">
          <v-icon :color="serverHealth === 'ok' ? 'primary' : 'error'" size="48" class="mb-2">
            {{
              serverHealth === "ok" ? "mdi-check-circle" : "mdi-close-circle"
            }}
          </v-icon>
          <div>
            {{
              serverHealth === "ok"
                ? $t("dashboard.serviceOk")
                : $t("dashboard.serviceError")
            }}
          </div>
          <div class="text-grey">
            {{ $t("dashboard.uptime", { time: formattedUptime }) }}
          </div>
        </v-card-text>
      </v-card>
    </v-col>

    <v-col cols="12" md="6">
      <v-card class="frosted-glass" style="height: 100%">
        <v-card-title>{{ $t("dashboard.modelStatus") }}</v-card-title>
        <v-divider></v-divider>
        <v-card-text class="text-center">
          <v-icon :color="modelStore.ready ? 'primary' : 'error'" size="48" class="mb-2">
            {{ modelStore.ready ? "mdi-check-circle" : "mdi-close-circle" }}
          </v-icon>
          <div>
            {{
              modelStore.ready
                ? $t("common.modelLoaded")
                : $t("common.modelNotLoaded")
            }}
          </div>
          <div class="text-grey">
            {{ $t("dashboard.modelSyncHint") }}
          </div>
        </v-card-text>
      </v-card>
    </v-col>

    <v-col cols="12" md="12">
      <v-card class="frosted-glass">
        <v-card-title>
          {{ $t("dashboard.trainingStatus") }}
        </v-card-title>
        <v-divider></v-divider>
        <v-card-text v-if="!isTrainingRunning" class="text-center">
          <v-icon icon="mdi-sleep" size="48" color="grey" class="mb-2"></v-icon>
          <div>{{ $t("dashboard.trainingIdle") }}</div>
          <div class="text-grey">
            {{ $t("dashboard.trainingIdleHint") }}
          </div>
        </v-card-text>
        <v-card-text v-else>
          <v-row>
            <v-col cols="12" md="6">
              <StatCard :title="$t('common.realtimeLoss')" :series="lossSeries"
                :xaxisLabels="trainingStore.realtimeLabels" :color="primaryColor" :smoothWindow="Math.max(trainingStore.realtimeLabels.length / 5, 3)
                  " />
            </v-col>
            <v-col cols="12" md="6">
              <StatCard :title="$t('common.realtimeAccuracy')" :series="accSeries"
                :xaxisLabels="trainingStore.realtimeLabels" :color="primaryColor" :smoothWindow="Math.max(trainingStore.realtimeLabels.length / 5, 3)
                  " />
            </v-col>
          </v-row>
        </v-card-text>
      </v-card>
    </v-col>
  </v-row>

  <v-row>
    <v-col cols="12" md="12">
      <v-card class="frosted-glass">
        <v-card-title>
          {{ $t("dashboard.trainingHistory") }}
        </v-card-title>
        <v-divider></v-divider>
        <v-card-text>
          <v-row>
            <v-col cols="12" md="6">
              <StatCard :title="$t('common.historyLoss')" :series="historyLossSeries"
                :xaxisLabels="trainingStore.persistedEpochLabels" :color="primaryColor" :smoothWindow="Math.max(trainingStore.persistedEpochLabels.length / 5, 3)
                  " />
            </v-col>
            <v-col cols="12" md="6">
              <StatCard :title="$t('common.historyAccuracy')" :series="historyAccSeries"
                :xaxisLabels="trainingStore.persistedEpochLabels" :color="primaryColor" :smoothWindow="Math.max(trainingStore.persistedEpochLabels.length / 5, 3)
                  " />
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

  <v-row>
    <v-col cols="12" md="12">
      <v-card class="frosted-glass">
        <v-card-title>{{ $t("dashboard.projectDescription") }}</v-card-title>
        <v-divider></v-divider>
        <v-card-text>
          <v-row>
            <v-col cols="12" md="12">
              <v-card class="frosted-glass pa-4">
                <div v-if="loadingReadme" class="text-center">
                  <v-progress-circular indeterminate color="primary"></v-progress-circular>
                  <div class="mt-2">{{ $t("common.loading") }}</div>
                </div>
                <div v-else-if="readmeError" class="text-center text-error">
                  {{ $t("dashboard.readmeLoadError") }}
                </div>
                <div v-else ref="readmeContainer" v-html="readmeHtml" class="markdown-body transparent"></div>
              </v-card>
            </v-col>
          </v-row>
        </v-card-text>
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
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from "vue";
import { marked } from "marked";
import { useTheme } from "vuetify";
import { useModelStore } from "@/stores/modelStore";
import { useTrainingStore } from "@/stores/trainingStore";
import { getHealth, getServerStatus } from "@/api";
import StatCard from "@/components/StatCard.vue";
import "github-markdown-css/github-markdown.css";
import { useI18n } from "vue-i18n";

const modelStore = useModelStore();

const trainingStore = useTrainingStore();

const theme = useTheme();

const { t } = useI18n();

const serverHealth = ref("unknown");

const uptimeSeconds = ref(0);

const loadingServer = ref(false);

let uptimeTimer = null;

const clearHistoryDialogVisible = ref(false);

const readmeHtml = ref("");

const loadingReadme = ref(true);

const readmeError = ref(false);

const readmeContainer = ref(null);

const formattedUptime = computed(() => {
  const seconds = uptimeSeconds.value;
  if (seconds === 0) return "?";
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  if (days > 0) return `${days}d ${hours}h ${minutes}m`;
  if (hours > 0) return `${hours}h ${minutes}m ${secs}s`;
  if (minutes > 0) return `${minutes}m ${secs}s`;
  return `${secs}s`;
});

const lossSeries = computed(() => [
  { name: t("common.realtimeLoss"), data: trainingStore.realtimeLossData },
]);

const accSeries = computed(() => [
  { name: t("common.realtimeAccuracy"), data: trainingStore.realtimeAccData },
]);

const historyLossSeries = computed(() => [
  { name: t("common.historyLoss"), data: trainingStore.persistedLossData },
]);

const historyAccSeries = computed(() => [
  { name: t("common.historyAccuracy"), data: trainingStore.persistedAccData },
]);

const isTrainingRunning = computed(
  () => trainingStore.status.state === "running",
);

const primaryColor = computed(() => theme.global.current.value.colors.primary);

async function refreshServerStatus() {
  loadingServer.value = true;
  try {
    const healthRes = await getHealth();
    serverHealth.value = healthRes.data.status;
    const statusRes = await getServerStatus();
    uptimeSeconds.value = statusRes.data.uptime_seconds;
  } catch (err) {
    console.error("Failed to get server status", err);
    serverHealth.value = "error";
  } finally {
    loadingServer.value = false;
  }
}

function startUptimeRefresh() {
  if (uptimeTimer) clearInterval(uptimeTimer);
  uptimeTimer = setInterval(() => {
    if (serverHealth.value === "ok") {
      refreshServerStatus();
    }
  }, 1000);
}

function openClearHistoryDialog() {
  clearHistoryDialogVisible.value = true;
}

function confirmClearHistory() {
  trainingStore.clearPersistedHistory();
  clearHistoryDialogVisible.value = false;
}

function handleAnchorClick(e) {
  e.preventDefault();
  const href = e.currentTarget.getAttribute("href");
  if (!href) return;

  if (href.startsWith("#")) {
    const targetId = href.slice(1);
    const targetElement = document.getElementById(targetId);
    if (targetElement) {
      targetElement.scrollIntoView({ behavior: "smooth" });
    }
    return;
  }
}

function bindReadmeAnchors() {
  if (!readmeContainer.value) return;
  const anchorLinks = readmeContainer.value.querySelectorAll('a[href^="#"]');
  anchorLinks.forEach((link) => {
    link.removeEventListener("click", handleAnchorClick);
    link.addEventListener("click", handleAnchorClick);
  });
}

function unbindReadmeAnchors() {
  if (!readmeContainer.value) return;
  const anchorLinks = readmeContainer.value.querySelectorAll('a[href^="#"]');
  anchorLinks.forEach((link) => {
    link.removeEventListener("click", handleAnchorClick);
  });
}

onMounted(async () => {
  await refreshServerStatus();
  startUptimeRefresh();

  await modelStore.checkReady();
  await trainingStore.fetchTrainingStatus();

  try {
    const response = await fetch(
      "https://raw.githubusercontent.com/Chrollis/Mint-C/main/README.md",
    );
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const markdown = await response.text();
    marked.setOptions({ breaks: true, gfm: true });
    readmeHtml.value = await marked.parse(markdown);
  } catch (err) {
    readmeError.value = true;
  } finally {
    loadingReadme.value = false;
  }
});

watch(readmeHtml, async () => {
  await nextTick();
  bindReadmeAnchors();
});

onUnmounted(() => {
  if (uptimeTimer) clearInterval(uptimeTimer);
  unbindReadmeAnchors();
});
</script>