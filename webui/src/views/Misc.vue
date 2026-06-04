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
        <v-card-title>
          {{ $t("misc.serverConfigTitle") }}
        </v-card-title>
        <v-divider></v-divider>
        <v-card-text>
          <v-text-field v-model="localConfig.port" :label="$t('misc.portLabel')" type="number" variant="outlined"
            density="compact" class="port-line"></v-text-field>
          <v-text-field v-model="localConfig.log_dir" :label="$t('misc.logDirLabel')" variant="outlined"
            density="compact"></v-text-field>
          <v-text-field v-model="localConfig.temp_dir" :label="$t('misc.tempDirLabel')" variant="outlined"
            density="compact"></v-text-field>
          <v-text-field v-model="localConfig.dataset_dir" :label="$t('misc.datasetDirLabel')" variant="outlined"
            density="compact"></v-text-field>
        </v-card-text>
        <v-divider></v-divider>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn @click="saveConfig" :loading="configStore.saving" prepend-icon="mdi-content-save" class="mr-2">
            {{ $t("common.save") }} </v-btn><v-btn variant="text" color="primary" @click="refreshConfig"
            :loading="configStore.loading" prepend-icon="mdi-refresh">
            {{ $t("common.refresh") }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-col>

    <v-col cols="12" md="12">
      <v-card class="frosted-glass" style="height: 100%">
        <v-card-title>
          {{ $t("misc.systemLogsTitle") }}
        </v-card-title>
        <v-divider></v-divider>
        <v-card-text class="pa-0 console-container">
          <div class="console-log" ref="consoleLogRef">
            <div v-for="(line, idx) in logStore.logs" :key="idx" class="log-line" :class="getLogClass(line)">
              {{ line }}
            </div>
            <div v-if="logStore.loading" class="text-center pa-2 text-grey">
              {{ $t("common.loading") }}
            </div>
            <div v-else-if="logStore.logs.length === 0" class="text-center pa-2 text-grey">
              {{ $t("common.noData") }}
            </div>
          </div>
        </v-card-text>
        <v-divider></v-divider>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-menu offset-y>
            <template v-slot:activator="{ props }">
              <v-btn variant="text" v-bind="props" prepend-icon="mdi-filter-variant" class="mr-2">
                {{ $t("misc.logLevelLabel") }}:
                {{ getLevelDisplayName(logStore.currentLevel) }}
              </v-btn>
            </template>
            <v-list class="frosted-glass">
              <v-list-item v-for="lvl in logLevels" :key="lvl" :value="lvl" @click="changeLogLevel(lvl)">
                <v-list-item-title>{{
                  getLevelDisplayName(lvl)
                }}</v-list-item-title>
              </v-list-item>
            </v-list>
          </v-menu>

          <v-menu offset-y>
            <template v-slot:activator="{ props }">
              <v-btn variant="text" v-bind="props" prepend-icon="mdi-format-list-numbered" class="mr-2">
                {{ $t("misc.logCountLabel") }}: {{ logStore.currentCount }}
              </v-btn>
            </template>
            <v-list class="frosted-glass">
              <v-list-item v-for="cnt in [100, 200, 500, 1000]" :key="cnt" :value="cnt" @click="changeLogCount(cnt)">
                <v-list-item-title>{{ cnt }}</v-list-item-title>
              </v-list-item>
            </v-list>
          </v-menu>
          <v-btn variant="text" color="primary" @click="refreshLogs" :loading="logStore.loading"
            prepend-icon="mdi-refresh">
            {{ $t("common.refresh") }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-col>

    <v-col cols="12" md="12">
      <v-card class="frosted-glass">
        <v-card-title>
          {{ $t("misc.aboutTitle") }}
        </v-card-title>
        <v-divider></v-divider>
        <v-card-text>
          <p>Mint-C - {{ $t("misc.aboutDesc") }}</p>
          <p>
            {{ $t("misc.copyright") }} -
            {{ $t("misc.licenseMsg") }}
          </p>
          <p class="text-grey">
            {{ $t("misc.thirdPartyNotice") }}
          </p>
        </v-card-text>
        <v-divider></v-divider>
        <v-expand-transition>
          <div v-if="showLicenseDetail" class="console-container">
            <pre class="console-log" style="color: beige">
          GNU GENERAL PUBLIC LICENSE
          Version 3, 29 June 2007

          This program is free software: you can redistribute it and/or modify
          it under the terms of the GNU General Public License as published by
          the Free Software Foundation, either version 3 of the License, or
          (at your option) any later version.

          This program is distributed in the hope that it will be useful,
          but WITHOUT ANY WARRANTY; without even the implied warranty of
          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
          GNU General Public License for more details.

          You should have received a copy of the GNU General Public License
          along with this program. If not, see https://www.gnu.org/licenses/.
        </pre>
          </div>
        </v-expand-transition>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn variant="text" href="https://www.gnu.org/licenses/gpl-3.0.html" target="_blank"
            prepend-icon="mdi-open-in-new">
            {{ $t("misc.licenseLinkText") }}
          </v-btn>
          <v-btn color="primary" variant="text" @click="showLicenseDetail = !showLicenseDetail"
            :prepend-icon="showLicenseDetail ? 'mdi-eye-off' : 'mdi-eye'">
            {{
              showLicenseDetail
                ? $t("misc.hideDetails")
                : $t("misc.showDetails")
            }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-col>
  </v-row>

  <v-snackbar v-model="snackbarVisible" :timeout="3000" color="primary" location="bottom end">
    {{ snackbarMessage }}
    <template v-slot:actions>
      <v-btn variant="text" @click="snackbarVisible = false">
        {{ $t("common.close") }}
      </v-btn>
    </template>
  </v-snackbar>
</template>

<script setup>
import { ref, reactive, onMounted, watch, nextTick } from "vue";
import { useI18n } from "vue-i18n";
import { useConfigStore } from "@/stores/configStore";
import { useLogStore } from "@/stores/logStore";

const { t } = useI18n();

const configStore = useConfigStore();

const logStore = useLogStore();

const localConfig = reactive({
  port: 6185,
  log_dir: "",
  temp_dir: "",
  dataset_dir: "",
});

const logLevels = ["debug", "info", "warning", "error"];

const snackbarVisible = ref(false);

const snackbarMessage = ref("");

const consoleLogRef = ref(null);

const showLicenseDetail = ref(false);

function getLevelDisplayName(level) {
  return t(`misc.logLevels.${level}`, level);
}

function showMessage(msg, color = "primary") {
  snackbarMessage.value = msg;
  snackbarVisible.value = true;
}

function getLogClass(line) {
  if (line.indexOf("[DEBUG]") !== -1) return "log-debug";
  if (line.indexOf("[INFO]") !== -1) return "log-info";
  if (line.indexOf("[WARNING]") !== -1) return "log-warning";
  if (line.indexOf("[ERROR]") !== -1) return "log-error";
  return "";
}

async function refreshConfig() {
  try {
    await configStore.fetchConfig();
    Object.assign(localConfig, configStore.config);
    showMessage(t("misc.configRefreshSuccess"), "primary");
  } catch (err) {
    console.error("Failed to load config", err);
    showMessage(t("misc.configLoadFailed"), "error");
  }
}

async function saveConfig() {
  try {
    await configStore.saveConfig(localConfig);
    showMessage(t("misc.configSaveSuccess"), "primary");
  } catch (err) {
    console.error("Failed to save config", err);
    showMessage(t("misc.configSaveFailed"), "error");
  }
}

async function refreshLogs() {
  try {
    await logStore.fetchLogs();
    await nextTick();
    if (consoleLogRef.value) {
      consoleLogRef.value.scrollTop = consoleLogRef.value.scrollHeight;
    }
  } catch (err) {
    console.error("Failed to load logs", err);
    showMessage(t("misc.logsLoadFailed"), "error");
  }
}

function changeLogLevel(level) {
  logStore.setLevel(level);
}

function changeLogCount(count) {
  logStore.setCount(count);
}

watch(
  () => logStore.logs.length,
  async () => {
    await nextTick();
    if (consoleLogRef.value) {
      consoleLogRef.value.scrollTop = consoleLogRef.value.scrollHeight;
    }
  },
);

onMounted(async () => {
  try {
    await configStore.fetchConfig();
    Object.assign(localConfig, configStore.config);
  } catch (err) {
    console.error("Failed to load config", err);
    showMessage(t("misc.configLoadFailed"), "error");
  }

  try {
    await logStore.fetchLogs();
    await nextTick();
    if (consoleLogRef.value) {
      consoleLogRef.value.scrollTop = consoleLogRef.value.scrollHeight;
    }
  } catch (err) {
    console.error("Failed to load logs", err);
    showMessage(t("misc.logsLoadFailed"), "error");
  }
});
</script>

<style scoped>
.console-container {
  background-color: #0c0c0c !important;
  padding: 0;
  overflow: hidden;
}

.console-log {
  height: 400px;
  overflow-y: auto;
  background-color: #0c0c0c;
  font-family: "Fira Code", "Courier New", monospace;
  font-size: 0.8rem;
  padding: 12px;
}

.port-line :deep(input[type="number"]::-webkit-inner-spin-button),
.port-line :deep(input[type="number"]::-webkit-outer-spin-button) {
  -webkit-appearance: none;
  margin: 0;
}

.log-line {
  white-space: pre-wrap;
  word-break: break-all;
  line-height: 1.4;
  margin-bottom: 2px;
}

.log-debug {
  color: #9cd9f0;
}

.log-info {
  color: #c3e88d;
}

.log-warning {
  color: #ffcb6b;
}

.log-error {
  color: #f07178;
}
</style>