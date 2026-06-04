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
  <v-app>
    <BackgroundAnimation :color="primaryColor" />
    <SplashScreen :visible="showSplash" :color="primaryColor" @close="showSplash = false"
      @updatePolygon="setSplashPolygon" />
    <GlobalCursor :color="primaryColor" :show-splash="showSplash" :splash-polygon="splashPolygon" />
    <div v-if="!showSplash">
      <v-app-bar app class="frosted-glass">
        <v-app-bar-nav-icon @click="toggleDrawer"></v-app-bar-nav-icon>
        <v-app-bar-title class="mint-c-title">
          <span>{{ $t("app.title") }}</span>
        </v-app-bar-title>
        <v-spacer></v-spacer>

        <v-btn :icon="currentTheme === 'green' ? 'mdi-weather-sunny' : 'mdi-weather-night'
          " @click="toggleTheme" variant="text"></v-btn>
        <v-menu offset-y>
          <template v-slot:activator="{ props }">
            <v-btn icon="mdi-translate" variant="text" v-bind="props"></v-btn>
          </template>
          <v-list class="frosted-glass">
            <v-list-item v-for="lang in supportedLocales" :key="lang.code" :value="lang.code"
              @click="switchLanguage(lang.code)">
              <v-list-item-title>{{ lang.name }}</v-list-item-title>
            </v-list-item>
          </v-list>
        </v-menu>
        <v-btn icon="mdi-animation-play-outline" variant="text" @click="activateSplash"></v-btn>
        <v-btn icon="mdi-github" href="https://github.com/Chrollis/Mint-C" target="_blank" variant="text"></v-btn>
        <div class="model-status d-none d-sm-flex" @click="refreshModelStatus">
          <v-icon :color="modelStore.ready ? 'primary' : 'error'" size="small">
            {{ modelStore.ready ? "mdi-check-circle" : "mdi-close-circle" }}
          </v-icon>
          <span class="status-text">
            {{
              modelStore.ready
                ? $t("common.modelLoaded")
                : $t("common.modelNotLoaded")
            }}
          </span>
        </div>
      </v-app-bar>

      <v-footer app class="frosted-glass">
        <div class="footer-left">
          <v-icon size="small" icon="mdi-console-line"></v-icon>
          <span class="text-truncatet">{{ latestLogMessage }}</span>
        </div>
        <v-spacer></v-spacer>
        <div class="footer-right" :class="{ 'training-active': isTrainingRunning }">
          <v-icon size="small" :icon="isTrainingRunning ? 'mdi-loading' : 'mdi-check-circle'"
            :class="{ spin: isTrainingRunning }"></v-icon>
          <span class="text-truncatet">{{ trainingStatusText }}</span>
        </div>
      </v-footer>

      <v-navigation-drawer v-model="drawer" app :permanent="mdAndUp" :expand-on-hover="mdAndUp && lgAndDown"
        :rail="mdAndUp && lgAndDown" class="frosted-glass"> <v-list>
          <v-list-item prepend-icon="mdi-view-dashboard" :title="$t('app.drawer.dashboard')" to="/"></v-list-item>
          <v-list-item prepend-icon="mdi-database" :title="$t('app.drawer.datasets')" to="/datasets"></v-list-item>
          <v-list-item prepend-icon="mdi-brain" :title="$t('app.drawer.model')" to="/model"></v-list-item>
          <v-list-item prepend-icon="mdi-chart-line" :title="$t('app.drawer.train')" to="/train"></v-list-item>
          <v-list-item prepend-icon="mdi-chart-bar" :title="$t('app.drawer.evaluate')" to="/evaluate"></v-list-item>
          <v-list-item prepend-icon="mdi-pencil" :title="$t('app.drawer.recognize')" to="/recognize"></v-list-item>
          <v-list-item prepend-icon="mdi-cog" :title="$t('app.drawer.misc')" to="/misc"></v-list-item>
        </v-list>
      </v-navigation-drawer>

      <v-main>
        <v-container fluid>
          <router-view />
        </v-container>
      </v-main>

      <v-snackbar v-model="snackbarStore.visible" :timeout="3000" :color="snackbarStore.color" location="bottom end">
        {{ snackbarStore.message }}
        <template v-slot:actions>
          <v-btn variant="text" @click="snackbarStore.hide">
            {{ t("common.close") }}
          </v-btn>
        </template>
      </v-snackbar>
    </div>
  </v-app>
</template>

<script setup>

import { ref, onMounted, onUnmounted, computed } from "vue";
import { useTheme, useDisplay, useLocale } from "vuetify";
import { useModelStore } from "./stores/modelStore";
import { useTrainingStore } from "./stores/trainingStore";
import { useLogStore } from "./stores/logStore";
import { useI18n } from "vue-i18n";
import i18n, { supportedLocales, switchLocale } from "./i18n";
import { useSnackbarStore } from './stores/snackbarStore'
import SplashScreen from "./components/SplashScreen.vue";
import BackgroundAnimation from "./components/BackgroundAnimation.vue";
import GlobalCursor from "./components/GlobalCursor.vue";

const modelStore = useModelStore();

const trainingStore = useTrainingStore();

const logStore = useLogStore();

const snackbarStore = useSnackbarStore()

const { mdAndUp, lgAndDown } = useDisplay();

const { current: vuetifyLocale } = useLocale();

const { t } = useI18n();

const theme = useTheme();

const drawer = ref(false);

const currentTheme = ref("green");

let latestLogInterval = null;

const showSplash = ref(true);

const splashPolygon = ref([]);

const latestLogMessage = computed(
  () => logStore.latestLog || t("app.footer.ready"),
);

const isTrainingRunning = computed(() => {
  return trainingStore.status.state === "running";
});

const trainingStatusText = computed(() => {
  const s = trainingStore.status;
  if (s.state === "running") {
    return t("app.footer.training", {
      epoch: s.current_epoch,
      total: s.total_epochs,
      loss: s.current_loss.toFixed(4),
    });
  }
  if (s.state === "finished") return t("app.footer.finished");
  if (s.state === "cancelled") return t("app.footer.cancelled");
  if (s.state === "error") return t("app.footer.error");
  return t("app.footer.idle");
});

const primaryColor = computed(() => theme.global.current.value.colors.primary);

function toggleDrawer() {
  drawer.value = !drawer.value;
  localStorage.setItem("mint-drawer", drawer.value ? "true" : "false");
}

function toggleTheme() {
  const newTheme = currentTheme.value === "green" ? "blue" : "green";
  theme.change(newTheme);
  currentTheme.value = newTheme;
  localStorage.setItem("mint-theme", newTheme);
}

async function switchLanguage(langCode) {
  switchLocale(langCode, vuetifyLocale);
}

async function refreshModelStatus() {
  await modelStore.checkReady();
}

function setSplashPolygon(polygon) {
  splashPolygon.value = polygon;
}

function activateSplash() {
  window.location.reload();
}

onMounted(async () => {
  const savedTheme = localStorage.getItem("mint-theme");
  if (savedTheme === "green" || savedTheme === "blue") {
    currentTheme.value = savedTheme;
    theme.change(savedTheme);
  }

  const savedDrawer = localStorage.getItem("mint-drawer");
  if (savedDrawer !== null) drawer.value = savedDrawer === "true";

  await modelStore.checkReady();
  await trainingStore.fetchTrainingStatus();
  await logStore.fetchLatestLog();
  await logStore.fetchLogs();

  latestLogInterval = setInterval(() => {
    logStore.fetchLatestLog();
  }, 5000);
});

onUnmounted(() => {
  if (latestLogInterval) clearInterval(latestLogInterval);
});
</script>

<style scoped>
.model-status {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  white-space: nowrap;
  display: flex;
  align-items: center;
  background-color: rgba(var(--v-theme-primary), 0.07);
  gap: 6px;
  padding: 4px 12px;
  border-radius: 12px;
  backdrop-filter: blur(4px);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
}

.model-status:hover {
  backdrop-filter: blur(8px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04) !important;
}

.status-text {
  font-size: 0.9rem;
  font-weight: 500;
}

.footer-left {
  display: flex;
  align-items: center;
  gap: 8px;
  background-color: rgba(0, 0, 0, 0.03);
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.8rem;
}

.footer-right {
  display: flex;
  align-items: center;
  gap: 8px;
  background-color: rgba(0, 0, 0, 0.03);
  padding: 4px 12px;
  border-radius: 20px;
  transition: all 0.3s;
  font-size: 0.8rem;
}

.footer-right.training-active {
  background-color: rgba(var(--v-theme-primary), 0.15);
  color: rgb(var(--v-theme-primary));
  animation: pulse 1.5s ease-in-out infinite;
}

.mint-c-title {
  display: flex;
  align-items: center;
  gap: 8px;
}
</style>