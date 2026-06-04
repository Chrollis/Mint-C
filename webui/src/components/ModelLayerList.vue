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
  <div>
    <div v-if="!layers || layers.length === 0" class="text-center pa-4 text-grey">
      {{ $t("modelLayerList.noLayers") }}
    </div>
    <v-list class="transparent">
      <v-list-item v-for="(layer, idx) in layers" :key="idx" class="layer-item">
        <template v-slot:prepend>
          <v-icon icon="mdi-drag-vertical" size="small" class="mr-2"></v-icon>
        </template>
        <div class="d-flex justify-space-between align-center" style="width: 100%">
          <div>
            <strong>{{ getLayerDisplayName(layer.type) }}</strong>
            <div class="text-grey mt-1" style="font-size: small">
              <template v-if="Object.keys(layer).length === 1">
                {{ $t("modelLayerList.noParams") }}
              </template>
              <template v-else>
                {{ formatParams(layer) }}
              </template>
            </div>
          </div>
          <v-btn v-if="editable && idx === layers.length - 1" icon="mdi-delete" size="x-small" variant="text"
            color="error" @click="remove(idx)"></v-btn>
        </div>
      </v-list-item>
    </v-list>
    <div v-if="!editable && layers && layers.length > 0" class="pa-2 text-grey text-center">
      {{ $t("modelLayerList.totalLayers", { count: layers.length }) }}
    </div>
  </div>
</template>

<script setup>
import { useI18n } from "vue-i18n";

const props = defineProps({
  layers: Array,
  inputShape: Array,
  numClasses: Number,
  editable: Boolean,
});

const emit = defineEmits(["remove-layer"]);

const { t } = useI18n();

function getLayerDisplayName(type) {
  return t(`layerTypes.${type}`);
}

function getParamName(key) {
  return t(`layerParams.${key}`) || key;
}

function formatParams(layer) {
  const entries = Object.entries(layer).filter(([k]) => k !== "type");
  return entries
    .map(([k, v]) => {
      const key = getParamName(k);
      return `${key}: ${v}`;
    })
    .join(", ");
}

function remove(index) {
  if (index === props.layers.length - 1) {
    emit("remove-layer", index);
  }
}
</script>

<style scoped>
.layer-item {
  padding: 8px 16px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}
</style>