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
    <div class="numeric-stepper">
        <v-text-field :model-value="internalValue" type="number" :label="label" :disabled="disabled" :min="min"
            :max="max" :step="step" :hide-details="hideDetails" :variant="variant" :density="density" :hint="hint"
            :persistent-hint="persistentHint" class="number-field-with-buttons" prepend-inner-icon="mdi-minus"
            append-inner-icon="mdi-plus" @click:prepend-inner="decrement()" @click:append-inner="increment()"
            @blur="onBlur()" @update:model-value="onInput($event)" />
    </div>
</template>

<script setup>
import { ref, watch } from "vue";

const props = defineProps({
    modelValue: { type: Number, default: 0 },
    label: { type: String, default: "" },
    min: { type: Number, default: 0 },
    max: { type: Number, default: 100 },
    step: { type: Number, default: 1 },
    disabled: { type: Boolean, default: false },
    hideDetails: { type: Boolean, default: true },
    variant: { type: String, default: "outlined" },
    density: { type: String, default: "compact" },
    hint: { type: String, default: "" },
    persistentHint: { type: Boolean, default: false },
});

const emit = defineEmits(["update:modelValue", "blur"]);

const internalValue = ref(props.modelValue);

watch(
    () => props.modelValue,
    (newVal) => {
        internalValue.value = newVal;
    },
);

function clamp(value) {
    let val = value;
    if (isNaN(val)) val = props.min;
    return Math.min(props.max, Math.max(props.min, val));
}

function onInput(val) {
    const num = parseFloat(val);
    const clamped = clamp(isNaN(num) ? props.min : num);
    internalValue.value = clamped;
    emit("update:modelValue", clamped);
}

function decrement() {
    let newVal = internalValue.value - props.step;
    if (newVal < props.min) newVal = props.min;
    internalValue.value = newVal;
    emit("update:modelValue", newVal);
}

function increment() {
    let newVal = internalValue.value + props.step;
    if (newVal > props.max) newVal = props.max;
    internalValue.value = newVal;
    emit("update:modelValue", newVal);
}

function onBlur() {
    const val = internalValue.value;
    const clamped = clamp(isNaN(val) ? props.min : val);
    internalValue.value = clamped;
    emit("update:modelValue", clamped);
    emit("blur");
}
</script>

<style scoped>
.number-field-with-buttons :deep(input) {
    text-align: center;
}

.number-field-with-buttons :deep(.v-icon) {
    transition: all 0.2s ease;
    cursor: pointer;
    border-radius: 8px;
    padding: 4px;
    margin: -4px;
}

.number-field-with-buttons :deep(.v-icon:hover) {
    background-color: rgba(0, 0, 0, 0.08);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    transform: scale(1.05);
}

.number-field-with-buttons :deep(input[type="number"]::-webkit-inner-spin-button),
.number-field-with-buttons :deep(input[type="number"]::-webkit-outer-spin-button) {
    -webkit-appearance: none;
    margin: 0;
}
</style>