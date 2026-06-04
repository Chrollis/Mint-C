// Mint-C - Web-based digit recognition tool with C++ CNN backend
// Copyright (C) 2026 Chrollis
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

import { ref, watch } from "vue";

export function usePersistedState(key, defaultValue) {
  let initialValue = defaultValue;
  try {
    const stored = localStorage.getItem(key);
    if (stored !== null) {
      initialValue = JSON.parse(stored);
    }
  } catch (e) {
    console.error(`Failed to load persisted state for key "${key}":`, e);
  }

  const data = ref(initialValue);

  watch(
    data,
    (newValue) => {
      try {
        localStorage.setItem(key, JSON.stringify(newValue));
      } catch (e) {
        console.error(`Failed to save persisted state for key "${key}":`, e);
      }
    },
    { deep: true },
  );

  return data;
}
