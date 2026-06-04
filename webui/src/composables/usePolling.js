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

import { ref, onUnmounted } from "vue";

export function usePolling(fetchFn, options = {}) {
  const { interval = 2000, immediate = false } = options;

  const intervalId = ref(null);

  const isPolling = ref(false);

  const start = (customInterval) => {
    if (intervalId.value) clearInterval(intervalId.value);
    const ms = customInterval ?? interval;
    isPolling.value = true;
    intervalId.value = setInterval(async () => {
      if (isPolling.value) {
        await fetchFn();
      }
    }, ms);
  };

  const cancel = () => {
    isPolling.value = false;
    if (intervalId.value) {
      clearInterval(intervalId.value);
      intervalId.value = null;
    }
  };

  onUnmounted(() => {
    cancel();
  });

  if (immediate) {
    start();
  }

  return { start, cancel, isPolling };
}
