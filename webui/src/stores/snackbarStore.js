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

import { defineStore } from "pinia";
import { ref } from "vue";

export const useSnackbarStore = defineStore("snackbar", () => {
  const visible = ref(false);

  const message = ref("");

  const color = ref("primary");

  let timeoutId = null;

  function show(msg, col = "primary", duration = 3000) {
    if (timeoutId) clearTimeout(timeoutId);
    message.value = msg;
    color.value = col;
    visible.value = true;

    timeoutId = setTimeout(() => {
      visible.value = false;
    }, duration);
  }

  function hide() {
    if (timeoutId) clearTimeout(timeoutId);
    visible.value = false;
  }

  return {
    visible,
    message,
    color,
    show,
    hide,
  };
});
