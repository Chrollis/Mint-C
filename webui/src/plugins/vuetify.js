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

import "vuetify/styles";
import { createVuetify } from "vuetify";
import * as components from "vuetify/components";
import * as directives from "vuetify/directives";
import "@mdi/font/css/materialdesignicons.css";
import { zhHans, en } from "vuetify/locale";

export default createVuetify({
  components,
  directives,
  locale: {
    locale: "zhHans",
    fallbackLocale: "zhHans",
    messages: { zhHans, en },
  },
  theme: {
    defaultTheme: "green",
    customProperties: true,
    themes: {
      green: {
        colors: {
          primary: "#2E7D32",
          bg0: "#E2FFE0",
          bg1: "#D5FFF3",
          bg2: "#E7EEFC",
          bg3: "#EDFEDC",
          bg4: "#FDFDF0",
        },
      },
      blue: {
        colors: {
          primary: "#364ab8",
          bg0: "#E0E7FF",
          bg1: "#E9D5FF",
          bg2: "#FCE7F3",
          bg3: "#FEF0DC",
          bg4: "#F0FEFA",
        },
      },
    },
  },
});
