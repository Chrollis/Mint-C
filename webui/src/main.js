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

import { createApp } from "vue";
import App from "./App.vue";
import vuetify from "./plugins/vuetify";
import "./styles/global.scss";
import router from "./router";
import { createPinia } from "pinia";
import i18n from "./i18n";
import VueApexCharts from "vue3-apexcharts";

const app = createApp(App);
const pinia = createPinia();

app.use(i18n);
app.use(vuetify);
app.use(router);
app.use(pinia);

app.component("VueApexCharts", VueApexCharts);

app.mount("#app");
