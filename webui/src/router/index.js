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

import { createRouter, createWebHashHistory } from "vue-router";
import Dashboard from "../views/Dashboard.vue";
import Datasets from "../views/Datasets.vue";
import Model from "../views/Model.vue";
import Train from "../views/Train.vue";
import Recognize from "../views/Recognize.vue";
import Misc from "../views/Misc.vue";
import Evaluate from "../views/Evaluate.vue";

const routes = [
  { path: "/", name: "Dashboard", component: Dashboard },
  { path: "/datasets", name: "Datasets", component: Datasets },
  { path: "/model", name: "Model", component: Model },
  { path: "/train", name: "Train", component: Train },
  { path: "/recognize", name: "Recognize", component: Recognize },
  { path: "/misc", name: "Misc", component: Misc },
  { path: "/evaluate", name: "Evaluate", component: Evaluate },
];

const router = createRouter({
  history: createWebHashHistory(),
  routes,
});

export default router;
