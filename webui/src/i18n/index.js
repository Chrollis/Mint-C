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

import { createI18n } from "vue-i18n";
import { useLocale } from "vuetify";

import zhCN from "./locales/zh-CN.js";
import enUS from "./locales/en-US.js";

export const supportedLocales = [
  { code: "zh-CN", name: "简体中文（中国大陆）", vuetifyCode: "zhHans" },
  { code: "en-US", name: "English (the United States)", vuetifyCode: "en" },
];

function getInitialLocale() {
  const saved = localStorage.getItem("mint-language");
  if (saved && supportedLocales.some((loc) => loc.code === saved)) {
    return saved;
  }
  return "zh-CN";
}

const i18n = createI18n({
  legacy: false,
  locale: getInitialLocale(),
  fallbackLocale: "zh-CN",
  messages: {
    "zh-CN": zhCN,
    "en-US": enUS,
  },
});

export function switchLocale(newLocale, vuetifyLocaleRef) {
  i18n.global.locale.value = newLocale;
  const target = supportedLocales.find((loc) => loc.code === newLocale);
  if (!vuetifyLocaleRef) console.log("VuetifyLocaleRef is null.");
  if (target && vuetifyLocaleRef) {
    vuetifyLocaleRef.value = target.vuetifyCode;
  }
  localStorage.setItem("mint-language", newLocale);
}

export default i18n;
