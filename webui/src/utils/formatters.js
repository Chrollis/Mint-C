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

export function toPercent(value) {
  if (value === undefined || value === null) return 0;
  return parseFloat((value * 100).toFixed(2));
}

export function toFixed(value, decimals = 4) {
  if (value === undefined || value === null) return 0;
  return parseFloat(value.toFixed(decimals));
}
