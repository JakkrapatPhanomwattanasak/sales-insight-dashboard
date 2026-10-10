let report;
const number = new Intl.NumberFormat("en", { maximumFractionDigits: 2 });
const message = document.querySelector("#message");
async function call(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error?.message || "Request failed");
  return payload;
}
function show(data) {
  report = data;
  document.querySelector("#results").hidden = false;
  document.querySelector("#rows").textContent = data.rows;
  document.querySelector("#total").textContent = number.format(data.total_revenue);
  document.querySelector("#average").textContent = number.format(data.average_revenue);
  const regions = document.querySelector("#regions");
  regions.replaceChildren();
  const maximum = Math.max(...data.regions.map(item => item.sum), 1);
  for (const item of data.regions) {
    const row = document.createElement("div"); row.className = "bar-row";
    const label = document.createElement("div"); label.className = "bar-label";
    const name = document.createElement("span"); name.textContent = item.region;
    const value = document.createElement("span"); value.textContent = number.format(item.sum);
    label.append(name, value);
    const track = document.createElement("div"); track.className = "bar-track";
    const bar = document.createElement("div"); bar.className = "bar";
    bar.style.width = `${item.sum / maximum * 100}%`; track.append(bar); row.append(label, track); regions.append(row);
  }
  const metrics = document.querySelector("#metrics"); metrics.replaceChildren();
  for (const [name, value] of [["Model MAE", data.metrics.mae], ["Baseline MAE", data.metrics.baseline_mae], ["RMSE", data.metrics.rmse], ["R²", data.metrics.r2]]) {
    const term = document.createElement("dt"); term.textContent = name;
    const detail = document.createElement("dd"); detail.textContent = number.format(value); metrics.append(term, detail);
  }
  for (const field of ["region", "channel"]) {
    const select = document.querySelector(`#${field}`); select.replaceChildren();
    for (const value of data.categories[field]) select.add(new Option(value, value));
  }
  document.querySelector("#estimate").textContent = "";
  message.textContent = `Analyzed ${data.rows} records from ${data.source}; removed ${data.duplicates_removed} duplicates.`;
}
async function analyze(url, options) {
  const buttons = document.querySelectorAll("button"); buttons.forEach(button => { button.disabled = true; });
  message.className = ""; message.textContent = "Analyzing dataset…";
  try { show(await call(url, options)); }
  catch (error) { message.textContent = error.message; message.className = "error"; }
  finally { buttons.forEach(button => { button.disabled = false; }); }
}
document.querySelector("#demo").addEventListener("click", () => analyze("/api/demo", { method: "POST" }));
document.querySelector("#upload").addEventListener("submit", event => {
  event.preventDefault(); const body = new FormData(); body.append("file", document.querySelector("#csv").files[0]);
  analyze("/api/analyze", { method: "POST", body });
});
document.querySelector("#predict").addEventListener("submit", async event => {
  event.preventDefault(); const payload = Object.fromEntries(new FormData(event.target));
  for (const field of ["marketing_spend", "visits"]) payload[field] = Number(payload[field]);
  const output = document.querySelector("#estimate");
  try {
    const result = await call("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    output.textContent = `Estimated revenue: ${number.format(result.predicted_revenue)}`;
  } catch (error) { output.textContent = error.message; }
});
document.querySelector("#download").addEventListener("click", () => {
  if (!report) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
  const link = document.createElement("a"); link.href = url; link.download = "sales-analysis.json"; link.click(); URL.revokeObjectURL(url);
});
