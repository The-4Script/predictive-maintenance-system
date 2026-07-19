// Predictive Maintenance dashboard -- talks to the FastAPI backend.
// No frameworks, no CDN. Change API_BASE if your backend runs elsewhere.

var API_BASE = "http://127.0.0.1:8000";

var machines = [];        // filled from GET /api/machines
var selected = null;      // currently selected machine
var showAllRows = false;  // "Show all" toggle for the table
var TOP_N = 8;            // rows shown before "Show all"

// ---- helpers ----
function api(path, opts) {
  return fetch(API_BASE + path, opts).then(function (res) {
    if (!res.ok) return res.json().then(function (j) { throw new Error(j.detail || res.status); });
    return res.json();
  });
}
function riskRank(label) { return label === "High" ? 0 : 1; }
function badge(label) { return '<span class="badge ' + label + '">' + label + "</span>"; }
function setMsg(id, text, ok) {
  var el = document.getElementById(id);
  el.textContent = text; el.className = "msg " + (ok ? "ok" : "bad");
}

// ---- backend status ----
function checkStatus() {
  var el = document.getElementById("apiStatus");
  api("/health").then(function (h) {
    el.innerHTML = 'Backend: <span class="ok">online</span> · model ' +
      (h.model_ready ? "✓" : "✗") + " · GenAI: " + (h.genai_enabled ? "live" : "fallback template");
  }).catch(function () {
    el.innerHTML = 'Backend: <span class="bad">offline</span> — run: python -m uvicorn backend.main:app --reload';
  });
}

// ---- summary (two tiers) ----
function renderSummary() {
  var high = 0, low = 0;
  machines.forEach(function (m) { m.risk_label === "High" ? high++ : low++; });
  document.getElementById("totalMachines").innerText = machines.length;
  document.getElementById("highRisk").innerText = high;
  document.getElementById("lowRisk").innerText = low;
}

// ---- machines table (ranked, with Show More) ----
function renderTable() {
  var tbody = document.querySelector("#machineTable tbody");
  tbody.innerHTML = "";

  var sorted = machines.slice().sort(function (a, b) {
    var r = riskRank(a.risk_label) - riskRank(b.risk_label);
    return r !== 0 ? r : b.confidence - a.confidence;
  });

  var rows = showAllRows ? sorted : sorted.slice(0, TOP_N);
  rows.forEach(function (m) {
    var tr = document.createElement("tr");
    tr.innerHTML =
      "<td>" + m.machine_id + "</td>" +
      "<td>" + m.machine_type + "</td>" +
      "<td>" + m.temperature + "</td>" +
      "<td>" + m.vibration + "</td>" +
      "<td>" + m.current + "</td>" +
      "<td>" + m.load + "</td>" +
      "<td>" + badge(m.risk_label) + "</td>" +
      "<td>" + (m.confidence * 100).toFixed(0) + "%</td>" +
      "<td>" + (m.risk_label === "High" ? "Needs Inspection" : "Normal") + "</td>";
    tr.onclick = function () { showDetails(m); };
    tbody.appendChild(tr);
  });

  // Show/hide the "Show all" button
  var moreBtn = document.getElementById("tableMoreBtn");
  if (sorted.length > TOP_N) {
    moreBtn.classList.remove("hidden");
    moreBtn.textContent = showAllRows
      ? "Show less ▴"
      : "Show all " + sorted.length + " ▾";
  } else {
    moreBtn.classList.add("hidden");
  }
}

document.getElementById("tableMoreBtn").onclick = function () {
  showAllRows = !showAllRows;
  renderTable();
};

// ---- details + AI recommendation ----
function showDetails(m) {
  selected = m;
  document.getElementById("machineDetails").innerHTML =
    "<p><span class='k'>Machine:</span> " + m.machine_id + " (" + m.machine_type + ")</p>" +
    "<p><span class='k'>Temperature:</span> " + m.temperature + " °C</p>" +
    "<p><span class='k'>Vibration:</span> " + m.vibration + " mm/s</p>" +
    "<p><span class='k'>Current:</span> " + m.current + " A</p>" +
    "<p><span class='k'>Load:</span> " + m.load + " %</p>" +
    "<p><span class='k'>Operating hours:</span> " + m.operating_hours + "</p>" +
    "<p><span class='k'>Maintenance count:</span> " + m.maintenance_count + "</p>" +
    (m.impact ?
      "<p><span class='k'>Downtime avoided by acting now:</span> " + m.impact.downtime_hours_avoided + " h</p>" +
      "<p><span class='k'>Estimated cost saved:</span> ₹" + Number(m.impact.cost_saved).toLocaleString("en-IN") + "</p>" : "");

  var ai = document.getElementById("aiExplanation");
  ai.innerHTML = "<p class='hint'>Generating AI explanation…</p>";
  document.getElementById("approveBtn").disabled = true;

  api("/explain", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      machine_id: m.machine_id, risk_label: m.risk_label,
      confidence: m.confidence, top_features: m.top_features || [],
      // sensor values (already loaded) enable sensor-anomaly detection
      temperature: m.temperature, vibration: m.vibration, current: m.current,
      load: m.load, operating_hours: m.operating_hours,
      maintenance_count: m.maintenance_count,
      ambient_temperature: m.ambient_temperature, humidity: m.humidity,
    }),
  }).then(function (ex) {
    ai.innerHTML =
      "<p>" + badge(m.risk_label) + " &nbsp; Confidence: " + (m.confidence * 100).toFixed(1) + "%</p>" +
      "<p class='hint'>Top factors: " + (m.top_features || []).map(function (f) { return f.feature; }).join(", ") + "</p>" +
      "<div class='explanation'>" + ex.explanation + "</div>" +
      "<p class='source'>Source: " + ex.source + "</p>" +
      "<p class='ai-disclaimer'>" + ex.disclaimer + "</p>";
    document.getElementById("approveBtn").disabled = m.risk_label !== "High";
  }).catch(function (e) {
    ai.innerHTML = "<p class='msg bad'>Could not get explanation: " + e.message + "</p>";
  });
}

// ---- business impact ----
function loadImpact() {
  api("/api/impact").then(function (imp) {
    var c = imp.currency || "₹";
    document.getElementById("impFlagged").innerText = imp.machines_flagged_for_inspection;
    document.getElementById("impDowntime").innerText = imp.unplanned_downtime_hours_avoided + " h";
    document.getElementById("impCost").innerText = c + Number(imp.estimated_cost_saved).toLocaleString("en-IN");
    var a = imp.assumptions;
    document.getElementById("impAssumptions").innerText =
      "Unplanned failure ≈ " + a.unplanned_downtime_hours + "h down + " + c + a.unplanned_repair_cost +
      " repair; planned ≈ " + a.planned_downtime_hours + "h + " + c + a.planned_maintenance_cost +
      "; downtime " + c + a.downtime_cost_per_hour + "/h.";
  }).catch(function () { /* best-effort */ });
}

document.getElementById("impMoreBtn").onclick = function () {
  var el = document.getElementById("impAssumptions");
  el.classList.toggle("hidden");
  this.textContent = el.classList.contains("hidden") ? "Show assumptions ▾" : "Hide assumptions ▴";
};

// ---- load machines ----
function loadMachines() {
  api("/api/machines").then(function (data) {
    machines = data;
    renderSummary();
    renderTable();
    loadImpact();
  }).catch(function (e) { setMsg("uploadMsg", "Could not load machines: " + e.message, false); });
}

// ---- upload CSV ----
document.getElementById("uploadBtn").onclick = function () {
  var file = document.getElementById("csvUpload").files[0];
  if (!file) { setMsg("uploadMsg", "Please choose a CSV file first.", false); return; }
  var fd = new FormData(); fd.append("file", file);
  setMsg("uploadMsg", "Uploading…", true);
  api("/upload", { method: "POST", body: fd }).then(function (r) {
    setMsg("uploadMsg", "Stored " + r.rows_stored + " rows. Click 'Run Predictions'.", true);
    loadMachines();
  }).catch(function (e) { setMsg("uploadMsg", e.message, false); });
};

// ---- run predictions (persists to history) ----
document.getElementById("predictBtn").onclick = function () {
  setMsg("uploadMsg", "Running predictions…", true);
  api("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })
    .then(function (rows) {
      setMsg("uploadMsg", "Predicted " + rows.length + " machines (saved to history).", true);
      loadMachines();
    }).catch(function (e) { setMsg("uploadMsg", e.message, false); });
};

// ---- approve work order (human-in-the-loop) ----
document.getElementById("approveBtn").onclick = function () {
  if (!selected) return;
  setMsg("workOrderMsg", "✅ Work order approved for " + selected.machine_id + " by authorized engineer.", true);
};

// ---- init ----
checkStatus();
loadMachines();
