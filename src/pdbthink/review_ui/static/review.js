/* Curator review interface. Plain ES modules-free JS so the server needs no build step. */

const state = { rows: [], filtered: [], current: null, detail: null, viewerReady: null, renderIndex: 0 };

const el = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function boot() {
  state.rows = await (await fetch("/api/instances")).json();
  fillFilter("filter-family", [...new Set(state.rows.map((r) => r.family))].sort());
  fillFilter("filter-protein", [...new Set(state.rows.map((r) => r.protein))].sort());
  ["filter-family", "filter-protein", "filter-source", "filter-status"].forEach((id) =>
    el(id).addEventListener("change", applyFilters)
  );
  document.addEventListener("keydown", onKey);
  applyFilters();
}

function fillFilter(id, values) {
  const select = el(id);
  values.forEach((v) => {
    const option = document.createElement("option");
    option.value = option.textContent = v;
    select.appendChild(option);
  });
}

function applyFilters() {
  const f = el("filter-family").value, p = el("filter-protein").value;
  const s = el("filter-source").value, st = el("filter-status").value;
  state.filtered = state.rows.filter(
    (r) => (!f || r.family === f) && (!p || r.protein === p) && (!s || r.source_type === s) && (!st || r.status === st)
  );
  renderList();
  const decided = state.rows.filter((r) => r.status !== "pending").length;
  el("progress").textContent = `${decided} / ${state.rows.length} decided · ${state.filtered.length} shown`;
}

function renderList() {
  const nav = el("list");
  nav.innerHTML = "";
  state.filtered.forEach((row) => {
    const div = document.createElement("div");
    div.className = "row" + (state.current === row.id ? " active" : "");
    div.innerHTML =
      `<span class="fam">${esc(row.family)}</span>` +
      `<span class="protein">${esc(row.protein)}</span>` +
      `<span class="pill ${esc(row.status)}">${esc(row.status)}</span>`;
    div.onclick = () => open(row.id);
    nav.appendChild(div);
  });
}

async function open(id) {
  state.current = id;
  state.renderIndex = 0;
  renderList();
  state.detail = await (await fetch(`/api/instance/${encodeURIComponent(id)}`)).json();
  renderDetail();
}

function renderDetail() {
  const d = state.detail;
  if (!d) return;
  const inst = d.instance;
  const renders = d.renders;
  const r = renders[state.renderIndex] || renders[0];

  el("detail").innerHTML = `
    <h2>${esc(inst.question_family)} · ${esc(inst.protein_group_id)}</h2>
    <div style="color:var(--muted);font-size:12px">${esc(inst.semantic_instance_id)} ·
      generator v${esc(inst.question_version)} · definitions ${esc(inst.definition_version)}</div>

    <div class="tabs" id="tabs"></div>

    <div class="grid2">
      <div>
        <h3>Structure</h3>
        <div id="viewer"></div>
        <div class="legend">
          <span><span class="swatch" style="background:#e8590c"></span>queried</span>
          <span><span class="swatch" style="background:#2f9e44"></span>gold answer</span>
          <span><span class="swatch" style="background:#1971c2"></span>evidence</span>
        </div>

        <h3>Gold answer</h3>
        <pre>${esc(JSON.stringify(inst.gold_answer, null, 2))}</pre>

        <h3>Selection margins</h3>
        <pre>${esc(JSON.stringify(inst.selection_margins, null, 2))}</pre>

        <h3>Hidden geometric evidence</h3>
        <pre>${esc(JSON.stringify(inst.gold_evidence, null, 2))}</pre>
      </div>

      <div>
        <h3>Model-visible prompt (${esc(r ? r.representation : "-")})</h3>
        <dl class="kv">
          <dt>tokens</dt><dd>${esc(r?.input_token_count)} (${esc(r?.tokenizer)})</dd>
          <dt>atoms</dt><dd>${esc(r?.atom_count)}</dd>
          <dt>rotation seed</dt><dd>${esc(r?.rotation_seed)}${r?.is_rotation_variant ? " (rotation variant)" : ""}</dd>
          <dt>crop</dt><dd>${r?.crop ? esc(JSON.stringify(r.crop)) : "none"}</dd>
          <dt>coords sha256</dt><dd>${esc((r?.displayed_coordinates_sha256 || "").slice(0, 16))}…</dd>
        </dl>
        <pre class="tall">${esc(r ? r.user_prompt : "")}</pre>

        <div class="private">
          <h3>Curator-only provenance</h3>
          <dl class="kv">
            <dt>source</dt><dd>${esc(inst.source_type)} ${esc((inst.source_entries || []).join(", "))}</dd>
            <dt>release dates</dt><dd>${esc((inst.release_dates || []).join(", ") || "-")}</dd>
            <dt>method</dt><dd>${esc(inst.experimental_method || "-")} ${inst.resolution ? esc(inst.resolution) + " Å" : ""}</dd>
            <dt>assembly</dt><dd>${esc((inst.biological_assembly_ids || []).join(", ") || "asymmetric unit")}</dd>
            <dt>chains</dt><dd>${esc((inst.selected_chains || []).join(", "))}</dd>
            <dt>publications</dt><dd>${esc((inst.source_publications || []).join(" · ") || "-")}</dd>
            <dt>file sha256</dt><dd>${esc((inst.source_file_sha256s || []).map((h) => h.slice(0, 12)).join(", "))}</dd>
          </dl>
        </div>

        <h3>Generator acceptance reasons</h3>
        <ul class="reasons">${(inst.acceptance_reasons || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
        <ul class="reasons">${(inst.criteria_passed || []).map((x) => `<li>criterion passed: ${esc(x)}</li>`).join("")}</ul>
        ${warnings(inst)}
      </div>
    </div>

    <div class="actions">
      <button class="accept" id="btn-accept">Accept (a)</button>
      <button class="reject" id="btn-reject">Reject (r)</button>
      <textarea id="notes" placeholder="notes / reason (required to reject or to override a label)">${esc(d.decision?.notes || "")}</textarea>
      <span id="save-state" style="color:var(--muted)">${d.decision ? "decision: " + esc(d.decision.decision) : ""}</span>
    </div>
  `;

  const tabs = el("tabs");
  renders.forEach((rr, i) => {
    const b = document.createElement("button");
    b.textContent = `${rr.representation}${rr.is_rotation_variant ? " (rot)" : ""}${rr.state_order_seed ? " (reversed)" : ""}`;
    b.className = i === state.renderIndex ? "active" : "";
    b.onclick = () => { state.renderIndex = i; renderDetail(); };
    tabs.appendChild(b);
  });

  el("btn-accept").onclick = () => decide("accept");
  el("btn-reject").onclick = () => decide("reject");
  mountViewer(r, d.highlights);
}

function warnings(inst) {
  const w = (inst.gold_evidence || {}).warnings || (inst.selection_margins || {}).episode_warnings;
  if (!w || !w.length) return "";
  return `<h3 class="warn">Validation warnings</h3><ul class="reasons warn">${w.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;
}

async function decide(kind) {
  const notes = el("notes").value.trim();
  if (kind === "reject" && !notes) {
    el("save-state").textContent = "a rejection needs a recorded reason";
    return;
  }
  const payload = {
    semantic_instance_id: state.current,
    decision: kind,
    reason: notes,
    notes,
    curator: el("curator").value.trim(),
  };
  const res = await (await fetch("/api/decision", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })).json();
  if (res.error) {
    el("save-state").textContent = "error: " + res.error;
    return;
  }
  const row = state.rows.find((x) => x.id === state.current);
  if (row) row.status = kind;
  el("save-state").textContent = "saved";
  applyFilters();
  next();
}

function next() {
  const i = state.filtered.findIndex((r) => r.id === state.current);
  const following = state.filtered[i + 1];
  if (following) open(following.id);
}

function previous() {
  const i = state.filtered.findIndex((r) => r.id === state.current);
  if (i > 0) open(state.filtered[i - 1].id);
}

function onKey(event) {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  if (event.key === "a") decide("accept");
  else if (event.key === "r") decide("reject");
  else if (event.key === "j" || event.key === "ArrowDown") { event.preventDefault(); next(); }
  else if (event.key === "k" || event.key === "ArrowUp") { event.preventDefault(); previous(); }
}

/* ---- 3D viewer ------------------------------------------------------- */

function loadViewer() {
  if (state.viewerReady) return state.viewerReady;
  state.viewerReady = fetch("/api/viewer-url")
    .then((r) => r.json())
    .then(({ url }) => new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = url;
      script.onload = resolve;
      script.onerror = () => reject(new Error("3Dmol.js could not be loaded"));
      document.head.appendChild(script);
    }));
  return state.viewerReady;
}

const ROLE_COLOUR = { query: "#e8590c", gold: "#2f9e44", evidence: "#1971c2" };

async function mountViewer(render, highlights) {
  const host = el("viewer");
  if (!render || !render.structures || !render.structures.length) {
    host.innerHTML = '<p class="placeholder" style="padding:10px">No coordinates in this variant.</p>';
    return;
  }
  try {
    await loadViewer();
  } catch (err) {
    host.innerHTML = `<p class="placeholder" style="padding:10px">${esc(err.message)} —
      the viewer needs one-time network access; the prompt text below is unaffected.</p>`;
    return;
  }
  host.innerHTML = "";
  const viewer = $3Dmol.createViewer(host, { backgroundColor: "0x00000000" });
  render.structures.forEach((s, i) => {
    viewer.addModel(s.pdb, "pdb");
    viewer.setStyle({ model: i }, { cartoon: { color: i === 0 ? "spectrum" : "grey" }, stick: { radius: 0.06 } });
  });
  highlights.forEach((h) => {
    const sel = { chain: h.chain, resi: h.resi };
    viewer.setStyle(sel, { stick: { colorscheme: undefined, color: ROLE_COLOUR[h.role], radius: 0.22 } });
    viewer.addLabel(h.label, {
      position: sel, backgroundOpacity: 0.65, fontSize: 10,
      backgroundColor: ROLE_COLOUR[h.role], alignment: "center",
    }, sel);
  });
  viewer.zoomTo();
  viewer.render();
}

boot();
