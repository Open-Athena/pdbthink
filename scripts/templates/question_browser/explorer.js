"use strict";
const $ = id => document.getElementById(id);
const labels = {correct: "Correct", wrong_answer: "Wrong answer", format_error: "Format error",
  refusal: "Scorer-detected refusal", context_exhausted: "Context exhausted", output_limit: "Native output limit",
  missing: "Missing", collection_failure: "Collection failure"};
const pretty = value => typeof value === "string" ? value : JSON.stringify(value, null, 2);
const element = (tag, text, className) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text;
  if (className) node.className = className; return node; };
const option = (value, text) => { const node = element("option", text); node.value = value; return node; };
const details = (title, value, open = false) => { const node = element("details"); node.open = open;
  node.append(element("summary", title), element("pre", pretty(value))); return node; };
const fetchJSON = async path => { const response = await fetch(path); if (!response.ok) throw new Error(`Could not load ${path} (${response.status})`); return response.json(); };
let index, current, filtered = [], generation = 0;
const panelGeneration = {left: 0, right: 0};
const reportError = error => { $("error").hidden = false; $("error").textContent = error.message; };
function updateURL() {
  const url = new URL(location.href); url.search = "";
  url.searchParams.set("question", $("question").value); url.searchParams.set("model", $("model-left").value);
  url.searchParams.set("compare", $("model-right").value); history.replaceState(null, "", url);
}
async function showPanel(side) {
  const ticket = ++panelGeneration[side], node = $(side), name = $("model-" + side).value;
  node.replaceChildren(); if (!current) return;
  const entry = current.models[name];
  node.append(element("p", labels[entry.status] || entry.status, "status"));
  if (!entry.attempts.length) { node.append(element("p", "No returned output or terminal collection failure in this snapshot.")); return; }
  const selector = element("select"), label = element("label", "Attempt");
  entry.attempts.forEach(attempt => selector.append(option(attempt.path,
    `${attempt.selected ? "Selected" : "Earlier / unscored"} · ${attempt.stage} · ${labels[attempt.status] || attempt.status}`)));
  selector.value = entry.selected || entry.attempts.at(-1).path; label.append(selector); node.append(label);
  const body = element("div"); node.append(body); let attemptGeneration = 0;
  async function loadAttempt() {
    const attemptTicket = ++attemptGeneration; body.replaceChildren(element("p", "Loading full output…"));
    try {
      const record = await fetchJSON(selector.value);
      if (ticket !== panelGeneration[side] || attemptTicket !== attemptGeneration) return;
      body.replaceChildren();
      if (!record.selected) body.append(element("p", "This attempt is not used for this question's headline score.", "notice"));
      if (record.partial) body.append(element("p", "Partial stream: generation did not complete. No final score was assigned.", "notice"));
      const audit = record.audit, usage = audit.usage || {};
      body.append(element("p", `Finish: ${record.choices.map(c => c.finish_reason || "not received").join(", ") || "not received"} · Prompt tokens: ${usage.prompt_tokens ?? "unknown"} · Output tokens: ${usage.completion_tokens ?? "unknown"} · Requested output: ${audit.max_output_tokens ?? "not submitted"}`, "meta"));
      const link = element("a", "Download full output and audit (JSON)"); link.href = selector.value; link.download = "";
      const paragraph = element("p"); paragraph.append(link); body.append(paragraph);
      if (record.scored_answer !== undefined) body.append(details("Answer passed to the scorer", record.scored_answer || "[Empty answer]", true));
      if (record.parsed) body.append(details("Parsed answer and score", {parsed: record.parsed, score: record.score, flags: record.flags}));
      let fields = 0;
      for (const choice of record.choices) {
        for (const [field, value] of Object.entries(choice.message || {})) {
          if (field === "role" || value === null || value === "") continue;
          fields++;
          body.append(details(`Full message.${field}`, value, field === "content"));
        }
      }
      if (!fields) body.append(element("p", "No model text was received for this attempt."));
      body.append(details("Request controls, budgets and hashes", audit));
      if (record.request_controls) body.append(details("Request controls for failed attempt", record.request_controls));
    } catch (error) { if (ticket === panelGeneration[side]) body.replaceChildren(element("p", error.message, "notice")); }
  }
  selector.addEventListener("change", loadAttempt); await loadAttempt();
}
async function showQuestion() {
  const ticket = ++generation, row = filtered.find(q => q.id === $("question").value);
  current = null; ++panelGeneration.left; ++panelGeneration.right;
  $("left").replaceChildren(); $("right").replaceChildren(); $("question-view").hidden = true;
  if (!row) return;
  const position = filtered.indexOf(row); $("previous").disabled = position === 0; $("next").disabled = position === filtered.length - 1;
  try {
    const question = await fetchJSON(row.path); if (ticket !== generation) return; current = question;
    $("question-title").textContent = `${row.family} · ${row.family_name} · ${row.id}`;
    $("question-id").textContent = `Frozen render: ${row.render_id}`;
    $("system-prompt").textContent = question.messages[0].content; $("user-prompt").textContent = question.messages[1].content;
    $("gold").textContent = pretty({answer_schema: question.answer_schema, gold_answer: question.gold_answer});
    $("prompt-download").href = row.path;
    $("model-table").replaceChildren();
    for (const [name, model] of Object.entries(index.models)) {
      const tr = element("tr"), buttons = element("td");
      for (const side of ["left", "right"]) {
        const button = element("button", side === "left" ? "Left" : "Right"); button.type = "button";
        button.addEventListener("click", () => { $("model-" + side).value = name; updateURL(); showPanel(side); }); buttons.append(button, document.createTextNode(" "));
      }
      tr.append(element("th", name), element("td", model.cohort), element("td", labels[question.models[name].status] || question.models[name].status), buttons);
      $("model-table").append(tr);
    }
    $("question-view").hidden = false; updateURL(); await Promise.all([showPanel("left"), showPanel("right")]);
  } catch (error) { reportError(error); }
}
function filterQuestions(preferred) {
  const family = $("family").value, model = $("filter-model").value, status = $("outcome").value;
  filtered = index.questions.filter(q => (!family || q.family === family) && (!status || Object.entries(q.statuses).some(([name, value]) =>
    (!model || name === model) && (status === "incorrect" ? value !== "correct" && value !== "missing" : value === status))));
  $("question").replaceChildren(...filtered.map(q => option(q.id, `${q.family} · ${q.family_name} · ${q.id}`)));
  if (filtered.some(q => q.id === preferred)) $("question").value = preferred;
  $("count").textContent = `${filtered.length} of ${index.questions.length} questions`;
  $("previous").disabled = $("next").disabled = !filtered.length; showQuestion();
}
async function main() {
  index = await fetchJSON("index.json"); $("snapshot").textContent = `Snapshot: ${index.generated_at} · ${Object.keys(index.models).length} model variants · ${index.dataset_fingerprint}`;
  const families = new Map(index.questions.map(q => [q.family, q.family_name]));
  families.forEach((name, family) => $("family").append(option(family, `${family} · ${name}`)));
  for (const name of Object.keys(index.models)) for (const id of ["filter-model", "model-left", "model-right"]) $(id).append(option(name, name));
  const params = new URLSearchParams(location.search);
  $("model-left").value = index.models[params.get("model")] ? params.get("model") : "qwen";
  $("model-right").value = index.models[params.get("compare")] ? params.get("compare") : "mochi";
  for (const id of ["family", "filter-model", "outcome"]) $(id).addEventListener("change", () => filterQuestions($("question").value));
  $("question").addEventListener("change", showQuestion);
  for (const side of ["left", "right"]) $("model-" + side).addEventListener("change", () => { updateURL(); showPanel(side); });
  for (const [id, step] of [["previous", -1], ["next", 1]]) $(id).addEventListener("click", () => {
    $("question").selectedIndex += step; showQuestion();
  });
  filterQuestions(params.get("question"));
}
main().catch(reportError);
