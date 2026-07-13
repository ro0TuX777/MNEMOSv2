const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const logic = require("../src/logic.js");

const fixturePath = path.join(__dirname, "fixtures", "legacy_state_v1.json");
const htmlPath = path.join(__dirname, "..", "src", "index.html");

test("existing core behavior still works", () => {
  let state = logic.createInitialState();
  state = logic.createIssue(state, {
    title: "Alpha issue",
    status: "todo",
    priority: "high"
  });
  assert.equal(state.issues.length, 1);

  state = logic.updateIssue(state, state.issues[0].id, { status: "done" });
  assert.equal(state.issues[0].status, "done");

  const visible = logic.getVisibleIssues(state, {
    statusFilters: ["done"],
    priorityFilters: ["high"],
    searchTerm: "alpha",
    sortMode: "updated_desc"
  });
  assert.equal(visible.length, 1);

  state = logic.deleteIssue(state, state.issues[0].id);
  assert.equal(state.issues.length, 0);
});

test("saved views are available and persist a named filter/sort state", () => {
  let state = logic.createInitialState();
  state = logic.saveView(state, {
    name: "Focus",
    statusFilters: ["todo"],
    priorityFilters: ["high"],
    searchTerm: "alpha",
    sortMode: "priority_desc"
  });
  assert.equal(state.savedViews.length, 1);
  assert.equal(state.savedViews[0].name, "Focus");
});

test("priority_desc sorting uses the documented deterministic tie-break contract", () => {
  const state = logic.migratePersistedState({
    issues: [
      {
        id: "b",
        title: "Bravo",
        status: "todo",
        priority: "high",
        createdAt: "2026-06-01T10:00:00Z",
        updatedAt: "2026-06-01T11:00:00Z"
      },
      {
        id: "a",
        title: "Alpha",
        status: "todo",
        priority: "high",
        createdAt: "2026-06-01T10:00:00Z",
        updatedAt: "2026-06-01T11:00:00Z"
      }
    ]
  });

  const visible = logic.getVisibleIssues(state, {
    statusFilters: [],
    priorityFilters: [],
    searchTerm: "",
    sortMode: "priority_desc"
  });

  assert.deepEqual(
    visible.map((issue) => issue.id),
    ["a", "b"]
  );
});

test("migration preserves valid issues, applies documented defaults, and is idempotent", () => {
  const legacy = JSON.parse(fs.readFileSync(fixturePath, "utf-8"));
  const migrated = logic.migratePersistedState(legacy);
  const migratedTwice = logic.migratePersistedState(migrated);

  assert.equal(migrated.issues.length, 2);
  assert.equal(migrated.issues[0].priority, "low");
  assert.equal(migrated.issues[1].priority, "low");
  assert.deepEqual(migratedTwice, migrated);
});

test("known defect is repaired: priority_desc no longer relies on insertion order", () => {
  const state = logic.migratePersistedState({
    issues: [
      {
        id: "c",
        title: "Charlie",
        status: "todo",
        priority: "medium",
        createdAt: "2026-06-01T10:00:00Z",
        updatedAt: "2026-06-02T10:00:00Z"
      },
      {
        id: "a",
        title: "Alpha",
        status: "todo",
        priority: "medium",
        createdAt: "2026-06-01T10:00:00Z",
        updatedAt: "2026-06-02T10:00:00Z"
      }
    ]
  });

  const visible = logic.getVisibleIssues(state, {
    statusFilters: [],
    priorityFilters: [],
    searchTerm: "",
    sortMode: "priority_desc"
  });

  assert.deepEqual(visible.map((issue) => issue.id), ["a", "c"]);
});

test("starter UI exposes the frozen app shell and saved-view area", () => {
  const html = fs.readFileSync(htmlPath, "utf-8");
  assert.match(html, /Local Issue Tracker/);
  assert.match(html, /Saved Views/);
  assert.match(html, /aria-label="Search issues"/);
});
