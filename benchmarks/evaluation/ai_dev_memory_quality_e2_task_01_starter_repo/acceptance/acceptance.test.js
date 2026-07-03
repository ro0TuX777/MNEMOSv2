const assert = require("node:assert/strict");
const test = require("node:test");

const logic = require("../src/logic.js");

test("migrates legacy state to the current v2 schema without stale sync fields", () => {
  const state = logic.migrateState({
    schemaVersion: 1,
    items: [
      { id: "a", title: "Waiting item", status: "waiting", updatedAt: "2026-06-01T00:00:00.000Z" },
      { id: "b", title: "Accepted item", status: "accepted", severity: 4, impact: 2, updatedAt: "2026-06-02T00:00:00.000Z" },
      { id: "c", title: "Deferred item", status: "deferred", severity: 5, impact: 5, blocker: true, updatedAt: "2026-06-03T00:00:00.000Z" }
    ]
  });

  assert.equal(state.schemaVersion, 2);
  assert.equal(state.syncEnabled, undefined);
  assert.equal(state.items[0].status, "in_review");
  assert.equal(state.items[0].severity, 1);
  assert.equal(state.items[0].impact, 1);
  assert.equal(state.items[0].blocker, false);
  assert.equal(state.items[1].status, "approved");
  assert.equal(state.items[2].status, "deferred");
});

test("calculates current risk score with severity, impact, and blocker bonus", () => {
  assert.equal(logic.calculateRiskScore({ severity: 3, impact: 4, blocker: false }), 12);
  assert.equal(logic.calculateRiskScore({ severity: 3, impact: 4, blocker: true }), 22);
  assert.equal(logic.calculateRiskScore({}), 1);
});

test("sorts risk_desc by risk, updatedAt descending, title ascending, then id", () => {
  const sorted = logic.sortReviewItems([
    { id: "z", title: "Zulu", severity: 4, impact: 4, blocker: false, updatedAt: "2026-06-02T00:00:00.000Z" },
    { id: "a", title: "Alpha", severity: 3, impact: 3, blocker: true, updatedAt: "2026-06-01T00:00:00.000Z" },
    { id: "b", title: "Beta", severity: 3, impact: 3, blocker: true, updatedAt: "2026-06-03T00:00:00.000Z" },
    { id: "c", title: "Alpha", severity: 3, impact: 3, blocker: true, updatedAt: "2026-06-03T00:00:00.000Z" }
  ], "risk_desc");

  assert.deepEqual(sorted.map((item) => item.id), ["c", "b", "a", "z"]);
});

test("builds default queue from only open and in_review unless closed items are requested", () => {
  const items = [
    { id: "open", title: "Open", status: "open", severity: 1, impact: 1, updatedAt: "2026-06-01T00:00:00.000Z" },
    { id: "review", title: "Review", status: "in_review", severity: 1, impact: 1, updatedAt: "2026-06-02T00:00:00.000Z" },
    { id: "approved", title: "Approved", status: "approved", severity: 5, impact: 5, updatedAt: "2026-06-03T00:00:00.000Z" },
    { id: "deferred", title: "Deferred", status: "deferred", severity: 5, impact: 5, blocker: true, updatedAt: "2026-06-04T00:00:00.000Z" }
  ];

  assert.deepEqual(logic.buildReviewQueue(items).map((item) => item.id), ["review", "open"]);
  assert.deepEqual(logic.buildReviewQueue(items, { includeClosed: true }).map((item) => item.id), ["deferred", "approved", "review", "open"]);
});

test("applies only current review decisions and rejects stale active statuses", () => {
  const item = { id: "x", title: "Decision", status: "open", decisionNotes: [] };
  const approved = logic.applyReviewDecision(item, "approved", "accepted after review");

  assert.equal(approved.status, "approved");
  assert.equal(approved.decisionNotes.length, 1);
  assert.throws(() => logic.applyReviewDecision(item, "accepted", "old status"), /Unsupported decision status/);
});

test("policy summary identifies current local-only behavior and stale archive boundary", () => {
  const summary = logic.getPolicySummary();

  assert.equal(summary.localOnly, true);
  assert.equal(summary.cloudSyncAllowed, false);
  assert.equal(summary.deferredPromotesToApproved, false);
  assert.match(summary.currentAuthority, /ADR 0007/);
  assert.match(summary.staleGuidanceBoundary, /superseded/i);
});

