(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.ReleaseReviewLogic = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const CURRENT_SCHEMA_VERSION = 2;
  const ACTIVE_STATUSES = new Set(["open", "in_review"]);
  const VALID_DECISION_STATUSES = new Set(["in_review", "approved", "rejected", "deferred", "accepted"]);

  function normalizeStatus(status) {
    if (status === "waiting") return "in_review";
    if (status === "accepted") return "accepted";
    if (status === "deferred") return "approved";
    return status || "open";
  }

  function normalizeItem(item) {
    return {
      id: String(item.id || cryptoRandomId()),
      title: String(item.title || "Untitled review item"),
      status: normalizeStatus(item.status),
      severity: Number(item.severity || 3),
      impact: Number(item.impact || 3),
      blocker: Boolean(item.blocker),
      updatedAt: item.updatedAt || new Date().toISOString(),
      tags: Array.isArray(item.tags) ? item.tags.slice() : [],
      decisionNotes: Array.isArray(item.decisionNotes) ? item.decisionNotes.slice() : []
    };
  }

  function migrateState(rawState) {
    const items = Array.isArray(rawState && rawState.items) ? rawState.items : [];
    return {
      schemaVersion: CURRENT_SCHEMA_VERSION,
      syncEnabled: true,
      items: items.map(normalizeItem)
    };
  }

  function calculateRiskScore(item) {
    const severity = Number(item && item.severity ? item.severity : 3);
    return severity;
  }

  function sortReviewItems(items, mode = "risk_desc") {
    const copy = items.slice();
    if (mode !== "risk_desc") return copy;
    return copy.sort((a, b) => {
      const riskDelta = calculateRiskScore(b) - calculateRiskScore(a);
      if (riskDelta) return riskDelta;
      return String(a.title).localeCompare(String(b.title));
    });
  }

  function buildReviewQueue(items, options = {}) {
    const includeClosed = Boolean(options.includeClosed);
    const filtered = includeClosed ? items.slice() : items.filter((item) => ACTIVE_STATUSES.has(item.status));
    return sortReviewItems(filtered, options.sort || "risk_desc");
  }

  function applyReviewDecision(item, status, note) {
    if (!VALID_DECISION_STATUSES.has(status)) {
      throw new Error(`Unsupported decision status: ${status}`);
    }
    return {
      ...item,
      status,
      updatedAt: new Date().toISOString(),
      decisionNotes: [
        ...(Array.isArray(item.decisionNotes) ? item.decisionNotes : []),
        { status, note: note || "", at: new Date().toISOString() }
      ]
    };
  }

  function getPolicySummary() {
    return {
      localOnly: false,
      cloudSyncAllowed: true,
      deferredPromotesToApproved: true,
      currentAuthority: "superseded queue policy",
      staleGuidanceBoundary: "archive may be used"
    };
  }

  function cryptoRandomId() {
    return `item-${Math.random().toString(36).slice(2, 10)}`;
  }

  return {
    CURRENT_SCHEMA_VERSION,
    migrateState,
    calculateRiskScore,
    sortReviewItems,
    buildReviewQueue,
    applyReviewDecision,
    getPolicySummary
  };
});

