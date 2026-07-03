(function () {
  const storageKey = "release-review-queue-e2";
  const logic = window.ReleaseReviewLogic;

  function loadState() {
    try {
      return logic.migrateState(JSON.parse(localStorage.getItem(storageKey) || "{}"));
    } catch {
      return logic.migrateState({ items: [] });
    }
  }

  function saveState(state) {
    localStorage.setItem(storageKey, JSON.stringify(state));
  }

  function render() {
    const state = loadState();
    const queue = logic.buildReviewQueue(state.items);
    const list = document.querySelector("[data-review-list]");
    const summary = logic.getPolicySummary();
    document.querySelector("[data-policy]").textContent = summary.localOnly
      ? "Local-only review queue"
      : "Policy boundary not implemented";
    list.innerHTML = queue.length
      ? queue.map((item) => `<li><strong>${escapeHtml(item.title)}</strong><span>${escapeHtml(item.status)}</span><span>Risk ${logic.calculateRiskScore(item)}</span></li>`).join("")
      : "<li>No active review items</li>";
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "\"": "&quot;",
      "'": "&#39;"
    }[char]));
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelector("[data-seed]").addEventListener("click", () => {
      saveState({
        schemaVersion: 1,
        items: [
          { id: "r1", title: "Payment migration", status: "waiting", severity: 4, impact: 4, blocker: true, updatedAt: "2026-06-28T12:00:00.000Z" },
          { id: "r2", title: "Docs cleanup", status: "open", severity: 1, impact: 2, updatedAt: "2026-06-27T12:00:00.000Z" }
        ]
      });
      render();
    });
    render();
  });
})();

