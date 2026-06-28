(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  root.IssueTrackerLogic = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const STATUS_VALUES = ["todo", "in_progress", "done"];
  const PRIORITY_VALUES = ["low", "medium", "high"];
  const PRIORITY_RANK = { low: 1, medium: 2, high: 3 };

  function nowIso() {
    return new Date().toISOString();
  }

  function createId(prefix) {
    return prefix + "-" + Math.random().toString(16).slice(2, 10);
  }

  function createInitialState() {
    return {
      schemaVersion: 2,
      issues: [],
      savedViews: [],
      uiState: {
        sortMode: "updated_desc"
      }
    };
  }

  function isValidStatus(value) {
    return STATUS_VALUES.includes(value);
  }

  function isValidPriority(value) {
    return PRIORITY_VALUES.includes(value);
  }

  function normalizeIssue(input, fallbackTime) {
    if (!input || typeof input !== "object") {
      return null;
    }
    if (typeof input.title !== "string" || !input.title.trim()) {
      return null;
    }
    if (!isValidStatus(input.status)) {
      return null;
    }
    const createdAt = typeof input.createdAt === "string" ? input.createdAt : fallbackTime;
    const updatedAt = typeof input.updatedAt === "string" ? input.updatedAt : createdAt;
    return {
      id: typeof input.id === "string" && input.id ? input.id : createId("issue"),
      title: input.title.trim(),
      description: typeof input.description === "string" ? input.description : "",
      status: input.status,
      priority: isValidPriority(input.priority) ? input.priority : "medium",
      createdAt,
      updatedAt
    };
  }

  function migratePersistedState(rawState) {
    const fallback = createInitialState();
    if (!rawState || typeof rawState !== "object") {
      return fallback;
    }
    const migrated = createInitialState();
    const issues = Array.isArray(rawState.issues) ? rawState.issues : [];
    const baselineTime = nowIso();
    migrated.issues = issues
      .map(function (issue) {
        return normalizeIssue(issue, baselineTime);
      })
      .filter(Boolean);
    migrated.savedViews = Array.isArray(rawState.savedViews) ? rawState.savedViews.slice() : [];
    migrated.uiState.sortMode =
      rawState.uiState && typeof rawState.uiState.sortMode === "string"
        ? rawState.uiState.sortMode
        : "updated_desc";
    return migrated;
  }

  function createIssue(state, draft) {
    const next = cloneState(state);
    const timestamp = nowIso();
    const issue = normalizeIssue(
      {
        id: createId("issue"),
        title: draft.title,
        description: draft.description || "",
        status: draft.status || "todo",
        priority: draft.priority || "medium",
        createdAt: timestamp,
        updatedAt: timestamp
      },
      timestamp
    );
    if (!issue) {
      throw new Error("Invalid issue draft");
    }
    next.issues.push(issue);
    return next;
  }

  function updateIssue(state, issueId, patch) {
    const next = cloneState(state);
    next.issues = next.issues.map(function (issue) {
      if (issue.id !== issueId) {
        return issue;
      }
      return normalizeIssue(
        {
          ...issue,
          ...patch,
          updatedAt: nowIso()
        },
        issue.createdAt
      );
    });
    return next;
  }

  function deleteIssue(state, issueId) {
    const next = cloneState(state);
    next.issues = next.issues.filter(function (issue) {
      return issue.id !== issueId;
    });
    return next;
  }

  function cloneState(state) {
    return JSON.parse(JSON.stringify(state));
  }

  function filterIssues(issues, viewState) {
    const searchTerm = (viewState.searchTerm || "").trim().toLowerCase();
    return issues.filter(function (issue) {
      if (viewState.statusFilters && viewState.statusFilters.length > 0 && !viewState.statusFilters.includes(issue.status)) {
        return false;
      }
      if (viewState.priorityFilters && viewState.priorityFilters.length > 0 && !viewState.priorityFilters.includes(issue.priority)) {
        return false;
      }
      if (!searchTerm) {
        return true;
      }
      return (issue.title + " " + issue.description).toLowerCase().includes(searchTerm);
    });
  }

  function compareIssues(sortMode, left, right) {
    if (sortMode === "title_asc") {
      return left.title.localeCompare(right.title, undefined, { sensitivity: "base" });
    }
    if (sortMode === "priority_desc") {
      return PRIORITY_RANK[right.priority] - PRIORITY_RANK[left.priority];
    }
    return String(right.updatedAt).localeCompare(String(left.updatedAt));
  }

  function getVisibleIssues(state, viewState) {
    const sortMode = viewState.sortMode || state.uiState.sortMode || "updated_desc";
    return filterIssues(state.issues, viewState).slice().sort(function (left, right) {
      return compareIssues(sortMode, left, right);
    });
  }

  function saveView() {
    throw new Error("Saved views are not implemented in the starter repository");
  }

  function renameSavedView() {
    throw new Error("Saved views are not implemented in the starter repository");
  }

  function deleteSavedView() {
    throw new Error("Saved views are not implemented in the starter repository");
  }

  function applySavedView() {
    throw new Error("Saved views are not implemented in the starter repository");
  }

  return {
    STORAGE_KEY: "issue-tracker-state-v2",
    createInitialState,
    migratePersistedState,
    createIssue,
    updateIssue,
    deleteIssue,
    getVisibleIssues,
    saveView,
    renameSavedView,
    deleteSavedView,
    applySavedView
  };
});
