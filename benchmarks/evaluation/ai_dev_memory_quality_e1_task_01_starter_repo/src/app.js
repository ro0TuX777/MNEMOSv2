(function () {
  const logic = window.IssueTrackerLogic;
  if (!logic) {
    throw new Error("IssueTrackerLogic is required before app.js");
  }

  const elements = {
    form: document.getElementById("issue-form"),
    title: document.getElementById("issue-title"),
    status: document.getElementById("issue-status"),
    priority: document.getElementById("issue-priority"),
    search: document.getElementById("search-input"),
    statusFilter: document.getElementById("status-filter"),
    priorityFilter: document.getElementById("priority-filter"),
    issueList: document.getElementById("issue-list"),
    issueEmptyState: document.getElementById("issue-empty-state")
  };

  let state = loadState();

  function loadState() {
    try {
      const raw = window.localStorage.getItem(logic.STORAGE_KEY);
      return logic.migratePersistedState(raw ? JSON.parse(raw) : null);
    } catch (error) {
      return logic.createInitialState();
    }
  }

  function persistState() {
    window.localStorage.setItem(logic.STORAGE_KEY, JSON.stringify(state));
  }

  function currentViewState() {
    return {
      statusFilters: elements.statusFilter.value ? [elements.statusFilter.value] : [],
      priorityFilters: elements.priorityFilter.value ? [elements.priorityFilter.value] : [],
      searchTerm: elements.search.value,
      sortMode: state.uiState.sortMode
    };
  }

  function render() {
    const visibleIssues = logic.getVisibleIssues(state, currentViewState());
    elements.issueList.innerHTML = "";
    visibleIssues.forEach(function (issue) {
      const li = document.createElement("li");
      li.className = "issue-row";
      li.innerHTML =
        "<div>" +
        "<strong>" + issue.title + "</strong>" +
        "<div class=\"issue-meta\">" +
        "<span>" + issue.status + "</span>" +
        "<span>" + issue.priority + "</span>" +
        "</div>" +
        "</div>" +
        "<button type=\"button\" data-delete=\"" + issue.id + "\">Delete</button>";
      elements.issueList.appendChild(li);
    });
    elements.issueEmptyState.classList.toggle("hidden", visibleIssues.length > 0);
  }

  elements.form.addEventListener("submit", function (event) {
    event.preventDefault();
    state = logic.createIssue(state, {
      title: elements.title.value,
      status: elements.status.value,
      priority: elements.priority.value
    });
    persistState();
    elements.form.reset();
    render();
  });

  elements.issueList.addEventListener("click", function (event) {
    const button = event.target.closest("[data-delete]");
    if (!button) {
      return;
    }
    state = logic.deleteIssue(state, button.getAttribute("data-delete"));
    persistState();
    render();
  });

  [elements.search, elements.statusFilter, elements.priorityFilter].forEach(function (element) {
    element.addEventListener("input", render);
    element.addEventListener("change", render);
  });

  render();
})();
