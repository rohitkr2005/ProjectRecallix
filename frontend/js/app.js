/**
 * Project Recallix — Application UI Logic (Phase 15, Steps 15.6 - 15.10)
 * Manages view switching, authentication states, chat, memory CRUD, and search.
 */

document.addEventListener("DOMContentLoaded", () => {
  const api = window.recallixAPI;

  // DOM Elements - Views
  const authView = document.getElementById("authView");
  const appContainer = document.getElementById("appContainer");

  // Auth Elements
  const authTabLogin = document.getElementById("authTabLogin");
  const authTabRegister = document.getElementById("authTabRegister");
  const loginForm = document.getElementById("loginForm");
  const registerForm = document.getElementById("registerForm");
  const authAlert = document.getElementById("authAlert");
  const currentUserName = document.getElementById("currentUserName");
  const currentUserAvatar = document.getElementById("currentUserAvatar");
  const logoutBtn = document.getElementById("logoutBtn");

  // Navigation Items & View Sections
  const navItems = document.querySelectorAll(".nav-item");
  const viewSections = document.querySelectorAll(".view-section");

  // Chat Elements
  const chatMessages = document.getElementById("chatMessages");
  const chatInput = document.getElementById("chatInput");
  const chatSendBtn = document.getElementById("chatSendBtn");

  // Memory Elements
  const memoryGrid = document.getElementById("memoryGrid");
  const memorySearchInput = document.getElementById("memorySearchInput");
  const memoryCategoryFilter = document.getElementById("memoryCategoryFilter");
  const showArchivedToggle = document.getElementById("showArchivedToggle");
  const addMemoryBtn = document.getElementById("addMemoryBtn");
  const memoryModal = document.getElementById("memoryModal");
  const closeMemoryModal = document.getElementById("closeMemoryModal");
  const createMemoryForm = document.getElementById("createMemoryForm");

  // Search Elements
  const searchInput = document.getElementById("searchInput");
  const searchBtn = document.getElementById("searchBtn");
  const searchResultsGrid = document.getElementById("searchResultsGrid");

  // Settings / Diagnostics Elements
  const diagDbStatus = document.getElementById("diagDbStatus");
  const diagLlmStatus = document.getElementById("diagLlmStatus");
  const diagVersion = document.getElementById("diagVersion");
  const diagUserId = document.getElementById("diagUserId");
  const diagUsername = document.getElementById("diagUsername");
  const diagEmail = document.getElementById("diagEmail");

  // -------------------------------------------------------------------------
  // Auth State & Tab Management (15.7)
  // -------------------------------------------------------------------------
  function showAuthAlert(message, type = "error") {
    authAlert.textContent = message;
    authAlert.className = `auth-alert ${type}`;
  }

  function hideAuthAlert() {
    authAlert.style.display = "none";
  }

  authTabLogin.addEventListener("click", () => {
    authTabLogin.classList.add("active");
    authTabRegister.classList.remove("active");
    loginForm.style.display = "block";
    registerForm.style.display = "none";
    hideAuthAlert();
  });

  authTabRegister.addEventListener("click", () => {
    authTabRegister.classList.add("active");
    authTabLogin.classList.remove("active");
    registerForm.style.display = "block";
    loginForm.style.display = "none";
    hideAuthAlert();
  });

  async function checkAuthSession() {
    if (api.isAuthenticated()) {
      try {
        const user = await api.getMe();
        setupAuthenticatedView(user);
      } catch (err) {
        showUnauthenticatedView();
      }
    } else {
      showUnauthenticatedView();
    }
  }

  function setupAuthenticatedView(user) {
    authView.style.display = "none";
    appContainer.style.display = "flex";

    currentUserName.textContent = user.username;
    currentUserAvatar.textContent = (user.username || "U")[0].toUpperCase();

    // Diagnostics in settings
    diagUserId.textContent = user.id;
    diagUsername.textContent = user.username;
    diagEmail.textContent = user.email;

    // Load initial data
    loadMemories();
    loadDiagnostics();
  }

  function showUnauthenticatedView() {
    appContainer.style.display = "none";
    authView.style.display = "flex";
  }

  // Handle Unauthorized events
  window.addEventListener("recallix:unauthorized", () => {
    showUnauthenticatedView();
    showAuthAlert("Your session has expired. Please sign in again.", "error");
  });

  // Login Submit
  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideAuthAlert();
    const username = document.getElementById("loginUsername").value.trim();
    const password = document.getElementById("loginPassword").value;

    try {
      const data = await api.login(username, password);
      setupAuthenticatedView(data.user);
    } catch (err) {
      showAuthAlert(err.message || "Invalid username or password");
    }
  });

  // Register Submit
  registerForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideAuthAlert();
    const username = document.getElementById("regUsername").value.trim();
    const email = document.getElementById("regEmail").value.trim();
    const password = document.getElementById("regPassword").value;

    try {
      await api.register(username, email, password);
      showAuthAlert("Account created successfully! Signing in...", "success");
      const loginData = await api.login(username, password);
      setupAuthenticatedView(loginData.user);
    } catch (err) {
      showAuthAlert(err.message || "Registration failed");
    }
  });

  // Logout
  logoutBtn.addEventListener("click", () => {
    api.clearToken();
    showUnauthenticatedView();
  });

  // -------------------------------------------------------------------------
  // View Navigation (15.6)
  // -------------------------------------------------------------------------
  navItems.forEach((item) => {
    item.addEventListener("click", () => {
      const targetView = item.getAttribute("data-view");

      navItems.forEach((n) => n.classList.remove("active"));
      viewSections.forEach((s) => s.classList.remove("active"));

      item.classList.add("active");
      const section = document.getElementById(`${targetView}Section`);
      if (section) section.classList.add("active");

      // Trigger view-specific data refresh
      if (targetView === "memories") loadMemories();
      if (targetView === "settings") loadDiagnostics();
    });
  });

  // -------------------------------------------------------------------------
  // Chat Interface (15.8 & 15.10)
  // -------------------------------------------------------------------------
  function appendChatMessage(text, sender = "user", meta = null) {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${sender}`;

    const textElem = document.createElement("div");
    textElem.textContent = text;
    bubble.appendChild(textElem);

    if (sender === "assistant" && meta) {
      // Grounding badge
      if (meta.grounded && meta.supported) {
        const badge = document.createElement("div");
        badge.className = "badge-grounded";
        badge.innerHTML = "✓ Grounded Memory";
        bubble.appendChild(badge);
      } else if (!meta.supported) {
        const badge = document.createElement("div");
        badge.className = "badge-unsupported";
        badge.innerHTML = "⚠ Unsupported / No Memory";
        bubble.appendChild(badge);
      }

      // Explainability drawer (15.10)
      if (meta.why_used && meta.why_used.length > 0) {
        const whyBox = document.createElement("div");
        whyBox.className = "why-used-box";
        whyBox.innerHTML = `<strong>Why used:</strong><br>${meta.why_used.join("<br>")}`;
        bubble.appendChild(whyBox);
      }
    }

    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  async function handleSendChat() {
    const message = chatInput.value.trim();
    if (!message) return;

    appendChatMessage(message, "user");
    chatInput.value = "";
    chatInput.disabled = true;
    chatSendBtn.disabled = true;

    try {
      const result = await api.chat(message, 5, true);
      appendChatMessage(result.response, "assistant", result);
    } catch (err) {
      appendChatMessage(`Error: ${err.message}`, "assistant");
    } finally {
      chatInput.disabled = false;
      chatSendBtn.disabled = false;
      chatInput.focus();
    }
  }

  chatSendBtn.addEventListener("click", handleSendChat);
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendChat();
    }
  });

  // -------------------------------------------------------------------------
  // Memory Management UI (15.9)
  // -------------------------------------------------------------------------
  async function loadMemories() {
    const search = memorySearchInput.value.trim() || null;
    const category = memoryCategoryFilter.value || null;
    const includeArchived = showArchivedToggle.checked;

    try {
      const data = await api.listMemories({
        page: 1,
        pageSize: 50,
        category,
        includeArchived,
        search,
      });
      renderMemoryCards(data.items);
    } catch (err) {
      memoryGrid.innerHTML = `<div style="color: var(--text-muted);">Failed to load memories: ${err.message}</div>`;
    }
  }

  function renderMemoryCards(memories) {
    if (!memories || memories.length === 0) {
      memoryGrid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-secondary);">No memories found. Start chatting or click "+ Add Memory" to store facts.</div>`;
      return;
    }

    memoryGrid.innerHTML = "";
    memories.forEach((mem) => {
      const card = document.createElement("div");
      card.className = "glass-panel memory-card";
      if (!mem.active) card.style.opacity = "0.6";

      const relationFormatted = mem.relation.replace(/_/g, " ");

      card.innerHTML = `
        <div class="memory-card-header">
          <span class="memory-category-tag">${mem.category || "GENERAL"}</span>
          <span class="memory-importance">Importance: ${mem.importance || 5}/10</span>
        </div>
        <div class="memory-fact">
          <span class="memory-relation">${mem.subject} ${relationFormatted}:</span>
          <span class="memory-value">${mem.value}</span>
        </div>
        <div class="memory-card-footer">
          <span>${mem.temporal_state || "PRESENT"}</span>
          <div class="card-actions">
            ${
              mem.active
                ? `<button class="btn-card-action danger" onclick="archiveMemoryItem(${mem.id})">Archive</button>`
                : `<button class="btn-card-action" onclick="restoreMemoryItem(${mem.id})">Restore</button>`
            }
          </div>
        </div>
      `;
      memoryGrid.appendChild(card);
    });
  }

  // Memory Actions (Exposed to window for inline onclick handlers)
  window.archiveMemoryItem = async function (id) {
    try {
      await api.archiveMemory(id);
      loadMemories();
    } catch (err) {
      alert(`Failed to archive: ${err.message}`);
    }
  };

  window.restoreMemoryItem = async function (id) {
    try {
      await api.restoreMemory(id);
      loadMemories();
    } catch (err) {
      alert(`Failed to restore: ${err.message}`);
    }
  };

  // Memory Filters & Search
  memorySearchInput.addEventListener("input", debounce(loadMemories, 300));
  memoryCategoryFilter.addEventListener("change", loadMemories);
  showArchivedToggle.addEventListener("change", loadMemories);

  // Add Memory Modal
  addMemoryBtn.addEventListener("click", () => {
    memoryModal.classList.add("active");
  });

  closeMemoryModal.addEventListener("click", () => {
    memoryModal.classList.remove("active");
  });

  createMemoryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const relation = document.getElementById("newMemRelation").value.trim();
    const value = document.getElementById("newMemValue").value.trim();
    const category = document.getElementById("newMemCategory").value;
    const importance = parseInt(document.getElementById("newMemImportance").value, 10);
    const temporalState = document.getElementById("newMemTemporal").value;

    try {
      await api.createMemory({
        subject: "User",
        relation,
        value,
        category,
        importance,
        temporalState,
      });
      memoryModal.classList.remove("active");
      createMemoryForm.reset();
      loadMemories();
    } catch (err) {
      alert(`Failed to create memory: ${err.message}`);
    }
  });

  // -------------------------------------------------------------------------
  // Search Sandbox (15.5 & 15.10)
  // -------------------------------------------------------------------------
  async function handleSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    searchResultsGrid.innerHTML = `<div style="color: var(--text-secondary); padding: 20px;">Searching memories...</div>`;

    try {
      const data = await api.searchMemories(query, 5, 0.0);
      if (!data.results || data.results.length === 0) {
        searchResultsGrid.innerHTML = `<div style="color: var(--text-secondary); padding: 20px;">No matching memories found for "${query}".</div>`;
        return;
      }

      searchResultsGrid.innerHTML = "";
      data.results.forEach((item) => {
        const mem = item.memory;
        const scorePercent = (item.score * 100).toFixed(1);
        const reasons = item.explanation?.reasons?.join(", ") || "Relevance match";

        const card = document.createElement("div");
        card.className = "glass-panel memory-card";
        card.innerHTML = `
          <div class="memory-card-header">
            <span class="memory-category-tag">${mem.category}</span>
            <span class="badge-grounded" style="margin: 0;">Score: ${scorePercent}%</span>
          </div>
          <div class="memory-fact">
            <span class="memory-relation">${mem.subject} ${mem.relation.replace(/_/g, " ")}:</span>
            <span class="memory-value">${mem.value}</span>
          </div>
          <div class="why-used-box" style="margin-top: 8px;">
            <strong>Ranking signal:</strong> ${reasons}
          </div>
        `;
        searchResultsGrid.appendChild(card);
      });
    } catch (err) {
      searchResultsGrid.innerHTML = `<div style="color: var(--status-error); padding: 20px;">Search failed: ${err.message}</div>`;
    }
  }

  searchBtn.addEventListener("click", handleSearch);
  searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleSearch();
  });

  // -------------------------------------------------------------------------
  // Settings & Diagnostics (15.11)
  // -------------------------------------------------------------------------
  async function loadDiagnostics() {
    try {
      const health = await api.getHealth();
      diagDbStatus.textContent = health.database;
      diagLlmStatus.textContent = health.llm;
      diagVersion.textContent = health.version;
    } catch (err) {
      diagDbStatus.textContent = "Error";
      diagLlmStatus.textContent = "Error";
    }
  }

  // Utility Debounce function
  function debounce(func, wait) {
    let timeout;
    return function (...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(this, args), wait);
    };
  }

  // Initialize
  checkAuthSession();
});
