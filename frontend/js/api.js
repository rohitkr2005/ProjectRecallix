/**
 * Project Recallix — Frontend API Client (Phase 15, Step 15.6)
 * Handles authentication token lifecycle, API communication, and error dispatch.
 */

class RecallixAPI {
  constructor(baseURL = "") {
    this.baseURL = baseURL || window.location.origin;
    this.tokenKey = "recallix_access_token";
    this.token = localStorage.getItem(this.tokenKey) || null;
    this.user = null;
  }

  setToken(token) {
    this.token = token;
    if (token) {
      localStorage.setItem(this.tokenKey, token);
    } else {
      localStorage.removeItem(this.tokenKey);
    }
  }

  clearToken() {
    this.setToken(null);
    this.user = null;
  }

  isAuthenticated() {
    return !!this.token;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const headers = {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    };

    if (this.token) {
      headers["Authorization"] = `Bearer ${this.token}`;
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      if (response.status === 401) {
        // Token expired or invalid
        this.clearToken();
        window.dispatchEvent(new CustomEvent("recallix:unauthorized"));
        throw new Error("Session expired. Please log in again.");
      }

      const data = await response.json();

      if (!response.ok) {
        const errorMsg = data.detail || data.error || `Request failed with status ${response.status}`;
        throw new Error(errorMsg);
      }

      return data;
    } catch (err) {
      console.error(`API Error [${endpoint}]:`, err);
      throw err;
    }
  }

  // -------------------------------------------------------------------------
  // Auth Endpoints
  // -------------------------------------------------------------------------
  async register(username, email, password) {
    return await this.request("/api/v1/auth/register", {
      method: "POST",
      body: JSON.stringify({ username, email, password }),
    });
  }

  async login(username, password) {
    const data = await this.request("/api/v1/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    if (data.access_token) {
      this.setToken(data.access_token);
      this.user = data.user;
    }
    return data;
  }

  async getMe() {
    const user = await this.request("/api/v1/auth/me", {
      method: "GET",
    });
    this.user = user;
    return user;
  }

  async getHealth() {
    return await this.request("/api/health", {
      method: "GET",
    });
  }

  // -------------------------------------------------------------------------
  // Chat Endpoint
  // -------------------------------------------------------------------------
  async chat(message, topK = 5, includeExplanations = true) {
    return await this.request("/api/v1/chat", {
      method: "POST",
      body: JSON.stringify({
        message,
        top_k: topK,
        include_explanations: includeExplanations,
      }),
    });
  }

  // -------------------------------------------------------------------------
  // Memory CRUD & Search Endpoints
  // -------------------------------------------------------------------------
  async listMemories({ page = 1, pageSize = 20, category = null, includeArchived = false, search = null } = {}) {
    const params = new URLSearchParams({
      page: page.toString(),
      page_size: pageSize.toString(),
      include_archived: includeArchived.toString(),
    });

    if (category) params.append("category", category);
    if (search) params.append("search", search);

    return await this.request(`/api/v1/memories?${params.toString()}`, {
      method: "GET",
    });
  }

  async createMemory({ subject = "User", relation, value, category = "GENERAL", importance = 5, temporalState = "PRESENT" }) {
    return await this.request("/api/v1/memories", {
      method: "POST",
      body: JSON.stringify({
        subject,
        relation,
        value,
        category,
        importance,
        temporal_state: temporalState,
      }),
    });
  }

  async getMemory(memoryId) {
    return await this.request(`/api/v1/memories/${memoryId}`, {
      method: "GET",
    });
  }

  async updateMemory(memoryId, updateData) {
    return await this.request(`/api/v1/memories/${memoryId}`, {
      method: "PUT",
      body: JSON.stringify(updateData),
    });
  }

  async archiveMemory(memoryId) {
    return await this.request(`/api/v1/memories/${memoryId}`, {
      method: "DELETE",
    });
  }

  async restoreMemory(memoryId) {
    return await this.request(`/api/v1/memories/${memoryId}/restore`, {
      method: "POST",
    });
  }

  async searchMemories(query, topK = 5, minScore = 0.0) {
    return await this.request("/api/v1/search", {
      method: "POST",
      body: JSON.stringify({
        query,
        top_k: topK,
        min_score: minScore,
      }),
    });
  }
}

// Global API singleton instance
window.recallixAPI = new RecallixAPI();
