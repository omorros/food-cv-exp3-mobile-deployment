import * as SecureStore from 'expo-secure-store';
import { Token, User, DraftItem, DraftItemCreate, InventoryItem, InventoryItemCreate, InventoryItemUpdate, LoginCredentials, RegisterCredentials } from '../types';

// Update this to your backend URL
// const API_BASE_URL = 'http://10.0.2.2:8000'; // Android emulator localhost
// const API_BASE_URL = 'http://localhost:8000'; // iOS simulator
const API_BASE_URL = 'http://172.20.10.4:8000'; // Physical device (your WiFi IP)

const TOKEN_KEY = 'auth_token';

class ApiService {
  private token: string | null = null;

  async init() {
    this.token = await SecureStore.getItemAsync(TOKEN_KEY);
  }

  private async getHeaders(): Promise<HeadersInit> {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    return headers;
  }

  async setToken(token: string) {
    this.token = token;
    await SecureStore.setItemAsync(TOKEN_KEY, token);
  }

  async clearToken() {
    this.token = null;
    await SecureStore.deleteItemAsync(TOKEN_KEY);
  }

  getToken() {
    return this.token;
  }

  // Auth endpoints
  async register(credentials: RegisterCredentials): Promise<Token> {
    const response = await fetch(`${API_BASE_URL}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(credentials),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Registration failed');
    }

    const data: Token = await response.json();
    await this.setToken(data.access_token);
    return data;
  }

  async login(credentials: LoginCredentials): Promise<Token> {
    const response = await fetch(`${API_BASE_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(credentials),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Login failed');
    }

    const data: Token = await response.json();
    await this.setToken(data.access_token);
    return data;
  }

  async logout() {
    await this.clearToken();
  }

  async getCurrentUser(): Promise<User> {
    const response = await fetch(`${API_BASE_URL}/auth/me`, {
      headers: await this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error('Failed to fetch user profile');
    }

    return response.json();
  }

  // Draft items endpoints
  async getDraftItems(): Promise<DraftItem[]> {
    const response = await fetch(`${API_BASE_URL}/api/draft-items`, {
      headers: await this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error('Failed to fetch draft items');
    }

    return response.json();
  }

  async createDraftItem(data: DraftItemCreate): Promise<DraftItem> {
    const response = await fetch(`${API_BASE_URL}/api/draft-items`, {
      method: 'POST',
      headers: await this.getHeaders(),
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to create draft item');
    }

    return response.json();
  }

  async deleteDraftItem(id: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/api/draft-items/${id}`, {
      method: 'DELETE',
      headers: await this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error('Failed to delete draft item');
    }
  }

  async confirmDraftItem(draftId: string, data: InventoryItemCreate): Promise<InventoryItem> {
    const response = await fetch(`${API_BASE_URL}/api/draft-items/${draftId}/confirm`, {
      method: 'POST',
      headers: await this.getHeaders(),
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to confirm draft item');
    }

    return response.json();
  }

  // Inventory endpoints
  async getInventoryItems(): Promise<InventoryItem[]> {
    const response = await fetch(`${API_BASE_URL}/api/inventory`, {
      headers: await this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error('Failed to fetch inventory items');
    }

    return response.json();
  }

  async deleteInventoryItem(id: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/api/inventory/${id}`, {
      method: 'DELETE',
      headers: await this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error('Failed to delete inventory item');
    }
  }

  async updateInventoryItem(id: string, data: InventoryItemUpdate): Promise<InventoryItem> {
    const response = await fetch(`${API_BASE_URL}/api/inventory/${id}`, {
      method: 'PUT',
      headers: await this.getHeaders(),
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to update inventory item');
    }

    return response.json();
  }

  // Helper: Add item directly to inventory (creates draft + confirms in one step)
  async addToInventory(data: InventoryItemCreate): Promise<InventoryItem> {
    // First create a draft
    const draftData: DraftItemCreate = {
      name: data.name,
      category: data.category,
      quantity: data.quantity,
      unit: data.unit,
      expiration_date: data.expiry_date,
      location: data.storage_location,
      source: 'manual',
    };

    const draft = await this.createDraftItem(draftData);

    // Immediately confirm it
    return this.confirmDraftItem(draft.id, data);
  }

  // Image ingestion endpoint
  async ingestImage(imageUri: string, storageLocation: string = 'fridge'): Promise<DraftItem[]> {
    const formData = new FormData();

    // Get file name and type from URI
    const filename = imageUri.split('/').pop() || 'photo.jpg';
    const match = /\.(\w+)$/.exec(filename);
    const type = match ? `image/${match[1]}` : 'image/jpeg';

    formData.append('image', {
      uri: imageUri,
      name: filename,
      type,
    } as any);
    formData.append('storage_location', storageLocation);

    const headers: HeadersInit = {};
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    const response = await fetch(`${API_BASE_URL}/api/ingest/image`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to process image');
    }

    return response.json();
  }
}

export const api = new ApiService();
