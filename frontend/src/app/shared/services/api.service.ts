import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Recipe, SearchResult, Recommendation, HistoryItem, GeneratedRecipe, UserProfile } from '../models/recipe.models';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private API_BASE = 'http://10.21.151.206:8000';

  constructor(private http: HttpClient) {}

  private getHeaders(): HttpHeaders {
    const token = localStorage.getItem('token') || '';
    return new HttpHeaders({
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    });
  }

  // Recommendations
  getRecommendations(count: number = 5): Observable<{recommendations: Recommendation[]}> {
    return this.http.get<{recommendations: Recommendation[]}>(`${this.API_BASE}/recommendations?count=${count}`, {
      headers: this.getHeaders()
    });
  }

  // History
  getHistory(limit: number = 10): Observable<{history: HistoryItem[]}> {
    return this.http.get<{history: HistoryItem[]}>(`${this.API_BASE}/history?limit=${limit}`, {
      headers: this.getHeaders()
    });
  }

  addToHistory(recipe: Recipe): Observable<any> {
    const requestBody = {
      recipe_id: recipe.id || 'generated',
      recipe_name: recipe.name || recipe.title || 'Custom Recipe',
      rating: 5,
      ingredients: recipe.ingredients,
      instructions: recipe.instructions
    };

    return this.http.post(`${this.API_BASE}/history/add`, requestBody, {
      headers: this.getHeaders()
    });
  }

  // Search
  searchRecipes(query: string, topK: number = 5): Observable<{results: SearchResult[]}> {
    return this.http.get<{results: SearchResult[]}>(`${this.API_BASE}/search?query=${encodeURIComponent(query)}&top_k=${topK}`);
  }

  // Recipe Generation
  generateRecipe(ingredients: string[], dietaryRestrictions: string[] = []): Observable<GeneratedRecipe> {
    const requestBody = {
      ingredients,
      dietary_restrictions: dietaryRestrictions,
      spice_level: 'medium',
      servings: 2
    };

    return this.http.post<GeneratedRecipe>(`${this.API_BASE}/generate-recipe`, requestBody);
  }

  // Profile
  getUserProfile(): Observable<UserProfile> {
    return this.http.get<UserProfile>(`${this.API_BASE}/auth/profile`, {
      headers: this.getHeaders()
    });
  }
}
