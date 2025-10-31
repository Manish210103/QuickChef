import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { map, Observable } from 'rxjs';
import { Recipe, SearchResult, Recommendation, HistoryItem, GeneratedRecipe, UserProfile } from '../models/recipe.models';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private API_BASE = 'https://quickchef-production.up.railway.app';

  constructor(private http: HttpClient) { }

  private getHeaders(): HttpHeaders {
    const token = localStorage.getItem('token') || '';
    return new HttpHeaders({
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    });
  }

  // Recommendations
  getRecommendations(count: number = 5): Observable<{ recommendations: Recommendation[] }> {
    return this.http.get<{ recommendations: Recommendation[] }>(`${this.API_BASE}/recommendations?count=${count}`, {
      headers: this.getHeaders()
    });
  }

  // History - with optional favourites filter
  getHistory(limit: number = 10, favouritesOnly: boolean = false): Observable<{ history: HistoryItem[], total: number }> {
    const url = `${this.API_BASE}/history?limit=${limit}&favourites_only=${favouritesOnly}`;
    return this.http.get<{ history: HistoryItem[], total: number }>(url, {
      headers: this.getHeaders()
    });
  }

  // Add to History - with optional favourite flag
  addToHistory(recipe: Recipe, favourite: boolean = false): Observable<any> {
    const requestBody = {
      recipe_id: recipe.id ?? 'generated',
      recipe_name: recipe.name ?? recipe.title ?? 'Custom Recipe',
      rating: 5,
      notes: '',
      favourite: favourite, // NEW FIELD
      ingredients: Array.isArray(recipe.ingredients)
        ? recipe.ingredients
        : typeof recipe.ingredients === 'string'
          ? recipe.ingredients.split(',').map(i => i.trim())
          : [],
      instructions: Array.isArray(recipe.instructions)
        ? recipe.instructions
        : typeof recipe.instructions === 'string'
          ? recipe.instructions.split('.').map(i => i.trim()).filter(Boolean)
          : []
    };

    console.log('📤 Sending to backend:', requestBody);

    return this.http.post(`${this.API_BASE}/history/add`, requestBody, {
      headers: this.getHeaders()
    });
  }

  // NEW: Toggle Favourite Status
  toggleFavourite(historyId: string, favourite: boolean): Observable<any> {
    return this.http.patch(
      `${this.API_BASE}/history/${historyId}/favourite?favourite=${favourite}`,
      {},
      { headers: this.getHeaders() }
    );
  }

  // Delete History Item
  deleteHistoryItem(id: string): Observable<{ message: string }> {
    return this.http.delete<{ message: string }>(`${this.API_BASE}/history/${id}`, {
      headers: this.getHeaders()
    });
  }

  // Search
  searchRecipes(query: string, topK: number = 5): Observable<{ results: SearchResult[] }> {
    return this.http
      .get<{ results: SearchResult[] }>(
        `${this.API_BASE}/search?query=${encodeURIComponent(query)}&top_k=${topK}`
      )
      .pipe(
        map(response => {
          const formattedResults = response.results.map(item => {
            const recipe = item.recipe || {};

            const formattedIngredients = Array.isArray(recipe.ingredients)
              ? recipe.ingredients
              : typeof recipe.ingredients === 'string'
                ? recipe.ingredients.split(',').map(i => i.trim())
                : [];

            const formattedInstructions = Array.isArray(recipe.instructions)
              ? recipe.instructions
              : typeof recipe.instructions === 'string'
                ? recipe.instructions
                  .split('.')
                  .map(i => i.trim())
                  .filter(Boolean)
                : [];

            return {
              ...item,
              recipe: {
                ...recipe,
                ingredients: formattedIngredients,
                instructions: formattedInstructions
              }
            };
          });

          return { results: formattedResults };
        })
      );
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

  // Update User Profile
  updateUserProfile(profileData: any): Observable<UserProfile> {
    return this.http.put<UserProfile>(`${this.API_BASE}/auth/update`, profileData, {
      headers: this.getHeaders()
    });
  }
}