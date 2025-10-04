import { Component, OnInit } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';

interface Recipe {
  id?: string;
  name?: string;
  title?: string;
  cuisine?: string;
  total_time?: number;
  ingredients?: string | string[];
  instructions?: string | string[];
  estimated_time?: number;
}

interface SearchResult {
  recipe: Recipe;
  score: number;
}

interface Recommendation {
  recipe: Recipe;
  score: number;
}

interface HistoryItem {
  _id: string;
  recipe_id: string;
  recipe_name: string;
  rating?: number;
  cooked_at: string;
  notes?: string;
}

interface GeneratedRecipe {
  title: string;
  estimated_time: number;
  ingredients: string[];
  instructions: string[];
}

@Component({
  selector: 'app-dashboard',
  imports: [CommonModule, FormsModule],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.scss'
})
export class Dashboard implements OnInit {
  // API Configuration
  private API_BASE = 'http://127.0.0.1:8000';
  private token = localStorage.getItem('token') || '';

  // Form Data
  ingredients: string = '';
  dietaryFilters: string[] = [];
  availableFilters: string[] = ['Vegetarian', 'Vegan', 'High-Protein', 'Low-Carb', 'Gluten-Free'];

  // Results
  searchResults: SearchResult[] = [];
  generatedRecipe!: GeneratedRecipe;
  recommendations: Recommendation[] = [];
  history: HistoryItem[] = [];
  similarDishes: SearchResult[] = [];

  // UI State
  loading: boolean = false;
  generating: boolean = false;
  activeTab: 'search' | 'generated' = 'search';

  constructor(private http: HttpClient, private router: Router) {}

  ngOnInit(): void {
    this.fetchRecommendations();
    this.fetchHistory();
  }

  private getHeaders(): HttpHeaders {
    return new HttpHeaders({
      'Authorization': `Bearer ${this.token}`,
      'Content-Type': 'application/json'
    });
  }

  fetchRecommendations(): void {
    this.http.get<any>(`${this.API_BASE}/recommendations?count=5`, {
      headers: this.getHeaders()
    }).subscribe({
      next: (data) => {
        this.recommendations = data.recommendations || [];
      },
      error: (error) => {
        console.error('Error fetching recommendations:', error);
      }
    });
  }

  fetchHistory(): void {
    this.http.get<any>(`${this.API_BASE}/history?limit=10`, {
      headers: this.getHeaders()
    }).subscribe({
      next: (data) => {
        this.history = data.history || [];
      },
      error: (error) => {
        console.error('Error fetching history:', error);
      }
    });
  }

  handleSearch(): void {
    if (!this.ingredients.trim()) return;

    this.loading = true;
    this.http.get<any>(`${this.API_BASE}/search?query=${encodeURIComponent(this.ingredients)}&top_k=5`)
      .subscribe({
        next: (data) => {
          this.searchResults = data.results || [];
          this.activeTab = 'search';

          // Fetch similar dishes based on first result
          if (data.results && data.results.length > 0) {
            const firstRecipe = data.results[0];
            const similarQuery = `${firstRecipe.recipe.cuisine} ${firstRecipe.recipe.name}`;
            
            this.http.get<any>(`${this.API_BASE}/search?query=${encodeURIComponent(similarQuery)}&top_k=4`)
              .subscribe({
                next: (similarData) => {
                  this.similarDishes = similarData.results || [];
                },
                error: (error) => console.error('Error fetching similar dishes:', error)
              });
          }

          this.loading = false;
        },
        error: (error) => {
          console.error('Error searching:', error);
          this.loading = false;
        }
      });
  }

  handleGenerateRecipe(): void {
    if (!this.ingredients.trim()) return;

    this.generating = true;
    const ingredientsList = this.ingredients
      .split(',')
      .map(i => i.trim())
      .filter(i => i);

    const requestBody = {
      ingredients: ingredientsList,
      dietary_restrictions: this.dietaryFilters,
      spice_level: 'medium',
      servings: 2
    };

    this.http.post<GeneratedRecipe>(`${this.API_BASE}/generate-recipe`, requestBody)
      .subscribe({
        next: (data) => {
          this.generatedRecipe = data;
          this.activeTab = 'generated';
          this.generating = false;
        },
        error: (error) => {
          console.error('Error generating recipe:', error);
          this.generating = false;
        }
      });
  }

  toggleDietaryFilter(filter: string): void {
    const index = this.dietaryFilters.indexOf(filter);
    if (index > -1) {
      this.dietaryFilters.splice(index, 1);
    } else {
      this.dietaryFilters.push(filter);
    }
  }

  isDietaryFilterActive(filter: string): boolean {
    return this.dietaryFilters.includes(filter);
  }

  handleAddToHistory(recipe: Recipe): void {
    const requestBody = {
      recipe_id: recipe.id || 'generated',
      recipe_name: recipe.name || recipe.title || 'Custom Recipe',
      rating: 5
    };

    this.http.post(`${this.API_BASE}/history/add`, requestBody, {
      headers: this.getHeaders()
    }).subscribe({
      next: () => {
        this.fetchHistory();
        alert('Recipe added to history!');
      },
      error: (error) => {
        console.error('Error adding to history:', error);
      }
    });
  }

  selectRecommendation(recipe: Recipe): void {
    this.ingredients = recipe.name || '';
  }

  selectSimilarDish(recipe: Recipe): void {
    this.ingredients = recipe.name || '';
  }

  formatDate(dateString: string): string {
    return new Date(dateString).toLocaleDateString();
  }

  truncateText(text: string | string[] | undefined, maxLength: number): string {
    if (!text) return '';

    const str = Array.isArray(text) ? text.join(', ') : text;

    if (str.length <= maxLength) return str;
    return str.substring(0, maxLength) + '...';
  }

  logout(): void {
    localStorage.removeItem('token');
    this.router.navigate(['/login']);
  }
}