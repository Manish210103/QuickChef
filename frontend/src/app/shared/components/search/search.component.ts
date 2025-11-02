import { Component, OnInit, OnChanges, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { SearchResult, GeneratedRecipe, Recipe } from '../../models/recipe.models';
import { RecipeModalComponent } from '../recipe-modal/recipe-modal.component';

@Component({
  selector: 'app-search',
  standalone: true,
  imports: [CommonModule, FormsModule, RecipeModalComponent],
  templateUrl: './search.component.html',
  styleUrl: './search.component.scss'
})
export class SearchComponent implements OnInit, OnChanges {
  @Input() selectedIngredients: string = '';

  ingredients: string = '';
  dietaryFilters: string[] = [];
  availableFilters: string[] = ['Vegetarian', 'Vegan', 'High-Protein', 'Low-Carb', 'Gluten-Free'];

  searchResults: SearchResult[] = [];
  generatedRecipe!: GeneratedRecipe;

  loading: boolean = false;
  generating: boolean = false;
  activeTab: 'search' | 'generated' = 'search';
  addingToHistory: { [key: string]: boolean } = {};

  error: string | null = null;
  success: string | null = null;

  savedRecipes: { [key: string]: { isSaved: boolean; historyId?: string } } = {};

  // Modal state for recipe details
  showDetailsModal: boolean = false;
  modalRecipe: any = null;

  // Feedback state (used for generated recipe or from modal)
  feedback: { spice_level?: string; would_cook_again?: boolean; rating?: number | null } = {};
  showGenFeedbackModal: boolean = false;

  constructor(private apiService: ApiService) { }

  ngOnInit(): void {
    if (this.selectedIngredients) {
      this.ingredients = this.selectedIngredients;
    }

    // Hydrate cached search state (persist across navigation)
    const cached = this.apiService.loadSearchState();
    if (cached) {
      this.ingredients = this.ingredients || cached.query;
      this.searchResults = cached.results || [];
    }

    // Preload favourites to pre-mark liked state
    this.apiService.getHistory(100, true).subscribe({
      next: (res) => {
        const map: { [key: string]: { isSaved: boolean; historyId?: string } } = {};
        for (const h of res.history) {
          const key = h.recipe_name;
          map[key] = { isSaved: true, historyId: h._id };
        }
        this.savedRecipes = { ...this.savedRecipes, ...map };
      },
      error: () => {}
    });
  }

  ngOnChanges(): void {
    if (this.selectedIngredients) {
      this.ingredients = this.selectedIngredients;
    }
  }

  handleSearch(): void {
    if (!this.ingredients.trim()) {
      this.showToast("Please enter at least one ingredient.", "error");
      return;
    }

    this.loading = true;
    this.apiService.searchRecipes(this.ingredients, 5).subscribe({
      next: (data) => {
        this.searchResults = data.results || [];
        this.activeTab = 'search';
        this.loading = false;
        this.showToast("Search completed successfully!", "success");
        // Persist to localStorage
        this.apiService.saveSearchState(this.ingredients, this.searchResults);
      },
      error: (error) => {
        console.error('Error searching:', error);
        this.loading = false;
        this.showToast("Failed to fetch recipes. Please try again.", "error");
      }
    });
  }

  handleGenerateRecipe(): void {
    if (!this.ingredients.trim()) {
      this.showToast("Please enter ingredients before generating a recipe.", "error");
      return;
    }

    this.generating = true;
    const ingredientsList = this.ingredients
      .split(',')
      .map(i => i.trim())
      .filter(i => i);

    this.apiService.generateRecipe(ingredientsList, this.dietaryFilters).subscribe({
      next: (data) => {
        this.generatedRecipe = data;
        this.activeTab = 'generated';
        this.generating = false;
        this.showToast("Recipe generated successfully!", "success");
      },
      error: (error) => {
        console.error('Error generating recipe:', error);
        this.generating = false;
        this.showToast("Recipe generation failed. Please try again.", "error");
      }
    });
  }

  toggleFavorite(recipe: Recipe): void {
    const recipeKey = recipe.name || recipe.title || 'recipe';
    const savedState = this.savedRecipes[recipeKey];

    if (this.addingToHistory[recipeKey]) return;

    this.addingToHistory[recipeKey] = true;

    if (savedState?.isSaved && savedState.historyId) {
      this.apiService.deleteHistoryItem(savedState.historyId).subscribe({
        next: () => {
          delete this.savedRecipes[recipeKey];
          this.addingToHistory[recipeKey] = false;
          this.showToast("Removed from favorites.", "error");
        },
        error: (error) => {
          console.error('Error removing from favorites:', error);
          this.addingToHistory[recipeKey] = false;
          this.showToast("Failed to remove recipe.", "error");
        }
      });
    } else {
      this.apiService.addToHistory(recipe, true).subscribe({
        next: (response) => {
          const historyId = response.history_id || response.data?.history_id;
          this.savedRecipes[recipeKey] = {
            isSaved: true,
            historyId: historyId
          };
          this.addingToHistory[recipeKey] = false;
          this.showToast("Added to favorites!", "success");
        },
        error: (error) => {
          console.error('Error adding to favorites:', error);
          this.addingToHistory[recipeKey] = false;
          this.showToast("Failed to save recipe.", "error");
        }
      });
    }
  }

  isRecipeSaved(recipe: Recipe): boolean {
    const recipeKey = recipe.name || recipe.title || 'recipe';
    return this.savedRecipes[recipeKey]?.isSaved || false;
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

  setIngredients(ingredients: string): void {
    this.ingredients = ingredients;
  }

  truncateText(text: string | string[] | undefined, maxLength: number): string {
    if (!text) return '';
    const str = Array.isArray(text) ? text.join(', ') : text;
    return str.length <= maxLength ? str : str.substring(0, maxLength) + '...';
  }

  isAddingToHistory(recipe: Recipe): boolean {
    const recipeKey = recipe.name || recipe.title || 'recipe';
    return this.addingToHistory[recipeKey] || false;
  }

  // ===== Details Modal =====
  openDetails(result: SearchResult): void {
    this.modalRecipe = result.recipe;
    this.showDetailsModal = true;
  }

  closeDetails(): void {
    this.showDetailsModal = false;
    this.modalRecipe = null;
  }

  openGenFeedbackModal(): void {
    this.showGenFeedbackModal = true;
  }

  closeGenFeedbackModal(): void {
    this.showGenFeedbackModal = false;
  }

  // ===== Feedback submission =====
  submitFeedbackFor(recipe: any): void {
    const payload = {
      recipe_id: recipe.id ?? null,
      recipe_name: recipe.name ?? recipe.title ?? 'Custom Recipe',
      answers: {
        spice_level: this.feedback.spice_level || null,
        would_cook_again: this.feedback.would_cook_again ?? null
      },
      rating: this.feedback.rating ?? null,
      generated: !!recipe.title && !recipe.name // heuristic: generated recipe has title
    };
    this.apiService.submitFeedback(payload).subscribe({
      next: () => {
        this.showToast('Feedback submitted. Thanks!', 'success');
        this.feedback = {};
      },
      error: () => this.showToast('Failed to submit feedback', 'error')
    });
  }

  // Handle feedback event from modal with its own local state
  onModalFeedback(e: { recipe: any; feedback: any }): void {
    const recipe = e.recipe || {};
    const f = e.feedback || {};
    const payload = {
      recipe_id: recipe.id ?? null,
      recipe_name: recipe.name ?? recipe.title ?? 'Custom Recipe',
      answers: {
        spice_level: f.spice_level ?? null,
        would_cook_again: f.would_cook_again ?? null
      },
      rating: f.rating ?? null,
      generated: !!recipe.title && !recipe.name
    };
    this.apiService.submitFeedback(payload).subscribe({
      next: () => this.showToast('Feedback submitted. Thanks!', 'success'),
      error: () => this.showToast('Failed to submit feedback', 'error')
    });
  }

  showToast(message: string, type: 'success' | 'error'): void {
    if (type === 'success') {
      this.success = message;
      this.error = null;
    } else {
      this.error = message;
      this.success = null;
    }

    setTimeout(() => {
      this.error = null;
      this.success = null;
    }, 3000);
  }
}
