import { Component, OnInit, OnChanges, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { SearchResult, GeneratedRecipe, Recipe } from '../../models/recipe.models';

@Component({
  selector: 'app-search',
  imports: [CommonModule, FormsModule],
  templateUrl: './search.component.html',
  styleUrl: './search.component.scss'
})
export class SearchComponent implements OnInit, OnChanges {
  @Input() selectedIngredients: string = '';

  // Form Data
  ingredients: string = '';
  dietaryFilters: string[] = [];
  availableFilters: string[] = ['Vegetarian', 'Vegan', 'High-Protein', 'Low-Carb', 'Gluten-Free'];

  // Results
  searchResults: SearchResult[] = [];
  generatedRecipe!: GeneratedRecipe;

  // UI State
  loading: boolean = false;
  generating: boolean = false;
  activeTab: 'search' | 'generated' = 'search';

  constructor(private apiService: ApiService) {}

  ngOnInit(): void {
    if (this.selectedIngredients) {
      this.ingredients = this.selectedIngredients;
    }
  }

  ngOnChanges(): void {
    if (this.selectedIngredients) {
      this.ingredients = this.selectedIngredients;
    }
  }

  handleSearch(): void {
    if (!this.ingredients.trim()) return;

    this.loading = true;
    this.apiService.searchRecipes(this.ingredients, 5).subscribe({
      next: (data) => {
        this.searchResults = data.results || [];
        this.activeTab = 'search';
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

    this.apiService.generateRecipe(ingredientsList, this.dietaryFilters).subscribe({
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
    this.apiService.addToHistory(recipe).subscribe({
      next: () => {
        alert('Recipe added to history!');
      },
      error: (error) => {
        console.error('Error adding to history:', error);
      }
    });
  }

  setIngredients(ingredients: string): void {
    this.ingredients = ingredients;
  }

  truncateText(text: string | string[] | undefined, maxLength: number): string {
    if (!text) return '';

    const str = Array.isArray(text) ? text.join(', ') : text;

    if (str.length <= maxLength) return str;
    return str.substring(0, maxLength) + '...';
  }
}
