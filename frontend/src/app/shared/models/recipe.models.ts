export interface Recipe {
  id?: string;
  name?: string;
  title?: string;
  cuisine?: string;
  total_time?: number;
  ingredients?: string | string[];
  instructions?: string | string[];
  estimated_time?: number;
}

export interface SearchResult {
  recipe_name: string;
  score: number;
  recipe: Recipe;
}

export interface Recommendation {
  recipe: Recipe;
  score: number;
}

export interface HistoryItem {
  _id: string;
  recipe_id: string;
  recipe_name: string;
  rating?: number;
  cooked_at: string;
  notes?: string;
  ingredients?: string[];
  instructions?: string[];
  favourite: boolean;
}

export interface GeneratedRecipe {
  title: string;
  estimated_time: number;
  ingredients: string[];
  instructions: string[];
}

export interface UserProfile {
  user_id: string;
  username: string;
  email: string;
  preferences: {
    dietary_restrictions?: string[];
    spice_level?: string;
    favorite_cuisines?: string[];
    preferred_cooking_time: number;
  };
  created_at: string;
}
