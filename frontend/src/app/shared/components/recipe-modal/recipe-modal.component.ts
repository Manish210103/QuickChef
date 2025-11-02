import { Component, EventEmitter, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-recipe-modal',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './recipe-modal.component.html',
  styleUrls: ['./recipe-modal.component.scss']
})
export class RecipeModalComponent {
  @Input() recipe: any;
  @Output() closed = new EventEmitter<void>();
  @Output() submitFeedback = new EventEmitter<{ recipe: any, feedback: any }>();

  feedback: { spice_level?: string; would_cook_again?: boolean; rating?: number | null } = {};

  // Predefined options used in feedback to avoid free text
  spiceLevels: string[] = ['mild','medium','hot','extra-hot'];

  // UI state
  submitted = false;

  close() {
    this.closed.emit();
  }

  onSubmitFeedback() {
    if (this.submitted) return;
    this.submitFeedback.emit({ recipe: this.recipe, feedback: this.feedback });
    this.submitted = true;
  }

  setRating(value: number) {
    if (this.submitted) return;
    this.feedback.rating = value;
  }

  parseIngredients(ingredients: string | string[] | undefined): string[] {
    if (!ingredients) return [];

    if (Array.isArray(ingredients)) {
      return ingredients;
    }

    // Split by common delimiters: newline, comma, semicolon, or bullet points
    return ingredients
      .split(/[\n,;]|•/)
      .map(item => item.trim())
      .filter(item => item.length > 0);
  }

  parseInstructions(instructions: string | string[] | undefined): string[] {
    if (!instructions) return [];

    if (Array.isArray(instructions)) {
      return instructions;
    }

    // Split by newlines or numbered patterns (1., 2., etc.)
    return instructions
      .split(/\n+|\d+\.\s*/)
      .map(item => item.trim())
      .filter(item => item.length > 0);
  }
}