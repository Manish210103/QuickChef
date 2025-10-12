import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { Recommendation } from '../../models/recipe.models';

@Component({
  selector: 'app-recommendations',
  imports: [CommonModule],
  templateUrl: './recommendations.component.html',
  styleUrl: './recommendations.component.scss'
})
export class RecommendationsComponent implements OnInit {
  @Input() recommendations: Recommendation[] = [];
  @Output() recommendationSelected = new EventEmitter<string>();

  constructor(private apiService: ApiService) {}

  ngOnInit(): void {
    if (this.recommendations.length === 0) {
      this.loadRecommendations();
    }
  }

  private loadRecommendations(): void {
    this.apiService.getRecommendations(5).subscribe({
      next: (data) => {
        this.recommendations = data.recommendations || [];
      },
      error: (error) => {
        console.error('Error fetching recommendations:', error);
      }
    });
  }

  onRecommendationClick(recipe: any): void {
    this.recommendationSelected.emit(recipe.name || '');
  }
}
