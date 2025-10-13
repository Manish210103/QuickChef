import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { HeaderComponent } from '../../shared/components/header/header.component';
import { RecommendationsComponent } from '../../shared/components/recommendations/recommendations.component';
import { SearchComponent } from '../../shared/components/search/search.component';
import { ApiService } from '../../shared/services/api.service';

@Component({
  selector: 'app-dashboard',
  imports: [
    CommonModule,
    HeaderComponent,
    RecommendationsComponent,
    SearchComponent
  ],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.scss'
})
export class DashboardComponent implements OnInit {
  selectedIngredients: string = '';

  constructor(
    private apiService: ApiService,
    private router: Router
  ) {}

  ngOnInit(): void {}

  onLogoutClick(): void {
    localStorage.removeItem('token');
    this.router.navigate(['/login']);
  }

  onRecommendationSelected(ingredients: string): void {
    this.selectedIngredients = ingredients;
  }
}
