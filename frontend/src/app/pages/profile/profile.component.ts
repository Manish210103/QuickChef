import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { ApiService } from '../../shared/services/api.service';
import { UserProfile } from '../../shared/models/recipe.models';

@Component({
  selector: 'app-profile',
  imports: [CommonModule],
  templateUrl: './profile.component.html',
  styleUrl: './profile.component.scss'
})
export class ProfileComponent implements OnInit {
  userProfile: UserProfile | null = null;
  loading: boolean = true;
  error: string | null = null;

  constructor(
    private apiService: ApiService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.loadUserProfile();
  }

  public loadUserProfile(): void {
    this.loading = true;
    this.error = null;

    this.apiService.getUserProfile().subscribe({
      next: (profile) => {
        this.userProfile = profile;
        this.loading = false;
      },
      error: (error) => {
        console.error('Error fetching profile:', error);
        this.error = 'Failed to load profile. Please try again.';
        this.loading = false;
        
        // If unauthorized, redirect to login
        if (error.status === 401) {
          this.router.navigate(['/login']);
        }
      }
    });
  }

  formatDate(dateString: string): string {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  }

  goBackToDashboard(): void {
    this.router.navigate(['/dashboard']);
  }
}
