import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { ApiService } from '../../shared/services/api.service';
import { UserProfile } from '../../shared/models/recipe.models';

interface EditFormData {
  username?: string;
  email?: string;
  preferences: {
    dietary_restrictions: string[];
    favorite_cuisines: string[];
    spice_level?: string;
  };
}

@Component({
  selector: 'app-profile',
  imports: [CommonModule, FormsModule],
  templateUrl: './profile.component.html',
  styleUrl: './profile.component.scss'
})
export class ProfileComponent implements OnInit {
  userProfile: UserProfile | null = null;
  loading: boolean = true;
  error: string | null = null;
  isEditMode: boolean = false;
  isSaving: boolean = false;

  editForm: EditFormData = {
    username: '',
    email: '',
    preferences: {
      dietary_restrictions: [],
      favorite_cuisines: [],
      spice_level: ''
    }
  };

  newRestriction: string = '';
  newCuisine: string = '';

  constructor(
    private apiService: ApiService,
    private router: Router
  ) { }

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

        if (error.status === 401) {
          this.router.navigate(['/login']);
        }
      }
    });
  }

  toggleEditMode(): void {
    if (this.isEditMode) {
      this.isEditMode = false;
    } else {
      this.initEditForm();
      this.isEditMode = true;
    }
  }

  initEditForm(): void {
    if (this.userProfile) {
      this.editForm = {
        username: this.userProfile.username,
        email: this.userProfile.email,
        preferences: {
          dietary_restrictions: [...(this.userProfile.preferences?.dietary_restrictions || [])],
          favorite_cuisines: [...(this.userProfile.preferences?.favorite_cuisines || [])],
          spice_level: this.userProfile.preferences?.spice_level || ''
        }
      };
    }
  }

  addRestriction(): void {
    const restriction = this.newRestriction.trim();
    if (restriction && !this.editForm.preferences.dietary_restrictions.includes(restriction)) {
      this.editForm.preferences.dietary_restrictions.push(restriction);
      this.newRestriction = '';
    }
  }

  removeRestriction(index: number): void {
    this.editForm.preferences.dietary_restrictions.splice(index, 1);
  }

  addCuisine(): void {
    const cuisine = this.newCuisine.trim();
    if (cuisine && !this.editForm.preferences.favorite_cuisines.includes(cuisine)) {
      this.editForm.preferences.favorite_cuisines.push(cuisine);
      this.newCuisine = '';
    }
  }

  removeCuisine(index: number): void {
    this.editForm.preferences.favorite_cuisines.splice(index, 1);
  }

  saveProfile(): void {
    this.isSaving = true;
    this.error = null;

    this.apiService.updateUserProfile(this.editForm).subscribe({
      next: (updatedProfile) => {
        this.userProfile = updatedProfile;
        this.isEditMode = false;
        this.isSaving = false;
        console.log('Profile updated successfully!');
      },
      error: (error) => {
        console.error('Error updating profile:', error);
        this.error = 'Failed to update profile. Please try again.';
        this.isSaving = false;
      }
    });
  }

  cancelEdit(): void {
    this.isEditMode = false;
    this.newRestriction = '';
    this.newCuisine = '';
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