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
  showPreferences: boolean = false;

  editForm: EditFormData = {
    username: '',
    email: '',
    preferences: {
      dietary_restrictions: [],
      favorite_cuisines: [],
      spice_level: ''
    }
  };

  // Predefined options
  availableDietaryRestrictions: string[] = [
    'Vegan','Vegetarian','Gluten-Free','Dairy-Free','Nut-Free','Low-Carb','High-Protein'
  ];
  availableCuisines: string[] = [
    'Italian','South Indian','North Indian','Continental','Mexican'
  ];
  spiceLevels: string[] = ['mild','medium','hot'];

  // Preferences model for modal
  preferences: {
    favorite_cuisines: string[];
    dietary_restrictions: string[];
    preferred_cooking_time: number | null;
    spice_level: string | '';
  } = {
    favorite_cuisines: [],
    dietary_restrictions: [],
    preferred_cooking_time: null,
    spice_level: ''
  };

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

  // Toggle helpers for option chips
  toggleRestriction(option: string): void {
    const arr = this.editForm.preferences.dietary_restrictions;
    const idx = arr.indexOf(option);
    if (idx > -1) arr.splice(idx, 1); else arr.push(option);
  }

  toggleCuisine(option: string): void {
    const arr = this.editForm.preferences.favorite_cuisines;
    const idx = arr.indexOf(option);
    if (idx > -1) arr.splice(idx, 1); else arr.push(option);
  }

  // Checkbox change versions (for (change) with checked flag)
  toggleRestrictionChecked(option: string, checked: boolean): void {
    const arr = this.editForm.preferences.dietary_restrictions;
    if (checked && !arr.includes(option)) arr.push(option);
    if (!checked) this.editForm.preferences.dietary_restrictions = arr.filter(x => x !== option);
  }

  toggleCuisineChecked(option: string, checked: boolean): void {
    const arr = this.editForm.preferences.favorite_cuisines;
    if (checked && !arr.includes(option)) arr.push(option);
    if (!checked) this.editForm.preferences.favorite_cuisines = arr.filter(x => x !== option);
  }

  isRestrictionSelected(option: string): boolean {
    return this.editForm.preferences.dietary_restrictions.includes(option);
  }

  isCuisineSelected(option: string): boolean {
    return this.editForm.preferences.favorite_cuisines.includes(option);
  }

  // ===== Preferences Modal API =====
  openPreferences(): void {
    const p = this.userProfile?.preferences || {} as any;
    this.preferences = {
      favorite_cuisines: [...(p.favorite_cuisines || [])],
      dietary_restrictions: [...(p.dietary_restrictions || [])],
      preferred_cooking_time: p.preferred_cooking_time ?? 30,
      spice_level: p.spice_level || ''
    };
    this.showPreferences = true;
  }

  toggleCuisinePref(cuisine: string, checked: boolean): void {
    const arr = this.preferences.favorite_cuisines;
    if (checked && !arr.includes(cuisine)) arr.push(cuisine);
    if (!checked) this.preferences.favorite_cuisines = arr.filter(c => c !== cuisine);
  }

  toggleRestrictionPref(r: string, checked: boolean): void {
    const arr = this.preferences.dietary_restrictions;
    if (checked && !arr.includes(r)) arr.push(r);
    if (!checked) this.preferences.dietary_restrictions = arr.filter(x => x !== r);
  }

  get canSavePreferences(): boolean {
    return (
      this.preferences.favorite_cuisines.length > 0 &&
      !!this.preferences.spice_level &&
      !!this.preferences.preferred_cooking_time &&
      this.preferences.preferred_cooking_time >= 5
    );
  }

  savePreferences(): void {
    if (!this.canSavePreferences) return;
    this.isSaving = true;
    this.apiService.updateUserProfile({
      preferences: {
        favorite_cuisines: this.preferences.favorite_cuisines,
        dietary_restrictions: this.preferences.dietary_restrictions,
        spice_level: this.preferences.spice_level,
        preferred_cooking_time: this.preferences.preferred_cooking_time
      }
    }).subscribe({
      next: (updated) => {
        this.userProfile = updated;
        this.isSaving = false;
        this.showPreferences = false;
      },
      error: () => {
        this.isSaving = false;
      }
    });
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