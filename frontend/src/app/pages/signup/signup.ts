import { Component } from '@angular/core';
import { AuthService } from '../../services/auth';
import { Router, RouterLink } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-signup',
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './signup.html',
  styleUrl: './signup.scss'
})
export class Signup {
  username!: string;
  email!: string;
  password!: string;
  preferences = {
    favorite_cuisines: [] as string[],
    dietary_restrictions: [] as string[],
    preferred_cooking_time: null as number | null,
    spice_level: 'medium'
  };

  error: string | null = null;
  success: string | null = null;
  showPreferences = false;

  cuisines = ['Italian', 'South Indian', 'North Indian', 'Continental', 'Mexican'];
  restrictions = ['Vegan', 'Vegetarian', 'Gluten-Free', 'Dairy-Free', 'Nut-Free'];
  spiceLevels = ['mild', 'medium', 'hot'];

  constructor(private authService: AuthService, private router: Router) {}

  onSignup() {

    const userData = {
      username: this.username,
      email: this.email,
      password: this.password,
      preferences: this.preferences
    };

    this.authService.register(userData).subscribe({
      next: (res) => {
        this.authService.saveToken(res.access_token);
        this.success = 'Signup successful!';
        this.error = null;

        setTimeout(() => { 
          this.success = null; 
          this.showPreferences = true;
        }, 2000);
      },
      error: (err) => {
        this.error = err.error.detail || 'Signup failed';
        this.success = null;
        setTimeout(() => { this.error = null; }, 4000);
      }
    });
  }

  toggleCuisine(cuisine: string, checked: boolean) {
    if (checked) {
      if (!this.preferences.favorite_cuisines.includes(cuisine)) {
        this.preferences.favorite_cuisines.push(cuisine);
      }
    } else {
      this.preferences.favorite_cuisines = this.preferences.favorite_cuisines.filter((c: string) => c !== cuisine);
    }
  }

  toggleRestriction(restriction: string, checked: boolean) {
    if (checked) {
      if (!this.preferences.dietary_restrictions.includes(restriction)) {
        this.preferences.dietary_restrictions.push(restriction);
      }
    } else {
      this.preferences.dietary_restrictions = this.preferences.dietary_restrictions.filter((r: string) => r !== restriction);
    }
  }

  get canSavePreferences(): boolean {
    return (
      this.preferences.favorite_cuisines.length > 0 &&
      this.preferences.preferred_cooking_time !== null &&
      this.preferences.preferred_cooking_time !== null &&
      !!this.preferences.spice_level
    );
  }

  savePreferences() {
    this.authService.updatePreferences(this.preferences).subscribe({
      next: () => {
        this.showPreferences = false;
        this.success = 'Preferences saved!';
        setTimeout(() => {
          this.success = null;
          this.router.navigate(['/dashboard']);
        }, 1000);
      },
      error: (err) => {
        console.error("Failed to save preferences", err);
        this.error = 'Failed to save preferences';
        setTimeout(() => { this.error = null; }, 4000);
      }
    });
  }
}