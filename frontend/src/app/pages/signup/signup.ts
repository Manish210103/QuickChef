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
  username = '';
  email = '';
  password = '';
  preferences: any = {};
  error: string | null = null;
  success: string | null = null;

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
          this.router.navigate(['/login']); 
        }, 3000);
      },
      error: (err) => {
        this.error = err.error.detail || 'Signup failed';
        this.success = null;

        setTimeout(() => { this.error = null; }, 4000);
      }
    });
  }
}