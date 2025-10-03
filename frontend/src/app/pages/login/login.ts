import { Component } from '@angular/core';
import { AuthService } from '../../services/auth';
import { Router, RouterLink } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-login',
  imports: [ FormsModule, CommonModule, RouterLink ],
  templateUrl: './login.html',
  styleUrl: './login.scss'
})
export class Login {
  username = '';
  password = '';
  error: string | null = null;
  success: string | null = null;

  constructor(private authService: AuthService, private router: Router) {}

  onLogin() {
    this.authService.login({ username: this.username, password: this.password }).subscribe({
      next: (res) => {
        this.authService.saveToken(res.access_token);
        this.success = 'Login successful!';
        this.error = null;

        setTimeout(() => { 
          this.success = null; 
          this.router.navigate(['/dashboard']); 
        }, 3000);
      },
      error: (err) => {
        this.error = err.error.message || "Invalid username or password";
        this.success = null;

        setTimeout(() => { this.error = null; }, 4000);
      }
    });
  }
}