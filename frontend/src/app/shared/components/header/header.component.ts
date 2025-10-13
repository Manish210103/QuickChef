import { Component, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';

@Component({
  selector: 'app-header',
  imports: [CommonModule],
  templateUrl: './header.component.html',
  styleUrl: './header.component.scss'
})
export class HeaderComponent {
  @Output() logoutClicked = new EventEmitter<void>();

  constructor(private router: Router) {}

  onSavedClick(): void {
    this.router.navigate(['/saved']);
  }

  onProfileClick(): void {
    this.router.navigate(['/profile']);
  }

  onLogoutClick(): void {
    this.logoutClicked.emit();
  }
}
