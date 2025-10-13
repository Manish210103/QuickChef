import { Component, OnInit, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../../app/shared/services/api.service';
import { HistoryItem } from '../../../app/shared/models/recipe.models';
import { Router } from '@angular/router';

@Component({
  selector: 'app-history',
  imports: [CommonModule],
  templateUrl: './history.html',
  styleUrl: './history.scss'
})
export class History implements OnInit {
  @Input() history: HistoryItem[] = [];
  selectedItem: HistoryItem | null = null;

  constructor(private apiService: ApiService, private router: Router) {}

  ngOnInit(): void {
    if (this.history.length === 0) {
      this.loadHistory();
    }
  }

  private loadHistory(): void {
    this.apiService.getHistory(10).subscribe({
      next: (data) => {
        this.history = data.history || [];
        // Auto-select first item if available
        if (this.history.length > 0) {
          this.selectedItem = this.history[0];
        }
      },
      error: (error) => {
        console.error('Error fetching history:', error);
      }
    });
  }

  selectItem(item: HistoryItem): void {
    this.selectedItem = item;
  }

  formatDate(dateString: string): string {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - date.getTime());
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;

    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric', 
      year: 'numeric' 
    });
  }

  goBackToDashboard(): void {
    this.router.navigate(['/dashboard']);
  }

  deleteHistoryItem(item: HistoryItem, event: MouseEvent): void {
    event.stopPropagation(); // prevent triggering selectItem()
    if (!confirm(`Delete recipe "${item.recipe_name}"?`)) return;

    this.apiService.deleteHistoryItem(item._id).subscribe({
      next: () => {
        this.history = this.history.filter(h => h._id !== item._id);
        if (this.selectedItem && this.selectedItem._id === item._id) {
          this.selectedItem = this.history.length > 0 ? this.history[0] : null;
        }
      },
      error: (error) => {
        console.error('Error deleting recipe:', error);
      }
    });
  }

}