import { Component, OnInit, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../../app/shared/services/api.service';
import { HistoryItem } from '../../../app/shared/models/recipe.models';
import { Router } from '@angular/router';

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './history.html',
  styleUrl: './history.scss'
})
export class History implements OnInit {
  @Input() history: HistoryItem[] = [];
  selectedItem: HistoryItem | null = null;

  // Toast messages
  success: string | null = null;
  error: string | null = null;

  // Modal state
  showDeleteModal: boolean = false;
  itemToDelete: HistoryItem | null = null;

  constructor(private apiService: ApiService, private router: Router) { }

  ngOnInit(): void {
    if (this.history.length === 0) {
      this.loadHistory();
    }
  }

  private loadHistory(): void {
    this.apiService.getHistory(10).subscribe({
      next: (data) => {
        this.history = data.history || [];
        if (this.history.length > 0) this.selectedItem = this.history[0];
      },
      error: (error) => {
        console.error('Error fetching history:', error);
        this.showToast('Failed to load history', 'error');
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
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  }

  goBackToDashboard(): void {
    this.router.navigate(['/dashboard']);
  }

  // --- Open confirmation modal ---
  confirmDelete(item: HistoryItem, event: MouseEvent): void {
    event.stopPropagation();
    this.itemToDelete = item;
    this.showDeleteModal = true;
  }

  // --- Actual delete after confirmation ---
  deleteConfirmed(): void {
    if (!this.itemToDelete) return;

    this.apiService.deleteHistoryItem(this.itemToDelete._id).subscribe({
      next: () => {
        this.history = this.history.filter(h => h._id !== this.itemToDelete!._id);
        if (this.selectedItem?._id === this.itemToDelete?._id) {
          this.selectedItem = this.history[0] || null;
        }
        this.showToast('Recipe deleted successfully!', 'success');
        this.closeModal();
      },
      error: (error) => {
        console.error('Error deleting recipe:', error);
        this.showToast('Failed to delete recipe', 'error');
        this.closeModal();
      }
    });
  }

  // --- Close modal without deleting ---
  closeModal(): void {
    this.showDeleteModal = false;
    this.itemToDelete = null;
  }

  // --- Toast function ---
  showToast(message: string, type: 'success' | 'error') {
    if (type === 'success') {
      this.success = message;
      this.error = null;
    } else {
      this.error = message;
      this.success = null;
    }

    setTimeout(() => {
      this.success = null;
      this.error = null;
    }, 3000);
  }
}
