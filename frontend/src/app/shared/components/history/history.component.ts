import { Component, OnInit, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { HistoryItem } from '../../models/recipe.models';

@Component({
  selector: 'app-history',
  imports: [CommonModule],
  templateUrl: './history.component.html',
  styleUrl: './history.component.scss'
})
export class HistoryComponent implements OnInit {
  @Input() history: HistoryItem[] = [];

  constructor(private apiService: ApiService) {}

  ngOnInit(): void {
    if (this.history.length === 0) {
      this.loadHistory();
    }
  }

  private loadHistory(): void {
    this.apiService.getHistory(10).subscribe({
      next: (data) => {
        this.history = data.history || [];
      },
      error: (error) => {
        console.error('Error fetching history:', error);
      }
    });
  }

  formatDate(dateString: string): string {
    return new Date(dateString).toLocaleDateString();
  }
}
