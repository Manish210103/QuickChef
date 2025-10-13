import { Routes } from '@angular/router';
import { Login } from './pages/login/login';
import { Signup } from './pages/signup/signup';
import { DashboardComponent } from './pages/dashboard/dashboard.component';
import { ProfileComponent } from './pages/profile/profile.component';
import { History } from './pages/history/history';

export const routes: Routes = [
    { path: 'login', component: Login },
    { path: 'signup', component: Signup },
    { path: 'dashboard', component: DashboardComponent },
    { path: 'saved', component: History },
    { path: 'profile', component: ProfileComponent },
    { path: '', redirectTo: '/login', pathMatch: 'full' }
];
