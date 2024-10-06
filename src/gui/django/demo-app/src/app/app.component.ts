import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';
import { ImageUploadComponent } from './components/image-upload/image-upload.component';

@Component({
  selector: 'app-root',
  standalone: true,
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
  imports: [CommonModule, RouterOutlet, ImageUploadComponent],
})
export class AppComponent {
  title = 'demo-app';
}
