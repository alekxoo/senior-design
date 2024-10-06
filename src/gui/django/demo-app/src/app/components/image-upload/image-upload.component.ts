import { Component, inject } from '@angular/core';
import { FakeImageUploadService } from './../../services/fake-image-upload.service';

@Component({
  selector: 'app-image-upload',
  standalone: true,
  imports: [],
  templateUrl: './image-upload.component.html',
  styleUrl: './image-upload.component.css'
})
export class ImageUploadComponent {
  fakeImageUploadService = inject(FakeImageUploadService);

  selectedImages!: FileList;
  onImageSelected(event: Event): void {
    const inputElement = event.target as HTMLInputElement;
    if (inputElement?.files && inputElement.files.length > 0) {
      this.selectedImages = inputElement.files;
    }
  }

  upload(): void {
    if (this.selectedImages) {
      this.uploadFiles(this.selectedImages);
    }
  }

  private uploadFiles(images: FileList): void {
    for (let index = 0; index < images.length; index++) {
      const element = images[index];
      this.fakeImageUploadService.uploadImage(element).subscribe((p) => {
		      console.log(p)
      });
    }
  }
}
