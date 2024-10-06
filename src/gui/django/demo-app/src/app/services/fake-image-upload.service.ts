import { Injectable } from '@angular/core';
import { Observable, delay, of, timeout } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class FakeImageUploadService {

  uploadImage(image: File): Observable<string> {
    console.log(`we are uploading fake upload ${image.name}`);
    return of('https://random.imagecdn.app/500/150').pipe(delay(3000));
  }
}
