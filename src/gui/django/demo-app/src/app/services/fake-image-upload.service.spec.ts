import { TestBed } from '@angular/core/testing';

import { FakeImageUploadService } from './fake-image-upload.service';

describe('FakeImageUploadService', () => {
  let service: FakeImageUploadService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(FakeImageUploadService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
