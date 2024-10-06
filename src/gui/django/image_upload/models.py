from django.db import models

# Create your models here.
class Image(models.Model):
    # upload_to defines storage location on filesys for images
    image = models.ImageField(upload_to='image_uploads/')
    # file = models.FileField(null=True, blank=True, validators=[])
    upload_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.image.name
