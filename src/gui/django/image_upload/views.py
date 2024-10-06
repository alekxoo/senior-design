# from django.shortcuts import render

# Create your views here.

# from django.http import HttpResponse

# def home(request):
#     return HttpResponse("Upload images")

import boto3
from django.conf import settings
from rest_framework.viewsets import ModelViewSet
from rest_framework.exceptions import ValidationError
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status

from .models import Image
from .serializers import ImageSerializer

s3 = boto3.client('s3', aws_access_key_id=settings.AWS_ACCESS_KEY_ID, aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)

class ImageViewset(ModelViewSet):
    queryset = Image.objects.all()
    serializer_class = ImageSerializer
    parser_classes = [MultiPartParser]
    
    def create(self, request, *args, **kwargs):
        if "file" not in request.FILES:
            raise ValidationError("No file in HTTP body.")
        
        file = request.FILES['file']
        
        try:
            instance = Image.objects.create(image=file)
            instance.save()
            
            file_path = instance.image.path
            
            print(file_path)
            
            s3.upload_file(
                file_path,
                settings.AWS_STORAGE_BUCKET_NAME,
                instance.image.name
            )
            
            image_url = request.build_absolute_uri(instance.id)
            media_path = instance.image.url # media/image_uploads/...
            
            s3_url = f"https://{settings.AWS_S3_CUSTOM_DOMAIN}/{instance.image.name}"
            
            return Response({
                "detail": "File uploaded successfully!",
                "image_url": image_url,
                "media_path": media_path,
                "s3_url": s3_url
                }, status=status.HTTP_201_CREATED)
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    # @action(
    #     detail=True,
    #     methods=["POST"],
    #     parser_classes=[MultiPartParser],
    #     url_path=r"upload/(?P<filename>[a-zA-Z0-9_]+\.(jpg|jpeg|png|gif))", # allows multiple image types
    # ) 
    # def upload(self, request, **kwargs):
    #     print("Received a request to upload an image.")
        
    #     if "file" not in request.FILES:
    #         raise ValidationError("There is no file in the HTTP body.")
        
        
    #     # file = request.FILES["file"]
    #     file = request.data.get('file')
        
    #     if not file:
    #         print("No file found in the request data.")
    #         return Response({"detail": "No file found."}, status=status.HTTP_400_BAD_REQUEST)
        
    #     try:
    #         instance = self.get_object()
    #         print(f"Uploading file: {file.name}")
    #     except Image.DoesNotExist:
    #         instance = Image.objects.create()
        
    #     try:
    #         instance.image.save(file.name, file)
    #         instance.save()
    #         # return Response(ImageSerializer(image).FILES)
            
    #         image_url = request.build_absolute_uri(instance.image.url)
    #         media_path = instance.image.url # media/image_uploads/...
            
    #         return Response({
    #             "detail": "File uploaded successfully!",
    #             "image_url": image_url,
    #             "media_path": media_path
    #             }, status=status.HTTP_200_OK)
    #     except Exception as e:
    #         print(f"Error occurred: {str(e)}")
    #         return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        
        


