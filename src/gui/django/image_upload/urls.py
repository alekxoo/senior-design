from django.urls import include, path
# from image_upload import views
from rest_framework.routers import DefaultRouter
from .views import ImageViewset

router = DefaultRouter()
# r'uploads' refers to API endpoint URL path for interacting w ImageViewset
# basename is DRF-specific internal name for URL reversing and routing
router.register(r'uploads', ImageViewset, basename='upload')

urlpatterns = [
    # path("", views.home, name="home"),
    path("", include(router.urls)),
]
