from django.conf import settings
from django.urls import path
from . import views
from django.conf.urls.static import static

urlpatterns = [
    path('image_questioning/', views.image_questioning, name='image_questioning'),
    path('function_calling/', views.llm_choice, name='function_calling'),
    path('pdf_questioning/', views.pdf_questioning, name='pdf_questioning'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)