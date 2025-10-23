from django.conf import settings
from ask_document.config import MEDIA_ROOT, MEDIA_URL
from django.urls import path
from . import controllers
from django.conf.urls.static import static

urlpatterns = [
    path('image_questioning/', controllers.image_questioning, name='image_questioning'),
    path('function_calling/', controllers.llm_choice, name='function_calling'),
    path('pdf_questioning/', controllers.pdf_questioning, name='pdf_questioning'),
]

if settings.DEBUG:
    urlpatterns += static(MEDIA_URL, document_root=MEDIA_ROOT)