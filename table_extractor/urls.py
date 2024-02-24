from django.urls import path, include
from table_extractor.views import TableExtractor, download_csv

urlpatterns = [
    path("extract/",TableExtractor.as_view(), name='table/extract'),
    path("download/",download_csv, name='table/download'),
]
