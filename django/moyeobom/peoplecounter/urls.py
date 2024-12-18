from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("count_view/", views.count_view, name="count_view"),
    path("area_view/", views.area_view, name="area_view"),
    path("video_feed/", views.video_feed, name="video_feed"),
    path("area_feed/", views.area_feed, name="area_feed"),
    path("get_count_data/", views.get_count_data, name="get_count_data"),
    path("get_ratio_data/", views.get_ratio_data, name="get_ratio_data"),
]