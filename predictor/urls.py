from django.urls import path
from .views import predict_api,home
from . import views

urlpatterns = [
    path('',            views.login_view,   name='root_login'),
    path('home/',       views.home,         name='home'),
    path('predict/',    views.predict_api,  name='predict'),
    path('login/',      views.login_view,   name='login'),
    path('register/',   views.register_view,name='register'),
    path('logout/',     views.logout_view,  name='logout'),
    path('dashboard/',  views.dashboard,    name='dashboard'),
    path('upload-csv/', views.upload_csv,   name='upload_csv'),
    path('api/history/',views.scan_history, name='scan_history'),
    path('scan/<int:scan_id>/delete/', views.delete_scan, name='delete_scan'),
    path('scan/<int:scan_id>/detail/', views.scan_detail, name='scan_detail'),
]