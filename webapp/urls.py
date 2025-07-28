from django.urls import path
from . import views

urlpatterns = [
	path('register/', views.registerPage, name="register"),
	path('login/', views.loginPage, name="login"),
	path('logout/', views.logoutUser, name="logout"),
	path('', views.home, name="home"),
	path("predict",views.predictImage,name='predictImage'),
	path("predict1",views.predictImage1,name='predictImage1'),
	path("home2", views.home2, name='home2'),
	path("home3", views.home3, name='home3'),
	path('live-detection/', views.live_detection_page, name='live_detection_page'),
	path('live-detection/stream/', views.live_detection, name='live_detection'),
]
