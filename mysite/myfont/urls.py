from django.urls import path

from . import views

app_name = 'myfont'
urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    path('register/', views.register, name='register'),
    path('forget/', views.forget, name='forget'),
    path('reset/', views.reset, name='reset'),

    path('index/', views.index, name='index'),
    path('userinfo/', views.userinfo, name='userinfo'),
    path('moduserinfo/', views.moduserinfo, name='moduserinfo'),
    path('activity/', views.activity, name='activity'),
    path('createfont/', views.createfont, name='createfont'),
    path('myfont/', views.myfont, name='myfont'),
    path('showfont/<int:index>/', views.showfont, name='showfont'),
    path('download/<str:level>/', views.download, name='download'),
    
    # 测试用
    path('addCharacter/', views.addCharacter, name='addCharacter'),
]