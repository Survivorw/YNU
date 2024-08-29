"""
URL configuration for employee project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from app01 import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.contrib import staticfiles
from django.contrib.staticfiles.urls import static
from . import settings


urlpatterns = [
    path('', views.login),
    path('register/', views.register),
    # 部门管理
    path('model/list/', views.model_list),
    path('model/add/', views.model_add),
    path('model/delete/', views.model_delete),
    path('model/<int:nid>/use/', views.model_use),
    # 上传文件
    path('upload/', views.upload),
    path('img/upload/', views.img_upload),
    # 用户管理
    path('user/list/', views.user_list),
    path('user/add/', views.user_add),
    path('user/<int:nid>/edit/', views.user_edit),
    path('user/<int:nid>/delete', views.user_delete),
    # 调用模型
    path('use', views.use),
    # 对抗管理
    path('pnum/list/', views.pnum_list),
    path('pnum/add/', views.pnum_add),
    path('pnum/<int:nid>/edit', views.pnum_edit),
    path('pnum/<int:nid>/delete', views.pnum_delete),
    path('model/<int:nid>/use/use/', views.use)


]
