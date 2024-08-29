from __future__ import print_function

from django.shortcuts import render, redirect, HttpResponse
from app01 import models
from django import forms
from app01.utils.pagination import Pagination
import torch
from PIL import Image
from get_adv import get_adv
from django.core.validators import RegexValidator


# Create your views here.
def register(request):
    if request.method == "GET":
        form = UserModelForm()
        return render(request, 'register.html', {"form": form})

    form = UserModelForm(data=request.POST)
    if form.is_valid():
        form.save()
        return redirect("/model/list/")

    return render(request, 'register.html', {"form": form})


def login(request):
    if request.method == "GET":
        return render(request, 'login.html')

    email = request.POST.get("email")
    password = request.POST.get("password")
    exists = models.UserInfo.objects.filter(email=email).exists()
    if not exists:
        e_error = "该用户不存在"
        return render(request, 'login.html', {"e_error": e_error})
    user = models.UserInfo.objects.filter(email=email).first()
    if password != user.password:
        p_error = "输入密码不正确"
        return render(request, 'login.html', {"p_error": p_error})
    return redirect("/model/list/")


def model_list(request):
    """模型列表"""
    queryset = models.AImodel.objects.all()
    page_object = Pagination(request, queryset, page_size=5)
    content = {
        "queryset": page_object.page_queryset,
        "page_string": page_object.html(),
    }
    return render(request, 'model_list.html', content)


def model_add(request):
    """添加模型"""
    if request.method == "GET":
        return render(request, 'model_add.html')

    my_file = request.FILES.get("pt", None)
    if not my_file:
        error = "上传失败！"
        return render(request, 'model_add.html', {"error": error})
    title = request.POST.get("title")
    destination = open(f"./app01/static/model/{title}.pt", "wb+")
    for chunk in my_file.chunks():
        destination.write(chunk)
    destination.close()
    models.AImodel.objects.create(title=title)
    return redirect("/model/list/")


def model_delete(request):
    """删除模型"""
    nid = request.GET.get('nid')
    models.AImodel.objects.filter(id=nid).delete()
    return redirect("/model/list/")


def model_use(request, nid):
    if request.method == "GET":
        return render(request, 'model_use.html')

    my_file = request.FILES.get("pic", None)
    if not my_file:
        error = "上传失败！"
        return render(request, 'model_use.html', {"error": error})
    title = request.POST.get("title")
    destination = open(f"./app01/static/img/{title}.jpg", "wb+")
    for chunk in my_file.chunks():
        destination.write(chunk)
    destination.close()
    return redirect(request.path + "use")


# 加载文件上传表单
def upload(request):
    return render(request, "upload.html")


# 执行文件上传处理
def img_upload(request):
    my_file = request.FILES.get("pic", None)
    if not my_file:
        return HttpResponse("没有上传的文件信息")
    destination = open("./app01/static/img/a.jpg", "wb+")
    for chunk in my_file.chunks():
        destination.write(chunk)
    destination.close()
    return HttpResponse("上传的文件:")


def use(request, nid):
    print(request.path)
    pretrained_model = "./app01/static/model/FGSM.pt"
    model = torch.load(pretrained_model, map_location="cpu")

    # 设置为验证模式.
    model.eval()
    image_path = "./app01/static/img/a.jpg"
    image = Image.open(image_path).convert("RGB")  # 加载图片并转换为RGB格式
    adv_image, sss = get_adv(model, image)
    adv_image.save("./app01/static/adv_image.jpg")
    request.session['msg']=sss
    #adv_image.show()
    return redirect("/model/" + str(nid) + "/use/")
    
    


def user_list(request):
    queryset = models.UserInfo.objects.all()
    page_object = Pagination(request, queryset, page_size=5)
    content = {
        "queryset": page_object.page_queryset,
        "page_string": page_object.html(),
    }
    return render(request, 'user_list.html', content)


class UserModelForm(forms.ModelForm):
    confirm_password = forms.CharField(
        label="确认密码",
        widget=forms.PasswordInput
    )

    class Meta:
        model = models.UserInfo
        fields = ["name", "email", "password", "confirm_password"]
        widgets = {
            "password": forms.PasswordInput,
            "confirm_password": forms.PasswordInput
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            field.widget.attrs = {"class": "form-control", "placeholder": field.label}


def user_add(request):
    if request.method == "GET":
        form = UserModelForm()
        return render(request, 'user_add.html', {"form": form})

    form = UserModelForm(data=request.POST)
    if form.is_valid():
        form.save()
        return redirect("/user/list/")

    return render(request, 'user_add.html', {"form": form})


def user_edit(request, nid):
    """编辑用户"""
    row_object = models.UserInfo.objects.filter(id=nid).first()
    if request.method == "GET":
        # 根据ID区数据库获取要编辑的那一行数据
        form = UserModelForm(instance=row_object)
        return render(request, 'usr_edit.html', {'form': form})

    form = UserModelForm(data=request.POST, instance=row_object)
    if form.is_valid():
        form.save()
        return redirect('/user/list/')
    return render(request, 'user_add.html', {"form": form})


def user_delete(request, nid):
    models.UserInfo.objects.filter(id=nid).delete()
    return redirect("/user/list/")


def pnum_list(request):
    data_dict = {}
    search_data = request.GET.get('q', "")
    if search_data:
        data_dict["mobile__contains"] = search_data
    queryset = models.Pnum.objects.filter(**data_dict).order_by("-level")
    page_object = Pagination(request, queryset)
    context = {
        "search_data": search_data,
        "queryset": page_object.page_queryset,
        "page_string": page_object.html()
    }
    return render(request, 'pnum_list.html', context)


class PrettyModelForm(forms.ModelForm):
    class Meta:
        model = models.Pnum
        fields = "__all__"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            field.widget.attrs = {"class": "form-control", "placeholder": field.label}

    # 验证：方式2
    def clean_mobile(self):
        txt_mobile = self.cleaned_data["mobile"]
        exists = models.Pnum.objects.filter(mobile=txt_mobile).exists()
        if exists:
            raise forms.ValidationError("手机号已存在")
        if len(txt_mobile) != 11:
            raise forms.ValidationError("格式错误")
        return txt_mobile


def pnum_add(request):
    if request.method == "GET":
        form = PrettyModelForm()
        return render(request, 'pnum_add.html', {"form": form})

    form = PrettyModelForm(data=request.POST)
    if form.is_valid():
        form.save()
        return redirect("/pnum/list/")

    return render(request, 'pnum_add.html', {"form": form})


class PrettyEditModelForm(forms.ModelForm):
    # mobile = forms.CharField(disabled=True, label="手机号")
    class Meta:
        model = models.Pnum
        fields = ['mobile', 'price', 'level', 'status']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            field.widget.attrs = {"class": "form-control", "placeholder": field.label}

    def clean_mobile(self):
        txt_mobile = self.cleaned_data["mobile"]
        exists = models.Pnum.objects.exclude(id=self.instance.pk).filter(mobile=txt_mobile).exists()
        if exists:
            raise forms.ValidationError("手机号已存在")
        if len(txt_mobile) != 11:
            raise forms.ValidationError("格式错误")
        return txt_mobile


def pnum_edit(request, nid):
    row_object = models.Pnum.objects.filter(id=nid).first()
    if request.method == "GET":
        form = PrettyEditModelForm(instance=row_object)
        return render(request, 'pnum_edit.html', {"form": form})

    form = PrettyEditModelForm(data=request.POST, instance=row_object)
    if form.is_valid():
        form.save()
        return redirect('/pnum/list/')

    return render(request, 'pnum_edit.html', {"form": form})


def pnum_delete(request, nid):
    models.Pnum.objects.filter(id=nid).delete()
    return redirect("/pnum/list/")
