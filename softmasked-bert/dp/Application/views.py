
from django.shortcuts import render
import test

def index(request):
    return render(request, 'index.html')


def use(request):
    text = test.Generator(request.POST.get("ori"))
    request.session['msg'] = text
    return render(request, 'index.html')
