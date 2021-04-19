import cv2
from django.shortcuts import render, get_object_or_404, get_list_or_404
from django.http import HttpResponse, Http404, HttpResponseRedirect, StreamingHttpResponse
from django.template import loader
from django.contrib.auth.hashers import make_password, check_password
from django.conf import settings
from django.core.paginator import Paginator
from .models import User, Character, User_Char
from .myimgutil import MyImgUtil, read_images, OneImgProcess, catImg

import os
import time
import random
import numpy as np

###############################################################
def login(request):
    context = {}
    if request.method == 'GET':
        return render(request, 'myfont/login.html/', context)
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        if User.objects.filter(email=email).exists():
            user = User.objects.get(email=email)
            if check_password(password, user.password):
                token = ''
                for i in range(20):
                    s = '1234567890abcdefghijklmnopqrstuvwxyz'
                    token += random.choice(s)
                now_time = int(time.time()) # 共10位
                token = 'TK' + token + str(now_time)
                response = HttpResponseRedirect('/index/')
                response.set_cookie('myfont_token', token, max_age=10000)
                user.token = token
                user.save()
                return response
            else:
                context['error_msg'] = '邮箱或密码错误'
                return render(request, 'myfont/login.html/', context)
        else:
            context['error_msg'] = '邮箱不存在'
            return render(request, 'myfont/login.html/', context)


def logout(request):
    context = {}
    if request.method == 'GET':
        response = HttpResponseRedirect('/login/')
        response.delete_cookie('myfont_token')
        return response
    return render(request, 'myfont/index.html/', context)


def register(request):
    context = {}
    if request.method == 'GET':
        return render(request, 'myfont/register.html/', context)
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        password = make_password(password)
        User.objects.create(name=name, email=email, password=password)
        return HttpResponseRedirect('/login/')


def forget(request):
    context = {}
    if request.method == 'GET':
        return render(request, 'myfont/forget.html/', context)
    if request.method == 'POST':
        email = request.POST.get('email')
        if User.objects.filter(email=email).exists():
            #TODO:改成发邮件，然后链接携带参数，跳转到reset.html
            context.update(email=email)
            return render(request, 'myfont/reset.html/', context)
        else:
            context['error_msg'] = '邮箱不存在'
            return render(request, 'myfont/forget.html/', context)


def reset(request):
    context = {}
    if request.method == 'GET':
        #TODO:配套邮件功能
        return render(request, 'myfont/reset.html/', context)
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        if User.objects.filter(email=email).exists():
            user = User.objects.get(email=email)
            user.password = make_password(password)
            user.save()
            return HttpResponseRedirect('/login/')
        else:
            raise Http404('Email does not exist')

###############################################################
def index(request):
    context = {}
    if request.method == 'GET':
        return render(request, 'myfont/index.html/', context)


def userinfo(request):
    context = {}
    if request.method == 'GET':
        return render(request, 'myfont/userinfo.html/', context)


def download(request, level):
    def down_chunk_file_manager(file_path, chuck_size=1024):
        with open(file_path, "rb") as file:
            while True:
                chuck_stream = file.read(chuck_size)
                if chuck_stream:
                    yield chuck_stream
                else:
                    break
    file_path = 'D:\\beifen\\mysite\\myfont\\static\\myfont\\assets\\fontTable\\GB2312_1.pdf'
    file_name = 'GB2312_1.pdf'
    response = StreamingHttpResponse(down_chunk_file_manager(file_path))
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename="{0}"'.format(file_name)
    return response


def moduserinfo(request):
    context = {}
    if request.method == 'GET':
        return render(request, 'myfont/moduserinfo.html/', context)
    if request.method == 'POST':
        name = request.POST.get('name')
        password = request.POST.get('password')
        email = request.myfont_user['user_email']
        user = User.objects.get(email=email)
        user.name = name
        user.password = make_password(password)
        user.save()
        return HttpResponseRedirect('/userinfo/')


def activity(request):
    context = {}
    if request.method == 'GET':
        return render(request, 'myfont/activity.html/', context)
    if request.method == 'POST':
        index = request.POST.get('index')
        article = request.POST.get('article')
        context.update(data=article)
        user_id = request.myfont_user['user_id']
        basepath = settings.BASE_PATH
        userpath = basepath + '\\' + str(user_id)
        userprocesspath = userpath + '\\process'
        img_save_path = userpath + '\\result.jpg'
        img_path_list = []
        # user = User.objects.get(id=user_id)
        # for word in article:
        #     if not Character.objects.filter(word=word):
        #         raise Http404('Word does not exist: ' + word)
        #     character = Character.objects.get(word=word)
        #     if not User_Char.objects.filter(user=user, character=character, index=int(index)):
        #         raise Http404('Cannot find this character in font index: ' + index)
        #     user_char = User_Char.objects.get(user=user, character=character, index=int(index))
        #     generate_img(user_char.turnpoint, user_char.ctrlpoint)
        for word in article:
            if not Character.objects.filter(word=word):
                raise Http404('Word does not exist: ' + word)
            character = Character.objects.get(word=word)
            img_path_list.append(userprocesspath + '\\' + index + '\\' + str(character.index) + '_' + str(character.page) + '_' + str(character.row) + '_' + str(character.col) + '.jpg')
        catImg(img_path_list=img_path_list, img_row=1, img_col=len(article), img_save_path=img_save_path)
        result_img_url = 'upload/' + str(user_id) + '/result.jpg'
        context.update(result_img_url=result_img_url)
        return render(request, 'myfont/activity.html/', context)


def createfont(request):
    context = {}
    if request.method == 'GET':
        return render(request, 'myfont/createfont.html/', context)
    if request.method == 'POST':
        user_id = request.myfont_user['user_id']
        # 获取上传的文件以及选择字体库的编号，如果没有文件，则默认为None
        index = request.POST.get('index')
        inputfiles = request.FILES.getlist('inputfiles', None)
        if not inputfiles: 
            raise Http404('no files or directory for upload!')
        basepath = settings.BASE_PATH
        userpath = basepath + '\\' + str(user_id)
        useruploadpath = userpath + '\\upload'
        userprocesspath = userpath + '\\process'
        useroutlinepath = userpath + '\\outline'
        if not os.path.exists(userpath):
            os.makedirs(userpath)
            os.makedirs(useruploadpath)
            os.makedirs(userprocesspath)
            os.makedirs(useroutlinepath)
            for i in range(1, 4):
                os.makedirs(useruploadpath + '\\' + str(i))
                os.makedirs(userprocesspath + '\\' + str(i))
                os.makedirs(useroutlinepath + '\\' + str(i))
        for inputfile in inputfiles:    # 依次处理各个文件
            destination = open(os.path.join(useruploadpath + '\\' + str(index), inputfile.name), 'wb+')
            for chunk in inputfile.chunks():    # 分块写入单个文件
                destination.write(chunk)
            destination.close()
        if imgpreprocess(useruploadpath, userprocesspath, index):
            img2outline(userprocesspath, useroutlinepath, index)
            outline2bezier(useroutlinepath, index, user_id)
            context.update(msg='上传并创建个性字体库成功')
        else:
            context.update(msg='创建字体失败，请上传符合要求的图片')
        return render(request, 'myfont/createfont.html/', context)


def myfont(request):
    context = {}
    if request.method == 'GET':
        return render(request, 'myfont/myfont.html/', context)


def showfont(request, index):
    context = {}
    if request.method == 'GET':
        user_id = request.myfont_user['user_id']
        page = request.GET.get('page', 1)
        page = int(page)
        path = settings.BASE_PATH + '\\' + str(user_id) + '\\process\\' + str(index)
        img_names = read_images(path)
        if len(img_names) == 0:
            return HttpResponseRedirect('/myfont/')
        start = (page - 1) * 20
        end = page * 20 if page * 20 < len(img_names) else len(img_names)
        img_paths = []
        #一页20个图片，分成4个一行
        row = 0
        cnt = 0
        for i in range(start, end):
            if cnt == 0:
                img_paths.append([])
            img_paths[row].append('upload/' + str(user_id) + '/process/' + str(index) + '/' + img_names[i])
            cnt = cnt + 1
            if cnt == 4:
                row = row + 1
                cnt = 0
        context.update(img_paths=img_paths)

        p = Paginator(img_names, 20)
        left = []
        right = []
        left_has_more = False
        right_has_more = False
        first = False
        last = False
        total_pages = p.num_pages
        page_range = p.page_range
        if page == 1 and page == total_pages:
            pass
        elif page == 1:
            right = page_range[page:page+2]
            if right[-1] < total_pages - 1:
                right_has_more = True
            if right[-1] < total_pages:
                last = True
        elif page == total_pages:
            left = page_range[(page-3) if (page-3) > 0 else 0:page-1]
            if left[0] > 2:
                left_has_more = True
            if left[0] > 1:
                first = True
        else:
            left = page_range[(page-3) if (page-3) > 0 else 0:page-1]
            right = page_range[page:page+2]
            if left[0] > 2:
                left_has_more = True
            if left[0] > 1:
                first = True
            if right[-1] < total_pages - 1:
                right_has_more = True
            if right[-1] < total_pages:
                last = True
        data = {
            'left':left,
            'right':right,
            'left_has_more':left_has_more,
            'right_has_more':right_has_more,
            'first':first,
            'last':last,
            'total_pages':total_pages,
            'page':page
        }
        context.update(data=data)
        print(img_paths)
        return render(request, 'myfont/showfont.html/', context)

###############################################################
def imgpreprocess(useruploadpath, userprocesspath, index):
    '''
    参数的一个样例为：useruploadpath —————— 'D:\\user_id\\upload'
                    userprocesspath —————— 'D:\\user_id\\process'
    index —— 1 or 2 or 3
    该参数表示用户选择的字体库序号，一个用户最多可以拥有3套属于自己的字库
    该函数将useruploadpath\\{{index}}文件夹中用户上传的文字图片进行切分以及预处理后，
    会把单字图片存储在userprocesspath\\{{index}}文件夹中
    单字图像的命名规则为：字名_页码_行号_列号
    返回值为True表示处理成功，False表示处理失败(也可以无返回值)
    '''
    readpath = useruploadpath + '\\' + str(index)
    writepath = userprocesspath + '\\' + str(index)
    # 读取readpath中的各个图片，并进行处理
    img_names = read_images(readpath)
    for img_name in img_names:
        img_read_path = readpath + '\\' + img_name #该变量就是要读取的图片的完整路径 'D:\\user_id\\upload\\1.jpg'
        # 进行后续处理
        OneImgProcess(img_read_path, writepath)
    # 将处理好的单字图片写入到writepath中
    return True


def img2outline(userprocesspath, useroutlinepath, index):
    '''
    把预处理好的文字图片提取出轮廓，以同样的名字写入到outline文件夹中
    '''
    readpath = userprocesspath + '\\' + str(index)
    writepath = useroutlinepath + '\\' + str(index)
    img_names = read_images(readpath)
    for img_name in img_names:
        img_read_path = readpath + '\\' + img_name
        img_write_path = writepath + '\\' + img_name
        img = MyImgUtil(img_read_path)
        img_outline = img.get_outline()
        cv2.imwrite(img_write_path, img_outline)


def outline2bezier(useroutlinepath, index, user_id):
    '''
    对这些图片依次读取后，提取控制点以及拐点，并存储到数据库中
    如果数据库中没有存储的字，则先创建这个字
    如果用户已经在该index对应的字库中上传过相同的字，则更新这个字的拐点与控制点
    '''
    path = useroutlinepath + '\\' + str(index)
    img_names = read_images(path)
    cnt = 0
    for img_name in img_names:
        word = img_name.split('_')[0]
        page = img_name.split('_')[1]
        row = img_name.split('_')[2]
        col = img_name.split('_')[3].split('.')[0]
        img_path = path + '\\' + img_name
        judge = MyImgUtil(img_path).getResult()
        if judge == False:
            print('find error!')
            continue
        cnt = cnt + 1
        turnpoint, ctrlpoint = judge
        if not Character.objects.filter(word=word).exists():
            Character.objects.create(word=word, page=page, row=row, col=col)
        user = User.objects.get(id=user_id)
        character = Character.objects.get(word=word)
        if not User_Char.objects.filter(user=user, character=character, index=index):
            User_Char.objects.create(user=user, character=character, index=index, turnpoint=turnpoint, ctrlpoint=ctrlpoint)
        else:
            user_char = User_Char.objects.get(user=user, character=character, index=index)
            user_char.turnpoint = turnpoint
            user_char.ctrlpoint = ctrlpoint
            user_char.save()
    print('成功的图片个数：', cnt)



def addCharacter(request):
    with open(r'D:\\beifen\\save_1_3755.txt', 'r') as f:
        str = f.read()
        for i, word in enumerate(str):
            index = i + 1
            page = i // 100 + 1
            col = i % 10 + 1
            row = i % 100 // 10 + 1
            print(word, page, row, col, index)
            Character.objects.create(word=word, page=page, row=row, col=col, index=index)
