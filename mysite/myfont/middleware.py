from django.http import HttpResponseRedirect
from django.utils.deprecation import MiddlewareMixin
from .models import User


class AuthMiddleware(MiddlewareMixin):
    def process_request(self, request):
        # 统一验证登录
        # return none 或者 不写return才会继续往下执行, 不需要执行
        if request.path == '/login/' or request.path == '/register/' or request.path == '/reset/' or request.path == '/forget/':
            return None
        token = request.COOKIES.get('myfont_token')
        if not token:
            return HttpResponseRedirect('/login/')

        myfont_user = User.objects.filter(token=token)
        if not myfont_user:
            return HttpResponseRedirect('/login/')
        # 将user赋值在request请求的user上，以后可以直接判断user有没有存在
        # 备注，django自带的有user值
        request.myfont_user = {'user_id': myfont_user[0].id,
                            'user_email': myfont_user[0].email,
                            'user_name': myfont_user[0].name,
                            'user_regtime': myfont_user[0].regtime}