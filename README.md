#### 部署需求：
安装django，opencv，numpy，matplotlib，PIL,mysqlclient,MySql-python 以及 安装mysql数据库
#### 部署方法：
1. 在mysql中建一个数据库，数据库名为myfont

2. 在mysite/mysite/settings.py中修改数据库的用户名密码，并将BASE PATH中, mysite前面的路径改成你实际存放位置（绝对路径）

3. 在mysite/myfont/views.py中：addCharacter()函数，把路径改成rar包里面save_1_3755.txt的绝对路径；download()函数，把路径改成myfontPRO\mysite\myfont\static\myfont\assets\fontTable\GB2312_1.pdf的绝对路径;

4. 在cmd中进入最外层的mysite文件夹，输入：

   ```shell
   python manage.py makemigrations myfont
   python manage.py migrate
   python manage.py runserver
   ```

5. 此时在浏览器中输入 [http://localhost:8000/]() 并进行注册登录 http://127.0.0.1:8000/

6. 登录之后，在浏览器中访问 [localhost:8000/addCharacter]() 导入字符数据 http://127.0.0.1:8000/addCharacter


#### 注

1. best_test.jpg是作为模拟用户上传的图片