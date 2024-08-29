from django.db import models


# Create your models here.
class UserInfo(models.Model):
    """用户表"""
    name = models.CharField(verbose_name="姓名", max_length=32)
    password = models.TextField(verbose_name="密码", max_length=64)
    email = models.EmailField(verbose_name="邮箱", null=True, blank=True)
    # age = models.IntegerField(verbose_name="年龄")
    # gender_choices = (
    #     (1, "男"),
    #     (2, "女"),
    # )
    # gender = models.SmallIntegerField(verbose_name="性别", choices=gender_choices)
    # 级联删除
    # model = models.ForeignKey(verbose_name="部门", to="app01.AImodel", to_field="id", on_delete=models.CASCADE)
    # 删除置空
    # depart = models.ForeignKey(to="Department", to_field="id", null=True, blank=True, on_delete=models.SET_NULL)


class AImodel(models.Model):
    """模型表"""
    title = models.CharField(verbose_name='标题', max_length=32)
    def __str__(self):
        return self.title

class Pnum(models.Model):
    """训练历史表"""
    mobile = models.CharField(verbose_name="号码", max_length=11)
    price = models.IntegerField(verbose_name="价格")
    level_choices = (
        (1, "1级"),
        (2, "2级"),
        (3, "3级"),
        (4, "4级"),
    )
    level = models.SmallIntegerField(verbose_name="级别", choices=level_choices, default=1)
    status_choices = (
        (1, "已占用"),
        (2, "未占用"),
    )
    status = models.SmallIntegerField(verbose_name="状态", choices=status_choices, default=2)
