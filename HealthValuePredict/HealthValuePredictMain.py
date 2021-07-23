from HealthValuePredict.BridgeCheck import *
from django.shortcuts import HttpResponse

# 输出桥梁健康的判断值
def HealthValuePredictMain(request):
    # 输出桥梁健康类别,1健康，2亚健康，3不健康
    state = classOut()

    return HttpResponse(state)