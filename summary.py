#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile

from nets.yolo import YoloBody

if __name__ == "__main__":
    input_shape     = [640, 640]
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes     = 80
    phi             = 'l'
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m       = YoloBody(anchors_mask, num_classes, phi, False).to(device)
    for i in m.children():
        print(i)
        print('==============================')
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(m.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))

import time


num_iterations = 100  # 定义进行多少次推理
total_infer_time = 0.0  # 总推理时间

for _ in range(num_iterations):
    torch.cuda.synchronize()
    start = time.time()

    # 进行推理

    result = m(dummy_input)

    torch.cuda.synchronize()
    end = time.time()

    infer_time = end - start
    total_infer_time += infer_time

# 计算平均推理时间
avg_infer_time = total_infer_time / num_iterations

# 计算FPS
fps = 1 / avg_infer_time

print('Average FPS:', fps)



