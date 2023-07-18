import time
import torch
# 创建一个输入张量
from nets.yolo import YoloBody
input_shape = [640, 640]
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
num_classes = 1
phi = 'l'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
m = YoloBody(anchors_mask, num_classes, phi, False).to(device)
# 进行多次推理，并计算平均推理时间
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