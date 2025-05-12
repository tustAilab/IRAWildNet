#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile

from nets.yolo import YoloBody
import time

if __name__ == "__main__":
    input_shape     = [640, 640]
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes     = 4
    phi             = 'n'
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m       = YoloBody(input_shape, num_classes, phi, False).to(device)
    for i in m.children():
        print(i)
        print            ('==============================')
    
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
    # Measure FPS
    times = []
    for _ in range(1000):  # Measure 10 times to get an average
        start_time = time.time()
        with torch.no_grad():
            m(dummy_input)
        end_time = time.time()
        times.append(end_time - start_time)

    # Calculate the average time per frame
    average_time = sum(times) / len(times)

    # Calculate FPS
    fps = 1.0 / average_time

    print(f'Average time per frame: {average_time:.3f} seconds')
    print(f'FPS: {fps:.2f}')
