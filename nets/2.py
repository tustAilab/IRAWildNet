import torch

# 加载权重文件
weights_path ='C:\experiment_project\infrared_small_traget\\test\\75\weight\\76.pth'
checkpoint = torch.load(weights_path, map_location='cpu')

# 打印权重文件中的键
print("Keys in the checkpoint:")
print(checkpoint.keys())