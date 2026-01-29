from models import efanet
import torch
from timm import create_model
from timm.data import create_dataset
import matplotlib.pyplot as plt
from tools import *
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

num_classes = 101
model = create_model('efanet_base_patch16_224', num_classes=num_classes)

state_dict = torch.load('efa_base.tar', map_location='cuda', weights_only=False)['state_dict']

model.load_state_dict(state_dict, strict=True)
model.to('cuda')
model.eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(224, 224), antialias=True),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
root = '../dataset/food101/val'

val_dataset = create_dataset(name='validation', root=root, transform=preprocess)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,  # 根据你的硬件情况调整
    shuffle=False,  # 验证集不需要打乱顺序
    num_workers=8,  # 根据你的CPU核心数和IO性能调整
    pin_memory=True # 如果使用GPU，设为True可以加速数据转移
)
steps = 100
total_samples = 0
# 用于累加所有曲线，初始化为零向量
sum_of_deletion_curves = torch.zeros(steps+1, device='cuda') 
sum_of_insertion_curves = torch.zeros(steps+1, device='cuda') 
# 用于累加所有 AUC 值，初始化为零
sum_of_deletion_aucs = 0.0
sum_of_insertion_aucs = 0.0
for images, labels in tqdm(val_loader):
    images = images.to('cuda')
    labels = labels.to('cuda')
    with torch.no_grad():
        logits = model(images)
        _, predicted_classes = logits.max(dim=1)
    
    saliency_maps = rollout_cam(model=model, image_tensor=images, target_classes=predicted_classes)
    deletion_curves = deletion_metric(model=model,
                                         image_tensor=images,
                                         saliency_maps=saliency_maps,
                                         target_classes=predicted_classes,
                                         device='cuda',
                                         steps=steps
                                        )
    insertion_curves = insertion_metric(model=model,
                                     image_tensor=images,
                                     saliency_maps=saliency_maps,
                                     target_classes=predicted_classes,
                                     device='cuda',
                                     steps=steps
                                    )
    deletion_aucs = calculate_auc(deletion_curves)
    insertion_aucs = calculate_auc(insertion_curves)
    
    sum_of_deletion_curves += deletion_curves.sum(dim=0)
    sum_of_deletion_aucs += deletion_aucs.sum().item()
    sum_of_insertion_curves += insertion_curves.sum(dim=0)
    sum_of_insertion_aucs += insertion_aucs.sum().item()
    total_samples += labels.size(0)
mean_deletion_curve = sum_of_deletion_curves / total_samples
mean_deletion_curve = mean_deletion_curve.detach().cpu().numpy()
mean_deletion_auc = sum_of_deletion_aucs / total_samples

mean_insertion_curve = sum_of_insertion_curves / total_samples
mean_insertion_curve = mean_insertion_curve.detach().cpu().numpy()
mean_insertion_auc = sum_of_insertion_aucs / total_samples

x_axis_percentage = np.linspace(0, 100, steps+1)

data_to_save = {
    'removed_pixels_percentage': x_axis_percentage,
    'deletion_confidence': mean_deletion_curve,
    'deletion_auc': mean_deletion_auc,
    'insertion_confidence': mean_insertion_curve,
    'insertion_auc': mean_insertion_auc,
}
df = pd.DataFrame(data_to_save)
df.to_csv('experiment/mean_curve_rollout_food101.csv', index=False, float_format='%.6f')
