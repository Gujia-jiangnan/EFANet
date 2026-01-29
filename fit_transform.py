import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from typing import Union, Optional
import random
from torch.utils.data import Dataset, Subset
def create_subset(dataset: Dataset, 
                  target_class: Union[str, int], 
                  num_samples: Optional[int] = None, 
                  random_seed: int = 42) -> Subset:
    """
    从数据集中提取子集。可以按指定类别提取，也可以从所有类别中随机提取。

    Args:
        dataset (Dataset): 通过 timm.create_dataset 或类似方式创建的数据集。
                           必须包含 'class_to_idx' 和 'targets' 属性。
        target_class (Union[str, int]): 
            - 字符串 (str): 按类别名称提取 (如 'cat')。
            - 整数 (int): 按类别索引提取 (如 0)。
            - -1: 一个特殊值，表示从所有类别中进行抽样。
        num_samples (Optional[int]): 要抽取的样本数量。
                                     如果为 None，则返回该类别/所有类别的全部样本。
                                     默认为 None。
        random_seed (int): 随机种子，用于保证抽样的可复现性。

    Returns:
        Subset: 一个 torch.utils.data.Subset 对象。
    
    Raises:
        AttributeError: 如果数据集没有 'class_to_idx' 或 'targets' 属性。
        ValueError: 如果指定的类别名称或索引无效。
        TypeError: 如果 'target_class' 类型不受支持。
    """
    random.seed(random_seed)
    targets = [item[1] for item in dataset.reader.samples]
    # --- 新增逻辑：处理 target_class == -1 的情况 ---
    if target_class == -1:
        print("目标类别为 -1，将从所有类别中进行抽样。")
        all_indices = list(range(len(dataset)))
        total_samples = len(all_indices)

        if num_samples is None:
            final_indices = all_indices
            print(f"返回数据集中全部 {total_samples} 个样本。")
        else:
            if num_samples > total_samples:
                print(f"警告: 请求抽取 {num_samples} 个样本，但整个数据集只有 {total_samples} 个。将返回所有样本。")
                final_indices = all_indices
            else:
                final_indices = random.sample(all_indices, num_samples)
                print(f"已从整个数据集中随机抽取 {len(final_indices)} 个样本。")
        
        return Subset(dataset, final_indices)
        
    class_to_idx = dataset.reader.class_to_idx
    num_classes = len(class_to_idx)
    
    if isinstance(target_class, str):
        if target_class not in class_to_idx:
            raise ValueError(f"类别名称 '{target_class}' 不在数据集中。可用类别: {list(class_to_idx.keys())}")
        target_idx = class_to_idx[target_class]
        class_identifier = f"名称为 '{target_class}' (索引 {target_idx})"
    elif isinstance(target_class, int):
        if not (0 <= target_class < num_classes):
            raise ValueError(f"类别索引 {target_class} 超出范围。有效索引范围是 [0, {num_classes - 1}]。")
        target_idx = target_class
        class_identifier = f"索引为 {target_idx}"
    else:
        raise TypeError(f"target_class 必须是字符串(str), 整数(int) 或 -1，但收到了 {type(target_class)}。")

    class_indices = [i for i, label in enumerate(targets) if label == target_idx]

    if not class_indices:
        print(f"警告: 数据集中未找到类别 {class_identifier} 的任何样本。返回一个空的 Subset。")
        return Subset(dataset, [])

    if num_samples is None:
        final_indices = class_indices
        print(f"为类别 {class_identifier} 找到 {len(final_indices)} 个样本，将返回所有样本。")
    else:
        if num_samples > len(class_indices):
            print(f"警告: 请求抽取 {num_samples} 个样本，但类别 {class_identifier} 只有 {len(class_indices)} 个可用样本。将返回所有可用样本。")
            final_indices = class_indices
        else:
            final_indices = random.sample(class_indices, num_samples)
            print(f"为类别 {class_identifier} 随机抽取了 {len(final_indices)} 个样本。")

    return Subset(dataset, final_indices)

def get_class_value_vit(model, tensor):
    feature_storage = {}
    def forward_hook(module, input, output):
        feature_storage['normed_tokens'] = output.detach()
    forward_handle = model.norm.register_forward_hook(forward_hook)
    with torch.no_grad():
        logits = model(tensor)
    forward_handle.remove()
    normed_tokens = feature_storage['normed_tokens']
    return normed_tokens
def get_class_value(model, tensor, layer):
    feature_storage = {}
    def forward_hook(module, input, output):
        feature_storage['normed_tokens'] = output.detach()
    forward_handle = model.blocks[layer].norm1.register_forward_hook(forward_hook)
    with torch.no_grad():
        logits = model(tensor)
    forward_handle.remove()
    normed_tokens = feature_storage['normed_tokens']
    with torch.no_grad():
        blk = model.blocks[layer]
        attn_module = blk.attn   
        # 手动执行 timm Attention 模块的 forward
        B, N, C = normed_tokens.shape
        qkv = attn_module.qkv(normed_tokens).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Query张量只取prompt
        q = q[:,:,model.patch_embed.num_patches:]
        # Key张量只取patch
        k = k[:,:,:model.patch_embed.num_patches]
        # Value张量取patch
        v = v[:,:,:model.patch_embed.num_patches]
        q, k = attn_module.q_norm(q), attn_module.k_norm(k)
        attn = (q @ k.transpose(-2, -1)) * attn_module.scale
        attn = attn.softmax(dim=-1) # <--- 这就是我们需要的张量！
        attn = attn_module.attn_drop(attn)
        attn_output = (attn @ v).transpose(1, 2)
        x = attn_output
        attn_output = attn_output.reshape(B, model.num_classes, C)
        attn_output = attn_module.norm(attn_output)
        attn_output = attn_module.proj(attn_output)
        attn_output = attn_module.proj_drop(attn_output)
        # 完成 Block 的剩余部分
        tokens = blk.drop_path1(attn_output)
        tokens = tokens + blk.drop_path2(blk.mlp(blk.norm2(tokens)))
        tokens = model.norm(tokens)
    return q,tokens
def tsne_vis(mock_tensor):
    print(f"原始张量形状: {mock_tensor.shape}")
    sample_num = mock_tensor.shape[0]
    C = mock_tensor.shape[1]
    feature_dim = mock_tensor.shape[2]
    
    # --- 第 2 步: 数据预处理 ---
    data_for_tsne = mock_tensor.cpu().numpy()
    num_points = sample_num * C
    reshaped_data = data_for_tsne.reshape(num_points, feature_dim)
    print(f"用于 t-SNE 的数据形状: {reshaped_data.shape}")
    
    labels = np.tile(np.arange(C), sample_num)
    print(f"标签数组形状: {labels.shape}")
    
    # --- 第 3 步: 执行 t-SNE 降维 ---
    print("开始执行 t-SNE 降维，这可能需要一些时间...")
    start_time = time.time()
    
    # 初始化 t-SNE 模型
    # ===================== 主要修改点在这里 =====================
    # 将 n_iter 修改为 max_iter
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, learning_rate='auto', init='pca')
    # ==========================================================
    
    # 执行降维
    tsne_results = tsne.fit_transform(reshaped_data)
    
    end_time = time.time()
    print(f"t-SNE 降维完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"降维后的数据形状: {tsne_results.shape}")
    
    
    # --- 第 4 步: 绘制 3D 散点图 ---
    print("开始绘制 3D 散点图...")
    
    fig, ax = plt.subplots(figsize=(8, 6),dpi=100)
    #ax = fig.add_subplot(111)
    
    colors = plt.get_cmap('tab10', C)
    
    for i in range(C):
        indices = np.where(labels == i)
        x = tsne_results[indices, 0]
        y = tsne_results[indices, 1]
        #z = tsne_results[indices, 2]
        ax.scatter(x, y, color=colors(i), label=f'Class {i}', s=15, alpha=0.7)
    
    ax.set_title(f't-SNE 2D Visualization (Sample Num={sample_num}, Class Num={C})', fontsize=16)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    #ax.set_zlabel('Dim 3')
    ax.legend(title='Class')
    ax.grid(True)
    
    plt.show()
def umap_vis(mock_tensor):
    # --- 步骤 1: 生成模拟数据 ---
    B = mock_tensor.shape[0]  # 样本数
    C = mock_tensor.shape[1]
    D = mock_tensor.shape[2]  # 特征维度
    
    features = mock_tensor.cpu().numpy().reshape(B*C, feature_dim)
    labels = np.tile(np.arange(C), B)
    
    # --- 步骤 2: 初始化并应用UMAP降维 ---
    print("正在使用UMAP进行降维...")
    reducer = umap.UMAP(
        n_neighbors=600,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42
    )
    
    embedding = reducer.fit_transform(features)
    
    print(f"降维后的数据 'embedding' 形状: {embedding.shape}")
    
    # --- 步骤 3: 使用Matplotlib进行可视化 (已修正) ---
    print("正在绘制2D散点图...")
    
    fig, ax = plt.subplots(figsize=(8, 6),dpi=100)
    
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap='tab10',
        s=15
    )
    
    # 设置图形属性
    ax.set_aspect('equal', 'datalim')
    ax.set_xlabel('UMAP Dimension 1', fontsize=10)
    ax.set_ylabel('UMAP Dimension 2', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # ---------- 这里是修改图例位置的关键代码 ----------
    ax.legend(
        handles=scatter.legend_elements()[0],
        labels=[f'Class {i}' for i in range(C)],
        loc='center left',         # 将图例的左边中点作为锚点
        bbox_to_anchor=(1.02, 0.5), # 将锚点放置在绘图区域右侧外部、垂直居中的位置
        frameon=False,
    )
    # ----------------------------------------------------
    
    # 自动调整布局，防止图例被裁剪
    plt.tight_layout()
    
    # 显示图形
    plt.show()
    
    print("任务完成！")