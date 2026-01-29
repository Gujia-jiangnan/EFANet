import numpy as np
import torch
import torch.nn.functional as F
import cv2
from sklearn.metrics import auc
from torchvision import transforms
from functools import partial
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([
    transforms.ToTensor(),                  #
    transforms.Resize(size=(224, 224), antialias=True),
    transforms.Normalize(                   #
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
def tensor_to_numpy(tensor):
    """
    Convert normalized PyTorch tensor back to displayable NumPy array.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    device = tensor.device
    # tensor * std + mean
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    unnormalized_tensor = tensor.clone() * std + mean
    unnormalized_tensor = torch.clamp(unnormalized_tensor, 0, 1)
    cpu_tensor = unnormalized_tensor.cpu()
    numpy_image = cpu_tensor.numpy().transpose((1, 2, 0))
    numpy_image = (numpy_image * 255).astype(np.uint8)
    return numpy_image
def apply_colormap_and_overlay(saliency_map, image_numpy):
    # Apply colormap and overlay
    size = image_numpy.shape[0]
    alpha = 0.6
    saliency_map = cv2.resize(
        saliency_map,
        dsize=(size, size), 
        interpolation=cv2.INTER_LINEAR
    )
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(heatmap, alpha, image_numpy, 1 - alpha, 0)
    return overlay

def raw_attention(model, image_tensor, target_classes):
    """
    Calculate raw attention
    Args:
        model: model
        image_tensor: input tensor (B, C, H, W)
        target_classes: target classes (B)
    Returns:
        scores: saliency map (B, 12, 14, 14)
    """
    activation_storage = {}
    
    target_layer = model.attn_head.cross_attn.attn_bridge
    def forward_hook(module, input, output):
        activation_storage['activation'] = output.detach()
    forward_handle = target_layer.register_forward_hook(forward_hook)
    with torch.no_grad():
        logits = model(image_tensor)
    forward_handle.remove()
    scores = activation_storage['activation'][:,:,target_classes,:].squeeze(2)
    # Normalize
    min_vals = torch.amin(scores, dim=2, keepdim=True)
    
    max_vals = torch.amax(scores, dim=2, keepdim=True)
    scores = (scores - min_vals) / (max_vals - min_vals)
    
    scores = scores.reshape(image_tensor.shape[0],12,14,14)
    return scores
    
def gwaa_cam(model, image_tensor, target_classes):
    """
    Gradient-Weighted Attention Attribution 
    Args:
        model: model
        image_tensor: input tensor (B, C, H, W)
        target_classes: target classes (B)
    Returns:
        scores: saliency map (B, 12, 14, 14)
    """
    activation_storage = {}
    gradient_storage = {}
    # target_layer = model.attn_head.cross_attn.attn_bridge
    target_layer = model.blocks[-1].attn.attn_drop

    def forward_hook(module, input, output):
        activation_storage['activation'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradient_storage['gradient'] = grad_output[0].detach()

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    logits = model(image_tensor)
    model.zero_grad()
    logits.gather(dim=1, index=target_classes.view(image_tensor.shape[0], 1)).sum().backward()
    
    forward_handle.remove()
    backward_handle.remove()

    B, H, N, D = activation_storage['activation'].shape
    
    idx = target_classes.view(B, 1, 1, 1)
    idx = idx.expand(B, H, 1, D)
    with torch.no_grad():
        gathered_attention = activation_storage['activation'].gather(dim=2, index=idx).squeeze(2)
        gathered_grads = gradient_storage['gradient'].gather(dim=2, index=idx).squeeze(2)


    gwaa_scores = F.relu(gathered_grads) * gathered_attention
    
    # Sum across heads for each patch
    gwaa_maps = torch.sum(gwaa_scores, dim=1)
    
    # Normalize
    min_vals = torch.amin(gwaa_maps, dim=1, keepdim=True)
    max_vals = torch.amax(gwaa_maps, dim=1, keepdim=True)
    gwaa_maps = (gwaa_maps - min_vals) / (max_vals - min_vals)
    
    gwaa_maps = gwaa_maps.reshape(B,1,14,14)
    return gwaa_maps
    
def att_cam(model, image_tensor, target_classes, m_steps=20):
    """
    Integrated Gradients on Attention
    """
    B = image_tensor.shape[0]
    model.eval()
    with torch.no_grad():
        _, attn_final = model(image_tensor, return_attention=True)
    
    _B, H, C, Np = attn_final.shape
    patch_size_h = patch_size_w = int(Np**0.5)

    attn_baseline = torch.ones_like(attn_final) / Np
    integrated_grads_accumulator = torch.zeros_like(attn_final)

    hook_state = {'interpolated_attn': None}


    def attention_hook(module, input):
        return (hook_state['interpolated_attn'],)

    target_module = model.blocks[-1].attn.attn_drop
    handle = target_module.register_forward_pre_hook(attention_hook)

    # Riemann approximation
    for k in range(1, m_steps + 1):
        alpha = float(k) / m_steps
        
        interpolated_attn = attn_baseline + alpha * (attn_final - attn_baseline)
        interpolated_attn.requires_grad_(True)
        hook_state['interpolated_attn'] = interpolated_attn

        logits = model(image_tensor, return_attention=False)
        
        model.zero_grad()
        logits.gather(dim=1, index=target_classes.view(B, 1)).sum().backward()
        
        if hook_state['interpolated_attn'].grad is not None:
            integrated_grads_accumulator += hook_state['interpolated_attn'].grad
            
    handle.remove()

    # Final attribution
    with torch.no_grad():
        avg_grads = integrated_grads_accumulator / m_steps
        ig_attribution = (attn_final - attn_baseline) * avg_grads

        idx = target_classes.view(B, 1, 1, 1).expand(B, H, 1, Np)
        gathered_ig_attribution = ig_attribution.gather(dim=2, index=idx).squeeze(2)

        ig_scores = F.relu(gathered_ig_attribution)
        ig_cam_maps = torch.mean(ig_scores, dim=1)
        
        min_vals = torch.amin(ig_cam_maps, dim=-1, keepdim=True)
        max_vals = torch.amax(ig_cam_maps, dim=-1, keepdim=True)
        ig_cam_maps = (ig_cam_maps - min_vals) / (max_vals - min_vals)
        
        ig_cam_maps = ig_cam_maps.reshape(B, 1, patch_size_h, patch_size_w)

    return ig_cam_maps

def grad_cam(model, image_tensor, target_classes):
    """
    Compute Grad-CAM.
    It calculates gradients of target class scores with respect to image patches,
    then uses these gradients as weights to perform a weighted sum over the patches,
    thereby generating a saliency map.
    Args:
        model: model
        image_tensor: input tensor (B, C, H, W)
        target_classes: target classes (B)
    Returns:
        scores: saliency map (B, 12, 14, 14)
    """
    # 创建一个字典来存储激活和梯度
    activation_storage = {}
    gradient_storage = {}
    target_layer = model.blocks[-1].norm1

    def forward_hook(module, input, output):
        activation_storage['activation'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradient_storage['gradient'] = grad_output[0].detach()
    
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook) 

    logits = model(image_tensor)
    model.zero_grad()
    logits.gather(dim=1, index=target_classes.view(image_tensor.shape[0], 1)).sum().backward() 
    forward_handle.remove()
    backward_handle.remove()

    activations = activation_storage['activation'] # Shape: [B, num_tokens, embed_dim]
    gradients = gradient_storage['gradient']       # Shape: [B, num_tokens, embed_dim]
    B, N, C = activations.shape
    
    weights = torch.mean(gradients[:, :196, :], dim=1) # Shape: [B, embed_dim]
    weighted_activations = activations[:, :196, :] * weights.unsqueeze(1) # Shape: [B, num_patches, embed_dim]
    gcam_maps = F.relu(torch.sum(weighted_activations, dim=2)) # Shape: [B, num_patches]
    
    # Normalize
    min_vals = torch.amin(gcam_maps, dim=1, keepdim=True)
    max_vals = torch.amax(gcam_maps, dim=1, keepdim=True)
    gcam_maps = (gcam_maps - min_vals) / (max_vals - min_vals)
    
    gcam_maps = gcam_maps.reshape(B,1,14,14)
    return gcam_maps

def rollout_cam(model, image_tensor, target_classes):
    """
    Attention Rollout
    Aggregates multi-layer attention information to generate a single heatmap
    that clearly visualizes the overall attention flow from the "input layer"
    to the "output layer". This explains which original image regions the model's
    final decision is based on.
    Args:
        model: model
        image_tensor: input tensor (B, C, H, W)
        target_classes: target classes (B)
    Returns:
        scores: saliency map (B, 12, 14, 14)
    """
    attention_storage = []
    handles = []
    
    def forward_hook(module, input, output):
        attention_storage.append(input[0].detach())
    
    for i in range(len(model.blocks)):
        forward_handle = model.blocks[i].attn.attn_drop.register_forward_hook(forward_hook)
        handles.append(forward_handle)
    
    with torch.no_grad():
        logits = model(image_tensor)
    
    for handle in handles:
        handle.remove()

    B, H, N, D = attention_storage[-1].shape
    idx = target_classes.view(B, 1, 1, 1)
    idx = idx.expand(B, H, 1, D)
    
    # Adapt our model EFANet
    n = len(attention_storage)
    for i in range(n):
        if i == n - 1:
            attention_storage[i] = attention_storage[i].gather(dim=2, index=idx)
        else:
            attention_storage[i] = attention_storage[i][:,:,:196,:196]
        # head average A = Eh(A)
        attention_storage[i] = torch.mean(attention_storage[i], dim = 1)
        # Residuals A = 0.5×A + 0.5×I
        if i == n - 1:
            attention_storage[i].mul_(0.5)
        else:
            I = torch.eye(D, device=image_tensor.device).unsqueeze(0).repeat(B,1,1)
            attention_storage[i].mul_(0.5).add_(I.mul_(0.5))

    # From output to input
    rollout_maps = attention_storage[-1].clone()
    for i in range(2, n):
        rollout_maps = rollout_maps @ attention_storage[-i]
    rollout_maps = rollout_maps.squeeze(1)
    # Normalize
    min_vals = torch.amin(rollout_maps, dim=1, keepdim=True)
    max_vals = torch.amax(rollout_maps, dim=1, keepdim=True)
    rollout_maps = (rollout_maps - min_vals) / (max_vals - min_vals)
    
    rollout_maps = rollout_maps.reshape(B,1,14,14)
    return rollout_maps

# def prompt_cam(model, image_tensor, target_classes):
#     """
#     Prompt-CAM
    
#     Args:
#         model: model
#         image_tensor: input tensor (B, C, H, W)
#         target_classes: target classes (B)
#     Returns:
#         scores: saliency map (B, 12, 14, 14)
#     """
#     B = image_tensor.shape[0]
#     num_heads = model.blocks[-1].attn.num_heads
#     head_index_per_batch = torch.zeros(B, dtype=torch.long, device=image_tensor.device)
#     with torch.no_grad():
#         logits, attn_maps = model(image_tensor, return_attention=True) # logits shape: [1, num_classes]
     
#     for i in range(B):
#         remaining_head_list = list(range(num_heads))
#         pruned_head_index = None
#         blur_head_lst = []
#         blur_head_probs = []
        
#         # Determine the ranking of heads by iteratively finding the head that, when blurred,
#         # gives the highest probability for the target.
#         while len(remaining_head_list) > 0:
#             highest_score=-1e8
#             remaining_head_scores= []
    
#             for head_idx in remaining_head_list:
#                 with torch.no_grad():
#                     output = model(image_tensor[i].unsqueeze(0),
#                                    blur_head_lst=blur_head_lst+[head_idx],
#                                    target_cls=target_classes[i])
                
#                 probabilities = torch.softmax(output.squeeze(-1), dim=-1)
#                 remaining_head_scores.append(probabilities[0,target_classes[i]].item())
    
#                 if remaining_head_scores[-1] > highest_score:
#                     highest_score=remaining_head_scores[-1] 
#                     pruned_head_index=head_idx
    
#             if pruned_head_index is not None:
#                 blur_head_lst.append(pruned_head_index)
#                 remaining_head_list.remove(pruned_head_index)
#                 blur_head_probs.append(highest_score)  
#         head_index_per_batch[i] = blur_head_lst[-1]

#     _, H, _, D = attn_maps.shape
#     idx = target_classes.view(B, 1, 1, 1)
#     idx = idx.expand(B, H, 1, D)
#     attn_maps = attn_maps.gather(dim=2, index=idx).squeeze(2)
#     head_index_per_batch = head_index_per_batch.view(B, 1, 1)
#     head_index_per_batch = head_index_per_batch.expand(B, 1, D)
#     attn_maps = attn_maps.gather(dim=1, index=head_index_per_batch).squeeze(1) # (B, 196)
    
#     # Normalize
#     min_vals = torch.amin(attn_maps, dim=1, keepdim=True)
#     max_vals = torch.amax(attn_maps, dim=1, keepdim=True)
#     attn_maps = (attn_maps - min_vals) / (max_vals - min_vals)
    
#     attn_maps = attn_maps.reshape(B,1,14,14)
#     return attn_maps
def prompt_cam(model, image_tensor, target_classes):
    """
    Prompt-CAM Batch Parallelized
    
    Args:
        model: model
        image_tensor: input tensor (B, C, H, W)
        target_classes: target classes (B)
    Returns:
        scores: saliency map (B, 12, 14, 14)
    """
    model.eval()
    B = image_tensor.shape[0]
    num_heads = model.blocks[-1].attn.num_heads
    device = image_tensor.device

    with torch.no_grad():
        # output shape: logits, attn_maps shape: (B, Heads, Num_Classes, Num_Patches)
        _, attn_maps_orig = model(image_tensor, return_attention=True)
    
    _, _, num_classes, num_patches = attn_maps_orig.shape
    # Shape: (B, Heads, Num_Classes, 1)
    current_mask = torch.ones((B, num_heads, num_classes, 1), device=device)
    
    batch_indices = torch.arange(B, device=device)
    
    removed_history = torch.zeros((B, num_heads), dtype=torch.long, device=device)
    active_heads_status = torch.ones((B, num_heads), device=device)
    for step in range(num_heads):
        candidate_scores = torch.full((B, num_heads), -float('inf'), device=device)
        for h in range(num_heads):
            test_mask = current_mask.clone()
            # test_mask[b, h, target_cls, :] = 0
            test_mask[batch_indices, h, target_classes, :] = 0.0
            with torch.no_grad():
                output = model(image_tensor, specific_attn_mask=test_mask)
            probs = torch.softmax(output, dim=-1)
            target_probs = probs.gather(1, target_classes.unsqueeze(1)).squeeze(1)
            candidate_scores[:, h] = target_probs
            
        candidate_scores[active_heads_status == 0] = -float('inf')
        heads_to_prune = torch.argmax(candidate_scores, dim=1) # (B,)
        removed_history[:, step] = heads_to_prune
        active_heads_status.scatter_(1, heads_to_prune.unsqueeze(1), 0.0)
        current_mask[batch_indices, heads_to_prune, target_classes, :] = 0.0
        
    best_head_indices = removed_history[:, -1] # (B,)

    # attn_maps_orig: (B, H, Q, P)
    head_idx = best_head_indices.view(B, 1, 1, 1).expand(B, 1, num_classes, num_patches)
    selected_head_map = attn_maps_orig.gather(dim=1, index=head_idx).squeeze(1)
    
    class_idx = target_classes.view(B, 1, 1).expand(B, 1, num_patches)
    final_cam = selected_head_map.gather(dim=1, index=class_idx).squeeze(1) 
    # (B, 196)

    min_vals = torch.amin(final_cam, dim=1, keepdim=True)
    max_vals = torch.amax(final_cam, dim=1, keepdim=True)
    final_cam = (final_cam - min_vals) / (max_vals - min_vals)

    # Reshape
    final_cam = final_cam.reshape(B, 1, 14, 14)
    
    return final_cam
def lrp_cam(model, image_tensor, target_classes):
    """
    Compute Gradient * Activation (as an approximation of LRP at the final layer).

    Note:
    Strictly speaking, LRP (Bach et al. 2015) is a rule-based full-network backpropagation.
    However, in Transformer visualization, computing Activations * Gradients is the standard method
    to obtain layer-specific relevance scores.
    This is mathematically equivalent to element-wise Grad-CAM.
    """
    activation_storage = {}
    gradient_storage = {}
    target_layer = model.blocks[-1].norm1

    def forward_hook(module, input, output):
        activation_storage['activation'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradient_storage['gradient'] = grad_output[0].detach()
    
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    logits = model(image_tensor)
    model.zero_grad()
    logits.gather(dim=1, index=target_classes.view(image_tensor.shape[0], 1)).sum().backward()
    
    forward_handle.remove()
    backward_handle.remove()

    activations = activation_storage['activation'] 
    gradients = gradient_storage['gradient']       
    B, N, C = activations.shape
    
    # Unlike Grad-CAM which first averages the gradients
    relevance = activations * gradients 
    
    cam_maps = torch.sum(relevance, dim=2) # [B, N]
    
    cam_maps = cam_maps[:, :196]
    cam_maps = F.relu(cam_maps)
    
    min_vals = torch.amin(cam_maps, dim=1, keepdim=True)
    max_vals = torch.amax(cam_maps, dim=1, keepdim=True)
    cam_maps = (cam_maps - min_vals) / (max_vals - min_vals + 1e-8)
    
    cam_maps = cam_maps.reshape(B, 1, 14, 14)
    return cam_maps

def deletion_metric(model, image_tensor, saliency_maps, target_classes, device, steps=100):
    """
    Compute the Deletion metric by gradually removing the most important pixels.

    Args:
        model (torch.nn.Module): The model to evaluate.
        image_tensor (torch.Tensor): Original image tensor with shape (B, C, H, W).
        saliency_maps (torch.Tensor): Saliency map tensor with shape (B, 1, H // patch_size, W // patch_size).
        target_classes (torch.Tensor): Target class indices with shape (B,).
        device: Computing device (e.g., 'cuda' or 'cpu').
        steps (int): Total number of deletion steps.

    Returns:
        torch.Tensor: Confidence curve tensor with shape (B, steps+1).
    """
    B, C, H, W = image_tensor.shape
    N = H * W
    saliency_maps = F.interpolate(
        saliency_maps,
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    )
    saliency_maps = saliency_maps.reshape(B, N)
    _, sorted_indices = torch.sort(saliency_maps, dim=1, descending=True)
    all_confidences = torch.zeros(B, steps+1, device=device)
    # Initial prediction confidence
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        predicted_class = target_classes.view(B, 1)
        confidences = probs.gather(dim=1, index=predicted_class).squeeze()
        all_confidences[:, 0] = confidences
        
    
    pixels_per_step = N // steps
    image_tensor_clone = image_tensor.clone()
    with torch.no_grad():
        for i in range(steps):
            # Calculate total number of pixels to be masked up to the current step
            num_pixels_to_mask = (i + 1) * pixels_per_step
            
            # Get flat indices of all pixels to be masked at current step (B, num_pixels_to_mask)
            indices_to_mask = sorted_indices[:, :num_pixels_to_mask]
    
            # --- Create masks ---
            # 1. Create a flat mask (B, N) initialized to 1
            mask_flat = torch.ones(B, N, device=device)
            # 2. Use scatter_ to set positions requiring masking to 0. This is a highly efficient in-place operation
            mask_flat.scatter_(dim=1, index=indices_to_mask, value=0.0)
            # 3. Reshape the flat mask to image dimensions for multiplication with the image
            current_mask = mask_flat.view(B, 1, H, W)
            
            # --- Inference ---
            # Apply mask to image using broadcasting mechanism
            masked_images = image_tensor_clone * current_mask
            
            # Forward
            logits = model(masked_images)
            probs = F.softmax(logits, dim=1)
    
            # --- Confidence curve ---
            if isinstance(target_classes, int):
                # Use the same target class for all batch items (generally not used, as insertion/deletion metrics are applied to predicted classes)
                confidences = probs[:, target_classes]
            else:
                predicted_class = target_classes.view(B, 1)
                confidences = probs.gather(dim=1, index=predicted_class).squeeze()
                all_confidences[:, i+1] = confidences
    
    return all_confidences
def insertion_metric(model, image_tensor, saliency_maps, target_classes, device, steps=100, baseline_type='black'):
    """
    Compute the Insertion metric by gradually inserting the most important pixels.

    Args:
        model (torch.nn.Module): The model to evaluate.
        image_tensor (torch.Tensor): Original image tensor with shape (B, C, H, W).
        saliency_maps (torch.Tensor): Saliency map tensor with shape (B, H, W).
        target_classes (torch.Tensor): Target class indices with shape (B,).
        device: Computing device (e.g., 'cuda' or 'cpu').
        steps (int): Total number of insertion steps.
        baseline_type (str): Type of baseline image, 'blur' (Gaussian blur) or 'black'.

    Returns:
        torch.Tensor: Confidence curve tensor with shape (B, steps+1).
    """
    B, C, H, W = image_tensor.shape
    N = H * W
    saliency_maps = F.interpolate(
        saliency_maps,
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    )
    # 1. Sort saliency map to obtain pixel indices from most important to least important
    saliency_maps_flat = saliency_maps.reshape(B, N)
    # Insertion and Deletion both start from the most important pixels, so the sorting method remains the same
    _, sorted_indices = torch.sort(saliency_maps_flat, dim=1, descending=True)
    
    # Tensor to store confidence scores for each step
    all_confidences = torch.zeros(B, steps + 1, device=device)

    # 2. Create baseline image
    if baseline_type == 'blur':
        # Create baseline image using Gaussian blur
        from torchvision.transforms.functional import gaussian_blur
        baseline_images = gaussian_blur(image_tensor, kernel_size=(51, 51), sigma=(50, 50))
    else: # 'black' or any other value
        baseline_images = torch.zeros_like(image_tensor, device=device)

    # 3. Compute initial confidence (baseline image) for Step 0
    with torch.no_grad():
        logits = model(baseline_images)
        probs = F.softmax(logits, dim=1)
        predicted_class = target_classes.view(B, 1)
        confidences = probs.gather(dim=1, index=predicted_class).squeeze()
        all_confidences[:, 0] = confidences
        
    # Number of pixels to insert at each step
    pixels_per_step = N // steps

    # 4. Loop to gradually insert pixels
    with torch.no_grad():
        for i in range(steps):
            # Calculate total number of pixels to be inserted up to the current step
            num_pixels_to_insert = (i + 1) * pixels_per_step
            
            # Get flat indices of all pixels to be inserted at current step (B, num_pixels_to_insert)
            indices_to_insert = sorted_indices[:, :num_pixels_to_insert]
    
            # --- Create mask ---
            # 1. Create a flat mask (B, N) initialized to 0 (all masked)
            mask_flat = torch.zeros(B, N, device=device)
            # 2. Use scatter_ to set positions requiring insertion to 1. This is an efficient in-place operation
            mask_flat.scatter_(dim=1, index=indices_to_insert, value=1.0)
            # 3. Reshape the flat mask to image dimensions
            current_mask = mask_flat.view(B, 1, H, W)
            
            # --- Construct current image ---
            # Use original image pixels where mask equals 1, and baseline image pixels where mask equals 0
            inserted_images = image_tensor * current_mask + baseline_images * (1 - current_mask)
            
            # Model forward pass
            logits = model(inserted_images)
            probs = F.softmax(logits, dim=1)
    
            # --- Record confidence scores ---
            confidences = probs.gather(dim=1, index=predicted_class).squeeze(dim=-1)
            all_confidences[:, i+1] = confidences
            
    return all_confidences
def calculate_auc(confidence_curves):
    """
    Compute the Area Under the Curve (AUC) of confidence curves.

    Args:
        confidence_curve_tensor (torch.Tensor): Tensor of shape (B, steps), recording confidence changes.

    Returns:
        torch.Tensor: Tensor of shape (B,), AUC values for each batch item.
    """
    # The y-values of the curve are the confidence scores
    y = confidence_curves
    
    # The x-axis is uniformly distributed from 0 to 1
    # To compute AUC correctly, we need to provide x-coordinates
    # torch.trapezoid(y, x) computes the trapezoidal area
    steps = y.shape[1]
    x = torch.linspace(0, 1, steps, device=y.device)
    
    # dim=1 indicates integration along the 'steps' dimension
    auc_values = torch.trapezoid(y, x, dim=1)

    return auc_values
