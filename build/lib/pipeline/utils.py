import subprocess
import numpy as np
import torch

def display_nvidia_smi_memory_usage():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'], check=True, text=True, stdout=subprocess.PIPE)
        output = result.stdout.strip().split('\n')
        for i, line in enumerate(output):
            total, used, free = map(int, line.split(','))
            print(f"GPU {i}: Total Memory: {total} MiB, Used Memory: {used} MiB, Free Memory: {free} MiB")
    except subprocess.CalledProcessError:
        print("Failed to execute nvidia-smi. Make sure you have the utility installed and that you have NVIDIA GPUs.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def bbox2_3D(img,margin=5):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    xmin, xmax = np.where(r)[0][[0, -1]]
    ymin, ymax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    xmin = max(0,xmin-margin)
    xmax = min(img.shape[0],xmax+margin)
    ymin = max(0,ymin-margin)
    ymax = min(img.shape[1],ymax+margin)
    zmin = max(0,zmin-margin)
    zmax = min(img.shape[2],zmax+margin)
    return xmin, xmax, ymin, ymax, zmin, zmax


def one_hot_encode(tensor, num_classes):
    """
    Convert tensor with indices (after argmax) to one-hot encoded tensor.
    """
    tensor = tensor.long()  # Ensure tensor is of dtype long
    shape = tensor.shape
    one_hot = torch.zeros(*shape, num_classes, dtype=torch.float32, device=tensor.device)
    return one_hot.scatter_(len(shape), tensor.unsqueeze(-1), 1.0)

def dice_per_class(y_true, y_pred, num_classes, eps=1e-7):
    """
    Calculate Dice Loss for each class.
    Parameters:
    - y_true: Ground truth masks with shape [batch_size, h, w, d].
    - y_pred: Predicted probabilities with the same shape as y_true.
    - num_classes: Total number of classes.
    - eps: Small value to avoid division by zero.
    Returns:
    - torch tensor of shape [num_classes], with dice loss per class.
    """
    y_true_one_hot = one_hot_encode(y_true, num_classes).squeeze(0).squeeze(0)
    y_pred_one_hot = one_hot_encode(y_pred,num_classes).squeeze(0).squeeze(0)  # Convert from probabilities to one-hot format
    intersection = torch.sum(y_true_one_hot * y_pred_one_hot, dim=(0, 1, 2))
    union = torch.sum(y_true_one_hot, dim=(0, 1, 2)) + torch.sum(y_pred_one_hot, dim=(0, 1, 2))
    dice_coeff = (2. * intersection + eps) / (union + eps)
    return dice_coeff

def average_dice(y_true, y_pred, num_classes, eps=1e-7):
    """
    Calculate the average Dice Loss over non-zero classes.
    Parameters:
    - y_true, y_pred, num_classes, eps: As before.
    Returns:
    - float, average dice loss over non-zero classes.
    """
    per_class_dice = dice_per_class(y_true, y_pred, num_classes, eps)
    # We'll exclude the zeroth class (often background) when averaging
    return torch.mean(per_class_dice[1:])


def displayDice(results):
    results = list(results)
    print(f'Average: {(sum(results[1:])/len(results[1:])):.4f} LUL: {results[1]:.4f} LLL: {results[2]:.4f} RUL: {results[3]:.4f} RML: {results[4]:.4f} RLL: {results[5]:.4f}')