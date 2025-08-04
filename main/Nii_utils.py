import json
import torch
import random
import numpy as np
import SimpleITK as sitk
from os.path import join
from sklearn.metrics import f1_score


# ---- Training

def setup_seed(seed):
    """
    Set random seed for reproducibility across PyTorch, NumPy, and Python.

    Args:
        seed (int): The seed to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # The operation of cuDNN is deterministic, and for neural network layers with the same inputs then the same outputs

    # Ensures reproducibility in cuDNN (NVIDIA's backend)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # cuDNN's auto-tuner
    torch.backends.cudnn.enabled = False
    # set_determinism(seed=seed)  # monai augmentation


def Save_Parameter(opt):
    """
    Save training parameters to a text and JSON file for logging and reproducibility.

    Args:
        args (argparse.Namespace): Contains all argument key-value pairs.
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        if k == 'data_split':
            continue
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------\n'
    print(message)

    # Save as plain text
    with open(join(opt.save_dir, 'train_parameter.txt'), 'wt') as f:
        f.write(message)
        f.write('\n')

    # Save as structured JSON
    vars(opt)['device'] = str(vars(opt)['device'])
    with open(join(opt.save_dir, 'train_parameter.json'), 'w') as f:
        json.dump(vars(opt), f, indent=4)


# ---- Data

def NiiDataRead(path, as_type=np.float32):
    """
    Read a NIfTI medical image and return its data, spacing, origin, and direction.

    Args:
        path (str): Path to the NIfTI file.
        as_type (np.dtype): Desired data type of the returned image array.

    Returns:
        volumn (ndarray): 3D image data in [z, y, x] order.
        spacing_ (ndarray): Voxel spacing in [z, y, x] order.
        origin (tuple): Image origin in physical space.
        direction (tuple): Image orientation (direction cosine matrix).
    """
    nii = sitk.ReadImage(path)
    spacing = nii.GetSpacing()  # Original spacing is [x, y, z]
    volumn = sitk.GetArrayFromImage(nii)  # Converts to numpy array in [z, y, x]
    origin = nii.GetOrigin()
    direction = nii.GetDirection()

    spacing_x = spacing[0]
    spacing_y = spacing[1]
    spacing_z = spacing[2]

    # Reorder spacing to match numpy array order
    spacing_ = np.array([spacing_z, spacing_y, spacing_x])
    return volumn.astype(as_type), spacing_.astype(np.float32), origin, direction


def NiiDataWrite(save_path, volumn, spacing, origin, direction, as_type=np.float32):
    """
    Save a 3D numpy array as a NIfTI file with spatial information.

    Args:
        save_path (str): Destination path to save the NIfTI file.
        volumn (ndarray): 3D image data [z, y, x].
        spacing (ndarray): Spacing in [z, y, x] order.
        origin (tuple): Image origin.
        direction (tuple): Image orientation.
        as_type (np.dtype): Data type to save as.
    """
    spacing = spacing.astype(np.float64)
    raw = sitk.GetImageFromArray(volumn[:, :, :].astype(as_type))

    # Convert spacing back to [x, y, z] for SimpleITK
    spacing_ = (spacing[2], spacing[1], spacing[0])
    raw.SetSpacing(spacing_)
    raw.SetOrigin(origin)
    raw.SetDirection(direction)
    sitk.WriteImage(raw, save_path)


def harmonize_mr(X, min=0, max=255):
    """
    Clip MR image values to [min, max] and normalize them to [-1, 1].

    Args:
        X (ndarray): Input image.
        min (float): Minimum intensity to clip.
        max (float): Maximum intensity to clip.

    Returns:
        Normalized image with values in [-1, 1].
    """
    X = np.clip(X, min, max)
    X = X / abs(max-min) * 2 - 1
    return X


def harmonize_mr_reverse(X, min=0, max=255):
    """
    Renormalize MR image [-1, 1] to form them to [min, max] and Clip.

    Args:
        X (ndarray): Input image.
        min (float): Minimum intensity to clip.
        max (float): Maximum intensity to clip.

    Returns:
        ReNormalized image with values in [min, max].
    """
    X = (X + 1) * abs(max-min) / 2
    X = np.clip(X, min, max)
    return X



# ---- Metrics

def compute_mae(pred, label, mask=None):
    """
    Compute Mean Absolute Error (MAE) between predicted and ground-truth images.

    Args:
        pred (ndarray): Predicted values.
        label (ndarray): Ground-truth values.
        mask (ndarray, optional): Binary mask to compute MAE only on specific regions.

    Returns:
        float: Mean absolute error.
    """
    mae = np.abs(pred - label)
    if mask.any():
        mae = np.mean(mae[mask == 1])
    else:
        mae = np.mean(mae)
    return mae


def dice_coefficient(prediction, target):
    """
    Compute Dice coefficient (F1 score) between binary predictions and targets.

    Args:
        prediction (ndarray or tensor): Binary predicted mask.
        target (ndarray or tensor): Binary ground-truth mask.

    Returns:
        float: Dice coefficient (ranges from 0 to 1).
    """
    # Convert the predicted glioma mask and the true glioma mask into a one-dimensional array
    prediction = prediction.ravel()
    target = target.ravel()


    dice = f1_score(target, prediction, average='binary')
    return dice




