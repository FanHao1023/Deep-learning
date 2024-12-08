"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A ResNet-50 model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (C, H /  8, W /  8)      stride =  8
        - level p4: (C, H / 16, W / 16)      stride = 16
        - level p5: (C, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a ResNet-50 backbone
    with FPN to get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.resnet50(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "layer2": "c3",
                "layer3": "c4",
                "layer4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NHWC format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # Initialize additional Conv layers for FPN.                         #
        ######################################################################
        self.fpn_params = nn.ModuleDict()

        self.fpn_params["fpn_c3"] = nn.Conv2d(dummy_out_shapes[0][1][1], out_channels, kernel_size=1)
        self.fpn_params["fpn_c4"] = nn.Conv2d(dummy_out_shapes[1][1][1], out_channels, kernel_size=1)
        self.fpn_params["fpn_c5"] = nn.Conv2d(dummy_out_shapes[2][1][1], out_channels, kernel_size=1)

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # Fill output FPN features (p3, p4, p5) using ResNet-50 features     #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################

        for i, (src_level, fpn_level) in enumerate(zip(["c3", "c4", "c5"], ["p3", "p4", "p5"])):
            # Get feature map from backbone
            x = backbone_feats[src_level]
            # Apply FPN conv layer
            x = self.fpn_params[f"fpn_c{i+3}"](x)
            # Store FPN feature
            fpn_feats[fpn_level] = x
            # For p4 and p5, interpolate from previous level (p3 and p4)
            if i > 0:
                fpn_feats[fpn_level] += F.interpolate(fpn_feats[prev_fpn_level], size=x.shape[-2:], mode="nearest")
            # Update previous level
            prev_fpn_level = fpn_level

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]
        H, W = feat_shape[-2], feat_shape[-1]

        ######################################################################
        # Implement logic to get location co-ordinates below.                #
        ######################################################################
        shifts_x = torch.arange(0, W * level_stride, step=level_stride, dtype=dtype, device=device)
        shifts_y = torch.arange(0, H * level_stride, step=level_stride, dtype=dtype, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + level_stride // 2
        location_coords[level_name] = locations

    return location_coords


def box_iou(box1, box2):
    """
    Calculate intersection over union (IoU) of two sets of boxes.

    Args:
        box1: (N, 4) Tensor of boxes
        box2: (M, 4) Tensor of boxes

    Returns:
        iou: (N, M) Tensor containing pairwise IoU values
    """
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)

    inter_min_xy = torch.max(box1[:, None, :2], box2[:, :2])
    inter_max_xy = torch.min(box1[:, None, 2:], box2[:, 2:])
    inter = (inter_max_xy - inter_min_xy).clamp(min=0)
    inter_area = inter[:, :, 0] * inter[:, :, 1]

    union_area = area1[:, None] + area2 - inter_area

    return inter_area / union_area


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long)

    keep = []
    _, indices = scores.sort(descending=True)

    while indices.numel() > 0:
        i = indices[0]
        keep.append(i.item())

        if indices.numel() == 1:
            break

        current_box = boxes[i, :].unsqueeze(0)
        other_boxes = boxes[indices[1:], :]

        iou = box_iou(current_box, other_boxes).squeeze(0)
        indices = indices[1:][iou <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep