import math
from typing import Dict, List, Optional

import torch
from a4_helper import *
from common import DetectorBackboneWithFPN, class_spec_nms, get_fpn_location_coords
from torch import nn
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate
from torchvision.ops import sigmoid_focal_loss

# Short hand type notation:
TensorDict = Dict[str, torch.Tensor]


def hello_one_stage_detector():
    print("Hello from one_stage_detector.py!")


########################################################
#                   Data augmentation                  #
########################################################

class VOC2007DetectionTiny_aug(datasets.VOCDetection):
    """
    A tiny version of PASCAL VOC 2007 Detection dataset that includes images and
    annotations with small images and no difficult boxes.
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        download: bool = False,
        image_size: int = 224,
    ):
        """
        Args:
            download: Whether to download PASCAL VOC 2007 to `dataset_dir`.
            image_size: Size of imges in the batch. The shorter edge of images
                will be resized to this size, followed by a center crop. For
                val, center crop will not be taken to capture all detections.
        """
        super().__init__(
            dataset_dir, year="2007", image_set=split, download=download
        )
        self.split = split
        self.image_size = image_size

        # fmt: off
        voc_classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "diningtable", "dog",
            "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        ]
        # fmt: on

        # Make a (class to ID) and inverse (ID to class) mapping.
        self.class_to_idx = {
            _class: _idx for _idx, _class in enumerate(voc_classes)
        }
        self.idx_to_class = {
            _idx: _class for _idx, _class in enumerate(voc_classes)
        }

        # Super class creates a list of image paths (JPG) and annotation paths
        # (XML) to read from everytime `__getitem__` is called. Here we parse
        # all annotation XMLs and only keep those which have at least one object
        # class in our required subset.
        filtered_instances = []

        for image_path, target_xml in zip(self.images, self.targets):

            target = self.parse_voc_xml(ET_parse(target_xml).getroot())
            # Only keep this sample if at least one instance belongs to subset.
            # NOTE: Ignore objects that are annotated as "difficult". These
            # are marked such because they are challenging to detect without
            # surrounding context, and VOC evaluation leaves them out. Hence
            # we discard them both during training and validation.
            _ann = [
                inst
                for inst in target["annotation"]["object"]
                if inst["name"] in voc_classes and inst["difficult"] == "0"
            ]
            if len(_ann) > 0:
                filtered_instances.append((image_path, _ann))

        self.instances = filtered_instances

        # Delete stuff from super class, we only use `self.instances`
        del self.images, self.targets

        # Define a transformation function for image: Resize the shorter image
        # edge then take a center crop (optional) and normalize.
        _transforms = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]

        # ColorJitter 
        _colorjitter_transforms = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ColorJitter(brightness=2.0, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        # Gray scale 
        _gray_transforms = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomGrayscale(p=1), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        # Gaussian blur
        _noise_transforms = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.GaussianBlur(kernel_size=3, sigma=0.3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        # if split == "train":
        #     _transforms.insert(1, transforms.CenterCrop(image_size))

        self.image_transform = transforms.Compose(_transforms)
        self.colorjitter_transform = transforms.Compose(_colorjitter_transforms)
        # self.hflip_transform = transforms.Compose(_hflip_transforms)
        self.gray_transform = transforms.Compose(_gray_transforms)
        self.noise_transform = transforms.Compose(_noise_transforms)

    def __len__(self):
        return len(self.instances)*3

    def __getitem__(self, index: int):
        # PIL image and dictionary of annotations.
        base_index = index // 3
        aug_type = index % 3

        image_path, ann = self.instances[base_index]
        image = Image.open(image_path).convert("RGB")

        # Collect a list of GT boxes: (N, 4), and GT classes: (N, )
        gt_boxes = [
            torch.Tensor(
                [
                    float(inst["bndbox"]["xmin"]),
                    float(inst["bndbox"]["ymin"]),
                    float(inst["bndbox"]["xmax"]),
                    float(inst["bndbox"]["ymax"]),
                ]
            )
            for inst in ann
        ]
        gt_boxes = torch.stack(gt_boxes)  # (N, 4)
        gt_classes = torch.Tensor([self.class_to_idx[inst["name"]] for inst in ann])
        gt_classes = gt_classes.unsqueeze(1)  # (N, 1)

        # Record original image size before transforming.
        original_width, original_height = image.size

        # Normalize bounding box co-ordinates to bring them in [0, 1]. This is
        # temporary, simply to ease the transformation logic.
        normalize_tens = torch.tensor(
            [original_width, original_height, original_width, original_height]
        )
        gt_boxes /= normalize_tens[None, :]

        # Transform input image to CHW tensor.
        if aug_type == 0:
            image = self.image_transform(image)
        elif aug_type == 1:
            image = self.colorjitter_transform(image)
            
        elif aug_type == 2:
            # image = self.hflip_transform(image)
            # image = self.gray_transform(image)
            image = self.noise_transform(image)
        """    
        elif aug_type == 3:
            image = self.noise_transform(image)
        """

        # image = image1 + image2 + image3
        # WARN: Even dimensions should be even numbers else it messes up
        # upsampling in FPN.

        # Apply image resizing transformation to bounding boxes.
        if self.image_size is not None:
            if original_height >= original_width:
                new_width = self.image_size
                new_height = original_height * self.image_size / original_width
            else:
                new_height = self.image_size
                new_width = original_width * self.image_size / original_height

            _x1 = (new_width - self.image_size) // 2
            _y1 = (new_height - self.image_size) // 2

            # Un-normalize bounding box co-ordinates and shift due to center crop.
            # Clamp to (0, image size).
            gt_boxes[:, 0] = torch.clamp(gt_boxes[:, 0] * new_width - _x1, min=0)
            gt_boxes[:, 1] = torch.clamp(gt_boxes[:, 1] * new_height - _y1, min=0)
            gt_boxes[:, 2] = torch.clamp(
                gt_boxes[:, 2] * new_width - _x1, max=self.image_size
            )
            gt_boxes[:, 3] = torch.clamp(
                gt_boxes[:, 3] * new_height - _y1, max=self.image_size
            )

        # Concatenate GT classes with GT boxes; shape: (N, 5)
        gt_boxes = torch.cat([gt_boxes, gt_classes], dim=1)

        # Center cropping may completely exclude certain boxes that were close
        # to image boundaries. Set them to -1
        invalid = (gt_boxes[:, 0] > gt_boxes[:, 2]) | (
            gt_boxes[:, 1] > gt_boxes[:, 3]
        )
        gt_boxes[invalid] = -1

        # Pad to max 40 boxes, that's enough for VOC.
        gt_boxes = torch.cat(
            [gt_boxes, torch.zeros(40 - len(gt_boxes), 5).fill_(-1.0)]
        )
        # Return image path because it is needed for evaluation.
        return image_path, image, gt_boxes

########################################################
#                  End of Data augmentation            #
########################################################   


class FCOSPredictionNetwork(nn.Module):
    """
    FCOS prediction network that accepts FPN feature maps from different levels
    and makes three predictions at every location: bounding boxes, class ID and
    centerness. This module contains a "stem" of convolution layers, along with
    one final layer per prediction. For a visual depiction, see Figure 2 (right
    side) in FCOS paper: https://arxiv.org/abs/1904.01355

    We will use feature maps from FPN levels (P3, P4, P5) and exclude (P6, P7).
    """

    def __init__(
        self, num_classes: int, in_channels: int, stem_channels: List[int]
    ):
        """
        Args:
            num_classes: Number of object classes for classification.
            in_channels: Number of channels in input feature maps. This value
                is same as the output channels of FPN, since the head directly
                operates on them.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
        """
        super().__init__()

        ######################################################################
        # TODO: Create a stem of alternating 3x3 convolution layers and RELU
        # activation modules. Note there are two separate stems for class and
        # box stem. The prediction layers for box regression and centerness
        # operate on the output of `stem_box`.
        # See FCOS figure again; both stems are identical.
        #
        # Use `in_channels` and `stem_channels` for creating these layers, the
        # docstring above tells you what they mean. Initialize weights of each
        # conv layer from a normal distribution with mean = 0 and std dev = 0.01
        # and all biases with zero. Use conv stride = 1 and zero padding such
        # that size of input features remains same: remember we need predictions
        # at every location in feature map, we shouldn't "lose" any locations.
        ######################################################################
        # Replace "pass" statement with your code
        stem_cls = []
        stem_box = []
        # Replace "pass" statement with your code
        for out_channels in stem_channels:
            stem_cls.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            stem_cls.append(nn.ReLU(inplace=True))
            stem_box.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            stem_box.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        # Wrap the layers defined by student into a `nn.Sequential` module:
        self.stem_cls = nn.Sequential(*stem_cls)
        self.stem_box = nn.Sequential(*stem_box)

        ######################################################################
        # TODO: Create THREE 3x3 conv layers for individually predicting three
        # things at every location of feature map:
        #     1. object class logits (`num_classes` outputs)
        #     2. box regression deltas (4 outputs: LTRB deltas from locations)
        #     3. centerness logits (1 output)
        #
        # Class probability and actual centerness are obtained by applying
        # sigmoid activation to these logits. However, DO NOT initialize those
        # modules here. This module should always output logits; PyTorch loss
        # functions have numerically stable implementations with logits. During
        # inference, logits are converted to probabilities by applying sigmoid,
        # BUT OUTSIDE this module.
        #
        ######################################################################

        # Replace these lines with your code, keep variable names unchanged.
        self.pred_cls = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.pred_box = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.pred_ctr = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        # Replace "pass" statement with your code
        for layer in self.stem_cls:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        for layer in self.stem_box:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.pred_cls.weight, mean=0, std=0.01)
        nn.init.constant_(self.pred_cls.bias, -math.log(99))
        nn.init.normal_(self.pred_box.weight, mean=0, std=0.01)
        nn.init.constant_(self.pred_box.bias, 0)
        nn.init.normal_(self.pred_ctr.weight, mean=0, std=0.01)
        nn.init.constant_(self.pred_ctr.bias, 0)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # OVERRIDE: Use a negative bias in `pred_cls` to improve training
        # stability. Without this, the training will most likely diverge.
        # STUDENTS: You do not need to get into details of why this is needed.
        torch.nn.init.constant_(self.pred_cls.bias, -math.log(99))

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict the desired outputs at every location
        (as described above). Format them such that channels are placed at the
        last dimension, and (H, W) are flattened (having channels at last is
        convenient for computing loss as well as perforning inference).

        Args:
            feats_per_fpn_level: Features from FPN, keys {"p3", "p4", "p5"}. Each
                tensor will have shape `(batch_size, fpn_channels, H, W)`. For an
                input (224, 224) image, H = W are (28, 14, 7) for (p3, p4, p5).

        Returns:
            List of dictionaries, each having keys {"p3", "p4", "p5"}:
            1. Classification logits: `(batch_size, H * W, num_classes)`.
            2. Box regression deltas: `(batch_size, H * W, 4)`
            3. Centerness logits:     `(batch_size, H * W, 1)`
        """

        ######################################################################
        # TODO: Iterate over every FPN feature map and obtain predictions using
        # the layers defined above. Remember that prediction layers of box
        # regression and centerness will operate on output of `stem_box`,
        # and classification layer operates separately on `stem_cls`.
        #
        # CAUTION: The original FCOS model uses shared stem for centerness and
        # classification. Recent follow-up papers commonly place centerness and
        # box regression predictors with a shared stem, which we follow here.
        #
        # DO NOT apply sigmoid to classification and centerness logits.
        ######################################################################
        # Fill these with keys: {"p3", "p4", "p5"}, same as input dictionary.
        class_logits = {}
        boxreg_deltas = {}
        centerness_logits = {}

        # Replace "pass" statement with your code
        x1 = self.stem_cls(feats_per_fpn_level['p3'])
        class_logits['p3'] = self.pred_cls(x1).flatten(start_dim=2).transpose(1, 2)

        x2 = self.stem_box(feats_per_fpn_level['p3'])
        boxreg_deltas['p3'] = self.pred_box(x2).flatten(start_dim=2).transpose(1, 2)
        centerness_logits['p3'] = self.pred_ctr(x2).flatten(start_dim=2).transpose(1, 2)

        y1 = self.stem_cls(feats_per_fpn_level['p4'])
        class_logits['p4'] = self.pred_cls(y1).flatten(start_dim=2).transpose(1, 2)

        y2 = self.stem_box(feats_per_fpn_level['p4'])
        boxreg_deltas['p4'] = self.pred_box(y2).flatten(start_dim=2).transpose(1, 2)
        centerness_logits['p4'] = self.pred_ctr(y2).flatten(start_dim=2).transpose(1, 2)

        z1 = self.stem_cls(feats_per_fpn_level['p5'])
        class_logits['p5'] = self.pred_cls(z1).flatten(start_dim=2).transpose(1, 2)

        z2 = self.stem_box(feats_per_fpn_level['p5'])
        boxreg_deltas['p5'] = self.pred_box(z2).flatten(start_dim=2).transpose(1, 2)
        centerness_logits['p5'] = self.pred_ctr(z2).flatten(start_dim=2).transpose(1, 2)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return [class_logits, boxreg_deltas, centerness_logits]


@torch.no_grad()
def fcos_match_locations_to_gt(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    gt_boxes: torch.Tensor,
) -> TensorDict:
    """
    Match centers of the locations of FPN feature with a set of GT bounding
    boxes of the input image. Since our model makes predictions at every FPN
    feature map location, we must supervise it with an appropriate GT box.
    There are multiple GT boxes in image, so FCOS has a set of heuristics to
    assign centers with GT, which we implement here.

    NOTE: This function is NOT BATCHED. Call separately for GT box batches.

    Args:
        locations_per_fpn_level: Centers at different levels of FPN (p3, p4, p5),
            that are already projected to absolute co-ordinates in input image
            dimension. Dictionary of three keys: (p3, p4, p5) giving tensors of
            shape `(H * W, 2)` where H = W is the size of feature map.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `common.py` for more details.
        gt_boxes: GT boxes of a single image, a batch of `(M, 5)` boxes with
            absolute co-ordinates and class ID `(x1, y1, x2, y2, C)`. In this
            codebase, this tensor is directly served by the dataloader.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(N, 5)` GT boxes, one for each center. They are
            one of M input boxes, or a dummy box called "background" that is
            `(-1, -1, -1, -1, -1)`. Background indicates that the center does
            not belong to any object.
    """

    matched_gt_boxes = {
        level_name: None for level_name in locations_per_fpn_level.keys()
    }

    # Do this matching individually per FPN level.
    for level_name, centers in locations_per_fpn_level.items():

        # Get stride for this FPN level.
        stride = strides_per_fpn_level[level_name]

        x, y = centers.unsqueeze(dim=2).unbind(dim=1)
        x0, y0, x1, y1 = gt_boxes[:, :4].unsqueeze(dim=0).unbind(dim=2)
        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)

        # Pairwise distance between every feature center and GT box edges:
        # shape: (num_gt_boxes, num_centers_this_level, 4)
        pairwise_dist = pairwise_dist.permute(1, 0, 2)

        # The original FCOS anchor matching rule: anchor point must be inside GT.
        match_matrix = pairwise_dist.min(dim=2).values > 0

        # Multilevel anchor matching in FCOS: each anchor is only responsible
        # for certain scale range.
        # Decide upper and lower bounds of limiting targets.
        pairwise_dist = pairwise_dist.max(dim=2).values

        lower_bound = stride * 4 if level_name != "p3" else 0
        upper_bound = stride * 8 if level_name != "p5" else float("inf")
        match_matrix &= (pairwise_dist > lower_bound) & (
            pairwise_dist < upper_bound
        )

        # Match the GT box with minimum area, if there are multiple GT matches.
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (
            gt_boxes[:, 3] - gt_boxes[:, 1]
        )

        # Get matches and their labels using match quality matrix.
        match_matrix = match_matrix.to(torch.float32)
        match_matrix *= 1e8 - gt_areas[:, None]

        # Find matched ground-truth instance per anchor (un-matched = -1).
        match_quality, matched_idxs = match_matrix.max(dim=0)
        matched_idxs[match_quality < 1e-5] = -1

        # Anchors with label 0 are treated as background.
        matched_boxes_this_level = gt_boxes[matched_idxs.clip(min=0)]
        matched_boxes_this_level[matched_idxs < 0, :] = -1

        matched_gt_boxes[level_name] = matched_boxes_this_level

    return matched_gt_boxes


def fcos_get_deltas_from_locations(
    locations: torch.Tensor, gt_boxes: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Compute distances from feature locations to GT box edges. These distances
    are called "deltas" - `(left, top, right, bottom)` or simply `LTRB`. The
    feature locations and GT boxes are given in absolute image co-ordinates.

    These deltas are used as targets for training FCOS to perform box regression
    and centerness regression. They must be "normalized" by the stride of FPN
    feature map (from which feature locations were computed, see the function
    `get_fpn_location_coords`). If GT boxes are "background", then deltas must
    be `(-1, -1, -1, -1)`.

    Args:
        locations: Tensor of shape `(N, 2)` giving `(xc, yc)` feature locations.
        gt_boxes: Tensor of shape `(N, 4)` giving GT boxes.
        stride: Stride of the FPN feature map.

    Returns:
        torch.Tensor
            Tensor of shape `(N, 4)` giving deltas from feature locations, that
            are normalized by feature stride.
    """
    ##########################################################################
    # TODO: Implement the logic to get deltas from feature locations.        #
    ##########################################################################
    # Set this to Tensor of shape (N, 4) giving deltas (left, top, right, bottom)
    # from the locations to GT box edges, normalized by FPN stride.
    
    # Compute the deltas
    l = (locations[:, 0] - gt_boxes[:, 0]) / stride  # left
    t = (locations[:, 1] - gt_boxes[:, 1]) / stride  # top
    r = (gt_boxes[:, 2] - locations[:, 0]) / stride  # right
    b = (gt_boxes[:, 3] - locations[:, 1]) / stride  # bottom

    deltas = torch.stack([l, t, r, b], dim=1)

    # Mark deltas as (-1, -1, -1, -1) for background boxes (gt_boxes with all -1s)
    is_background = (gt_boxes == -1).all(dim=1)
    deltas[is_background] = -1

    return deltas
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################



def fcos_apply_deltas_to_locations(
    deltas: torch.Tensor, locations: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Implement the inverse of `fcos_get_deltas_from_locations` here:

    Given edge deltas (left, top, right, bottom) and feature locations of FPN, get
    the resulting bounding box co-ordinates by applying deltas on locations. This
    method is used for inference in FCOS: deltas are outputs from model, and
    applying them to anchors will give us final box predictions.

    Recall in above method, we were required to normalize the deltas by feature
    stride. Similarly, we have to un-normalize the input deltas with feature
    stride before applying them to locations, because the given input locations are
    already absolute co-ordinates in image dimensions.

    Args:
        deltas: Tensor of shape `(N, 4)` giving edge deltas to apply to locations.
        locations: Locations to apply deltas on. shape: `(N, 4)`
        stride: Stride of the FPN feature map.

    Returns:
        torch.Tensor
            Same shape as deltas and locations, giving co-ordinates of the
            resulting boxes `(x1, y1, x2, y2)`, absolute in image dimensions.
    """
    ##########################################################################
    # TODO: Implement the transformation logic to get boxes.                 #
    #                                                                        #
    # NOTE: The model predicted deltas MAY BE negative, which is not valid   #
    # for our use-case because the feature center must lie INSIDE the final  #
    # box. Make sure to clip them to zero.                                   #
    ##########################################################################

    # Un-normalize the deltas with the stride
    deltas = deltas * stride

    # Clip negative deltas to zero
    deltas = torch.clamp(deltas, min=0)

    # Compute the boxes using the deltas and locations
    x1 = locations[:, 0] - deltas[:, 0]  # x1 = xc - l
    y1 = locations[:, 1] - deltas[:, 1]  # y1 = yc - t
    x2 = locations[:, 0] + deltas[:, 2]  # x2 = xc + r
    y2 = locations[:, 1] + deltas[:, 3]  # y2 = yc + b

    output_boxes = torch.stack([x1, y1, x2, y2], dim=1)

    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return output_boxes



def fcos_make_centerness_targets(deltas: torch.Tensor):
    """
    Given LTRB deltas of GT boxes, compute GT targets for supervising the
    centerness regression predictor. See `fcos_get_deltas_from_locations` on
    how deltas are computed. If GT boxes are "background" => deltas are
    `(-1, -1, -1, -1)`, then centerness should be `-1`.

    For reference, centerness equation is available in FCOS paper
    https://arxiv.org/abs/1904.01355 (Equation 3).

    Args:
        deltas: Tensor of shape `(N, 4)` giving LTRB deltas for GT boxes.

    Returns:
        torch.Tensor
            Tensor of shape `(N, )` giving centerness regression targets.
    """
    ##########################################################################
    # TODO: Implement the centerness calculation logic.                      #
    ##########################################################################
    # Replace "pass" statement with your code
    # Get the L, T, R, B deltas
    left = deltas[:, 0]
    top = deltas[:, 1]
    right = deltas[:, 2]
    bottom = deltas[:, 3]

    # Calculate the centerness targets
    centerness = torch.sqrt(
        (torch.min(left, right) / torch.max(left, right)) * 
        (torch.min(top, bottom) / torch.max(top, bottom))
    )

    # Set centerness to -1 for background (deltas are -1)
    centerness[deltas[:, 0] == -1] = -1

    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return centerness



class FCOS(nn.Module):
    """
    FCOS: Fully-Convolutional One-Stage Detector

    This class puts together everything you implemented so far. It contains a
    backbone with FPN, and prediction layers (head). It computes loss during
    training and predicts boxes during inference.
    """

    def __init__(
        self, num_classes: int, fpn_channels: int, stem_channels: List[int]
    ):
        super().__init__()
        self.num_classes = num_classes

        ######################################################################
        # TODO: Initialize backbone and prediction network using arguments.  #
        ######################################################################
        # Replace "pass" statement with your code

        self.backbone = DetectorBackboneWithFPN(out_channels = fpn_channels)
        self.pred_net = FCOSPredictionNetwork(self.num_classes, fpn_channels, stem_channels)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Averaging factor for training loss; EMA of foreground locations.
        # STUDENTS: See its use in `forward` when you implement losses.
        self._normalizer = 150  # per image

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        Args:
            images: Batch of images, tensors of shape `(B, C, H, W)`.
            gt_boxes: Batch of training boxes, tensors of shape `(B, N, 5)`.
                `gt_boxes[i, j] = (x1, y1, x2, y2, C)` gives information about
                the `j`th object in `images[i]`. The position of the top-left
                corner of the box is `(x1, y1)` and the position of bottom-right
                corner of the box is `(x2, x2)`. These coordinates are
                real-valued in `[H, W]`. `C` is an integer giving the category
                label for this bounding box. Not provided during inference.
            test_score_thresh: During inference, discard predictions with a
                confidence score less than this value. Ignored during training.
            test_nms_thresh: IoU threshold for NMS during inference. Ignored
                during training.

        Returns:
            Losses during training and predictions during inference.
        """

        ######################################################################
        # TODO: Process the image through backbone, FPN, and prediction head #
        # to obtain model predictions at every FPN location.                 #
        # Get dictionaries of keys {"p3", "p4", "p5"} giving predicted class #
        # logits, deltas, and centerness.                                    #
        ######################################################################
        # Replace "pass" statement with your code
        feats_per_fpn_level = self.backbone.forward(images)
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = self.pred_net.forward(feats_per_fpn_level)

        fpn_feats_shapes = {
            level_name: feat.shape for level_name, feat in feats_per_fpn_level.items()
        }
        ######################################################################
        # TODO: Get absolute co-ordinates `(xc, yc)` for every location in
        # FPN levels.
        #
        # HINT: You have already implemented everything, just have to
        # call the functions properly.
        ######################################################################
        # Replace "pass" statement with your code
        locations_per_fpn_level = get_fpn_location_coords(fpn_feats_shapes, self.backbone.fpn_strides, device = images.device)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        if not self.training:
            # During inference, just go to this method and skip rest of the
            # forward pass.
            # fmt: off
            return self.inference(
                images, locations_per_fpn_level,
                pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )
            # fmt: on

        ######################################################################
        # TODO: Assign ground-truth boxes to feature locations. We have this
        # implemented in a `fcos_match_locations_to_gt`. This operation is NOT
        # batched so call it separately per GT boxes in batch.
        ######################################################################
        # List of dictionaries with keys {"p3", "p4", "p5"} giving matched
        # boxes for locations per FPN level, per image. Fill this list:
        matched_gt_boxes = []
        # Replace "pass" statement with your code
        for i in range(gt_boxes.shape[0]):
            boxes_per_fpn_level = fcos_match_locations_to_gt(
                locations_per_fpn_level, 
                self.backbone.fpn_strides, 
                gt_boxes[i]
            )
            matched_gt_boxes.append(boxes_per_fpn_level)

        # Calculate GT deltas for these matched boxes. Similar structure
        # as `matched_gt_boxes` above. Fill this list:
        matched_gt_deltas = []
        # Replace "pass" statement with your code
        for boxes_per_fpn_level in matched_gt_boxes:
            gt_deltas_per_level = {}
            for level in boxes_per_fpn_level.keys():       
                gt_deltas_per_level[level] = fcos_get_deltas_from_locations(locations_per_fpn_level[level], 
                                                                                 boxes_per_fpn_level[level], 
                                                                                 self.backbone.fpn_strides[level])

            matched_gt_deltas.append(gt_deltas_per_level)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Collate lists of dictionaries, to dictionaries of batched tensors.
        # These are dictionaries with keys {"p3", "p4", "p5"} and values as
        # tensors of shape (batch_size, locations_per_fpn_level, 5 or 4)
        matched_gt_boxes = default_collate(matched_gt_boxes)
        matched_gt_deltas = default_collate(matched_gt_deltas)

        # Combine predictions and GT from across all FPN levels.
        # shape: (batch_size, num_locations_across_fpn_levels, ...)
        matched_gt_boxes = self._cat_across_fpn_levels(matched_gt_boxes)
        matched_gt_deltas = self._cat_across_fpn_levels(matched_gt_deltas)
        pred_cls_logits = self._cat_across_fpn_levels(pred_cls_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)
        pred_ctr_logits = self._cat_across_fpn_levels(pred_ctr_logits)

        # Perform EMA update of normalizer by number of positive locations.
        num_pos_locations = (matched_gt_boxes[:, :, 4] != -1).sum()
        pos_loc_per_image = num_pos_locations.item() / images.shape[0]
        self._normalizer = 0.9 * self._normalizer + 0.1 * pos_loc_per_image

        #######################################################################
        # TODO: Calculate losses per location for classification, box reg and
        # centerness. Remember to set box/centerness losses for "background"
        # positions to zero.
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        loss_cls, loss_box, loss_ctr = None, None, None

        # Replace "pass" statement with your code
        # refer to loss functions
        indices = matched_gt_boxes[:, :, -1].to(torch.int64) + 1
        one_hot_index = torch.cat((torch.zeros(1, self.num_classes), torch.eye(self.num_classes)), dim = 0).to(device=indices.device)
        gt_classes = one_hot_index[indices]
        loss_cls = sigmoid_focal_loss(pred_cls_logits, gt_classes)
        loss_box = 0.25 * F.l1_loss(pred_boxreg_deltas, matched_gt_deltas, reduction="none")
        loss_box[matched_gt_deltas < 0] *= 0.0
        centerness = fcos_make_centerness_targets(matched_gt_deltas.view(-1, 4))
        loss_ctr = F.binary_cross_entropy_with_logits(pred_ctr_logits.flatten(), centerness, reduction="none")
        loss_ctr[centerness < 0] *= 0.0
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################
        # Sum all locations and average by the EMA of foreground locations.
        # In training code, we simply add these three and call `.backward()`
        return {
            "loss_cls": loss_cls.sum() / (self._normalizer * images.shape[0]),
            "loss_box": loss_box.sum() / (self._normalizer * images.shape[0]),
            "loss_ctr": loss_ctr.sum() / (self._normalizer * images.shape[0]),
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(
        self,
        images: torch.Tensor,
        locations_per_fpn_level: Dict[str, torch.Tensor],
        pred_cls_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        pred_ctr_logits: Dict[str, torch.Tensor],
        test_score_thresh: float = 0.3,
        test_nms_thresh: float = 0.5,
    ):
        """
        Run inference on a single input image (batch size = 1). Other input
        arguments are same as those computed in `forward` method. This method
        should not be called from anywhere except from inside `forward`.

        Returns:
            Three tensors:
                - pred_boxes: Tensor of shape `(N, 4)` giving *absolute* XYXY
                  co-ordinates of predicted boxes.

                - pred_classes: Tensor of shape `(N, )` giving predicted class
                  labels for these boxes (one of `num_classes` labels). Make
                  sure there are no background predictions (-1).

                - pred_scores: Tensor of shape `(N, )` giving confidence scores
                  for predictions: these values are `sqrt(class_prob * ctrness)`
                  where class_prob and ctrness are obtained by applying sigmoid
                  to corresponding logits.
        """

        # Gather scores and boxes from all FPN levels in this list. Once
        # gathered, we will perform NMS to filter highly overlapping predictions.
        pred_boxes_all_levels = []
        pred_classes_all_levels = []
        pred_scores_all_levels = []

        for level_name in locations_per_fpn_level.keys():

            # Get locations and predictions from a single level.
            # We index predictions by `[0]` to remove batch dimension.
            level_locations = locations_per_fpn_level[level_name]
            level_cls_logits = pred_cls_logits[level_name][0]
            level_deltas = pred_boxreg_deltas[level_name][0]
            level_ctr_logits = pred_ctr_logits[level_name][0]

            ##################################################################
            # TODO: FCOS uses the geometric mean of class probability and
            # centerness as the final confidence score. This helps in getting
            # rid of excessive amount of boxes far away from object centers.
            # Compute this value here (recall sigmoid(logits) = probabilities)
            #
            # Then perform the following steps in order:
            #   1. Get the most confidently predicted class and its score for
            #      every box. Use level_pred_scores: (N, num_classes) => (N, )
            #   2. Only retain prediction that have a confidence score higher
            #      than provided threshold in arguments.
            #   3. Obtain predicted boxes using predicted deltas and locations
            #   4. Clip XYXY box-cordinates that go beyond thr height and
            #      and width of input image.
            ##################################################################
            # Feel free to delete this line: (but keep variable names same)
            level_pred_boxes, level_pred_classes, level_pred_scores = (
                None,
                None,
                None,  # Need tensors of shape: (N, 4) (N, ) (N, )
            )

            # Compute geometric mean of class logits and centerness:
            level_pred_scores = torch.sqrt(
                level_cls_logits.sigmoid_() * level_ctr_logits.sigmoid_()
            )
            # Step 1:
            # Replace "pass" statement with your code
            level_pred_scores, indices = torch.max(level_pred_scores, dim = 1)

            # Step 2:
            # Replace "pass" statement with your code
            keep_indices = (level_pred_scores > test_score_thresh).nonzero()
            level_pred_scores = level_pred_scores[keep_indices].flatten()
            level_pred_classes = indices[keep_indices].flatten()

            # Step 3:
            # Replace "pass" statement with your code
            level_pred_boxes = fcos_apply_deltas_to_locations(level_deltas[keep_indices].reshape(-1, 4), 
                                                              level_locations[keep_indices].reshape(-1, 2), 
                                                              self.backbone.fpn_strides[level_name])

            # Step 4: Use `images` to get (height, width) for clipping.
            # Replace "pass" statement with your code
            level_pred_boxes[:, 0] = level_pred_boxes[:, 0].clip(min=0)    # start from 0
            level_pred_boxes[:, 1] = level_pred_boxes[:, 1].clip(min=0)    # start from 0
            level_pred_boxes[:, 2] = level_pred_boxes[:, 2].clip(max=images.shape[2])
            level_pred_boxes[:, 3] = level_pred_boxes[:, 3].clip(max=images.shape[3])
            ##################################################################
            #                          END OF YOUR CODE                      #
            ##################################################################

            pred_boxes_all_levels.append(level_pred_boxes)
            pred_classes_all_levels.append(level_pred_classes)
            pred_scores_all_levels.append(level_pred_scores)

        ######################################################################
        # Combine predictions from all levels and perform NMS.
        pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
        pred_classes_all_levels = torch.cat(pred_classes_all_levels)
        pred_scores_all_levels = torch.cat(pred_scores_all_levels)

        # STUDENTS: This function depends on your implementation of NMS.
        keep = class_spec_nms(
            pred_boxes_all_levels,
            pred_scores_all_levels,
            pred_classes_all_levels,
            iou_threshold=test_nms_thresh,
        )
        pred_boxes_all_levels = pred_boxes_all_levels[keep]
        pred_classes_all_levels = pred_classes_all_levels[keep]
        pred_scores_all_levels = pred_scores_all_levels[keep]
        return (
            pred_boxes_all_levels,
            pred_classes_all_levels,
            pred_scores_all_levels,
        )



#######################################################
#         Implement Backbone using ResNet50           #
#######################################################
        
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
          