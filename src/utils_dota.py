import json
import random
import torch
import torchvision
import torchvision.transforms.functional as F
from pycocotools.coco import COCO


# -----------------------------
# Custom Transform Wrapper
# -----------------------------
class ComposeWithTarget:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        # chỉ convert khi còn là PIL
        if not isinstance(image, torch.Tensor):
            image = F.to_tensor(image)
        return image, target
#

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            # check loại ảnh
            if isinstance(image, torch.Tensor):
                _, h, w = image.shape
                image = torch.flip(image, dims=[2])  # flip ngang (W)
            else:
                w, h = image.size
                image = F.hflip(image)

            # flip box trong target
            if "boxes" in target:
                boxes = target["boxes"]
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target

def get_transform(train=True):
    transforms = []
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    transforms.append(ToTensor())
    return ComposeWithTarget(transforms)


# -----------------------------
# COCO Utilities
# -----------------------------
def load_coco(ann_file):
    """Load COCO annotations."""
    return COCO(ann_file)


def write_subset_coco(original_coco_ann, out_ann_file, chosen_image_ids):
    """Create subset COCO annotation file with only chosen IDs."""
    with open(original_coco_ann, 'r') as f:
        data = json.load(f)

    id_set = set(chosen_image_ids)
    imgs = [img for img in data['images'] if img['id'] in id_set]
    anns = [ann for ann in data['annotations'] if ann['image_id'] in id_set]

    # filter categories only present in subset
    present_cats = {ann['category_id'] for ann in anns}
    cats = [c for c in data['categories'] if c['id'] in present_cats]

    out = {
        'images': imgs,
        'annotations': anns,
        'categories': cats
    }
    with open(out_ann_file, 'w') as f:
        json.dump(out, f)


# -----------------------------
# Training Utilities
# -----------------------------
def collate_fn(batch):
    """Custom collate function for dataloader."""
    return tuple(zip(*batch))


def get_model(num_classes):
    """Load Faster R-CNN ResNet50 model with new head."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model


def select_k_shot_indices(coco, k, seed=42):
    """Randomly sample k image IDs from COCO."""
    random.seed(seed)
    ids = list(coco.imgs.keys())
    return random.sample(ids, k)
