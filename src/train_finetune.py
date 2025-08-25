import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection

from utils_dota import get_transform, collate_fn, get_model, select_k_shot_indices
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import nms
from pycocotools.cocoeval import COCOeval

def evaluate(model, data_loader, device, iou_thresh=0.5):
    model.eval()
    coco_gt = data_loader.dataset.coco
    coco_results = []

    total_gt = 0
    total_correct = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"])
                boxes = output["boxes"].cpu()
                scores = output["scores"].cpu()
                labels = output["labels"].cpu()

                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()

                total_gt += len(gt_boxes)

                # --- Tính accuracy ---
                matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
                for box, score, label in zip(boxes, scores, labels):
                    if score < 0.5:  # bỏ dự đoán yếu
                        continue
                    ious = torchvision.ops.box_iou(box.unsqueeze(0), gt_boxes)[0]  # IoU với tất cả GT
                    max_iou, max_idx = ious.max(0)
                    if max_iou >= iou_thresh and label == gt_labels[max_idx] and not matched[max_idx]:
                        total_correct += 1
                        matched[max_idx] = True

                    coco_results.append({
                        "image_id": image_id,
                        "category_id": data_loader.dataset.coco.getCatIds()[label-1],
                        "bbox": [
                            float(box[0]), float(box[1]),
                            float(box[2] - box[0]), float(box[3] - box[1]),
                        ],
                        "score": float(score),
                    })

    # Tính mAP bằng COCOeval
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # In ra accuracy
    accuracy = total_correct / max(1, total_gt)
    print(f"Classification Accuracy (IoU>{iou_thresh}): {accuracy:.4f}")
    return accuracy




# ===================== Dataset Custom ====================
class CocoDetectionCustom(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile)
        self.transforms = transforms

        # ánh xạ category_id (gốc COCO) -> nhãn liên tục [1..N]
        cat_ids = self.coco.getCatIds()
        self.catid2label = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)

        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            xmin, ymin, w, h = ann["bbox"]
            xmax, ymax = xmin + w, ymin + h
            boxes.append([xmin, ymin, xmax, ymax])

            # map category_id -> chỉ số liên tục
            labels.append(self.catid2label[ann["category_id"]])

            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        # apply transform (img, target)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


# ===================== Main =====================
def main(args):
    # ==== Load dataset paths ====
    data_root = args.data_root
    train_ann = os.path.join(data_root, 'annotations', 'instances_train2017.json')
    val_ann = os.path.join(data_root, 'annotations', 'instances_val2017.json')

    train_img_root = os.path.join(data_root, 'train')
    val_img_root = os.path.join(data_root, 'val')

    # ==== K-shot sampling ====
    coco = COCO(train_ann)
    chosen_ids = select_k_shot_indices(coco, args.k, seed=args.seed)

    train_dataset = CocoDetectionCustom(
        root=train_img_root, annFile=train_ann,
        transforms=get_transform(train=True)
    )
    val_dataset = CocoDetectionCustom(
        root=val_img_root, annFile=val_ann,
        transforms=get_transform(train=False)
    )

    # map image ids -> dataset indices
    imgid_to_index = {img['id']: idx for idx, img in enumerate(train_dataset.coco.loadImgs(train_dataset.ids))}
    chosen_indices = [imgid_to_index[i] for i in chosen_ids if i in imgid_to_index]

    print(f"==> Total train images: {len(train_dataset)}")
    print(f"==> Subset chosen: {len(chosen_indices)}")

    if len(chosen_indices) == 0:
        raise ValueError("Subset rỗng! Kiểm tra lại select_k_shot_indices.")

    # tạo subset few-shot
    subset = Subset(train_dataset, chosen_indices)

    # ==== Model ====
    num_classes = len(train_dataset.coco.getCatIds()) + 1  # +1 background
    model = get_model(num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ==== DataLoader ====
    data_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    data_loader_val = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # ==== Optimizer ====
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0001)

    # ==== Training loop ====
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            if step % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(data_loader)}] "
                      f"Loss: {losses.item():.4f}")

        epoch_loss = running_loss / max(1, len(data_loader))
        print(f"[Epoch {epoch}] Avg Loss: {epoch_loss:.4f}")
        avg_loss = running_loss / len(data_loader)
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")

        os.makedirs('outputs', exist_ok=True)
        torch.save(model.state_dict(), f'outputs/model_epoch_{epoch}.pth')

    # save final model
    torch.save(model.state_dict(), 'outputs/model_final.pth')
    print("Training finished. Final model saved to outputs/model_final.pth")
    print("Evaluating final model...")
    evaluate(model, data_loader_val, device, iou_thresh=0.5)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='data/COCO_TINY')
    parser.add_argument('--k', type=int, default=5, help="Number of shots per class")
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    main(args)
