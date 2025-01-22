import cv2
import numpy as np
import pandas as pd
import os
import torch
from LightGlue.lightglue import LightGlue, SuperPoint, DISK
from LightGlue.lightglue.utils import load_image, rbd
import tifffile
from pathlib import Path
import sys
import torch.nn as nn
import torchvision
import torchvision.models as models
import argparse
from collections import OrderedDict
import torch.nn.functional as F

# Add RT-DETR path
current_dir = os.path.dirname(os.path.abspath(__file__))
rtdetr_root = os.path.join(current_dir, 'RT-DETR/rtdetr_pytorch')
sys.path.append(rtdetr_root)

from src.nn.backbone.presnet import PResNet
from src.nn.backbone.common import ConvNormLayer
from src.zoo.rtdetr.rtdetr import RTDETR
from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
from src.zoo.rtdetr.rtdetr_criterion import SetCriterion
from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
from src.zoo.rtdetr.matcher import HungarianMatcher

def test_backbone(model, device='cuda'):
    """Test the backbone with a random tensor."""
    # Create a random batch of 4-channel images (B, C, H, W)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 4, 640, 640).to(device)
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
        print(f"\nBackbone output shape: {output.shape}")
    
    return output

def read_multispectral_image(base_path, image_name, split='test'):
    """Read all channels of a multispectral image."""
    split_dir = f"{split.capitalize()}_Images"
    channels = {
        'Red': os.path.join(base_path, 'Red_Channel', split_dir, image_name),
        'Green': os.path.join(base_path, 'Green_Channel', split_dir, image_name),
        'RedEdge': os.path.join(base_path, 'Red_Edge_Channel', split_dir, image_name),
        'NIR': os.path.join(base_path, 'Near_Infrared_Channel', split_dir, image_name)
    }
    
    images = {}
    for channel, path in channels.items():
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image from {path}")
        images[channel] = img
        
    return images

def align_to_reference(reference_img, target_img, extractor, matcher, device):
    """Align target image to reference image using LightGlue."""
    # Convert grayscale to 3-channel and normalize to [0,1]
    ref_3ch = cv2.merge([reference_img] * 3) / 255.0
    target_3ch = cv2.merge([target_img] * 3) / 255.0
    
    # Convert to torch tensors and move to device
    ref_tensor = torch.from_numpy(ref_3ch).float().permute(2, 0, 1).to(device)
    target_tensor = torch.from_numpy(target_3ch).float().permute(2, 0, 1).to(device)
    
    # Extract features
    feats0 = extractor.extract(ref_tensor)
    feats1 = extractor.extract(target_tensor)
    
    # Match features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    
    # Get matching keypoints
    matches = matches01['matches']
    kpts0 = feats0['keypoints'][matches[..., 0]]
    kpts1 = feats1['keypoints'][matches[..., 1]]
    
    # Find homography
    H, _ = cv2.findHomography(
        srcPoints=kpts1.cpu().numpy(),
        dstPoints=kpts0.cpu().numpy(),
        method=cv2.USAC_ACCURATE,
        maxIters=1000000,
        confidence=0.999
    )
    
    if H is None:
        raise ValueError("Could not find homography matrix")
    
    # Warp image
    height, width = reference_img.shape
    aligned_img = cv2.warpPerspective(target_img, H, (width, height))
    
    return aligned_img

def align_and_save_visualization(image_path, output_dir, extractor, matcher, device, split='test'):
    """Align a single image and save visualization of the results."""
    images = read_multispectral_image(image_path, os.path.basename(image_path), split)
    
    # Use Green channel as reference
    reference_img = images['Green']
    
    # Align other channels to Green
    aligned_images = {'Green': reference_img}
    for channel in ['Red', 'NIR', 'RedEdge']:
        print(f"Aligning {channel} channel...")
        aligned_images[channel] = align_to_reference(
            reference_img, 
            images[channel], 
            extractor, 
            matcher, 
            device
        )
    
    # Save individual aligned channels as JPG for visualization
    for channel, img in aligned_images.items():
        output_path = os.path.join(output_dir, f'aligned_{channel}.jpg')
        cv2.imwrite(output_path, img)
        print(f"Saved aligned {channel} channel to {output_path}")
    
    return aligned_images

def create_aligned_dataset(workspace_root, split='test'):
    """Create aligned dataset for either test or train split."""
    print(f"Processing {split} set...")
    
    # Setup paths
    base_path = os.path.join(workspace_root, 'multispectral/Spectral_Images')
    output_base = os.path.join(workspace_root, 'multispectral/aligned', split)
    os.makedirs(output_base, exist_ok=True)
    
    # Initialize LightGlue
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)
    
    # Get list of images
    split_dir = f"{split.capitalize()}_Images"
    image_dir = os.path.join(base_path, 'Green_Channel', split_dir)
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    for idx, image_name in enumerate(image_files):
        print(f"Processing image {idx+1}/{len(image_files)}: {image_name}")
        try:
            # Read and align images
            images = read_multispectral_image(base_path, image_name, split)
            reference_img = images['Green']
            
            aligned_images = {'Green': reference_img}
            for channel in ['Red', 'NIR', 'RedEdge']:
                print(f"  Aligning {channel} channel...")
                aligned_images[channel] = align_to_reference(
                    reference_img,
                    images[channel],
                    extractor,
                    matcher,
                    device
                )
            
            # Stack channels in correct order: green, red, nir, rededge
            stacked_image = np.stack([
                aligned_images['Green'],
                aligned_images['Red'],
                aligned_images['NIR'],
                aligned_images['RedEdge']
            ], axis=0)
            
            # Save as TIFF
            output_name = f"{Path(image_name).stem}.tif"
            output_path = os.path.join(output_base, output_name)
            tifffile.imwrite(output_path, stacked_image)
            print(f"  Saved aligned image to {output_path}")
            
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
            continue

def load_rtdetr_model(num_channels=4, num_classes=2):
    # Initialize ResNet18 backbone with 4 input channels
    model = PResNet(depth=18, variant='d', pretrained=False)
    
    # Modify first conv layer to accept 4 channels
    if num_channels != 3:
        # Get the original conv1 parameters
        conv_def = [
            [num_channels, 32, 3, 2, "conv1_1"],
            [32, 32, 3, 1, "conv1_2"],
            [32, 64, 3, 1, "conv1_3"],
        ]
        
        # Create new conv1 with 4 channels
        model.conv1 = nn.Sequential(OrderedDict([
            (_name, ConvNormLayer(c_in, c_out, k, s, act='relu')) 
            for c_in, c_out, k, s, _name in conv_def
        ]))
    
    return model

def load_and_preprocess_tiff(image_path):
    """Load and preprocess a 4-channel TIFF image for the model."""
    # Load TIFF image
    img = tifffile.imread(image_path)
    
    # Convert to float32 
    img = img.astype(np.float32) / 255.0
    
    # Convert to torch tensor and add batch dimension
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)  # Add batch dimension
    
    return img

class MultispectralDataset(torch.utils.data.Dataset):
    def __init__(self, workspace_root, split='train', validation=False):
        self.workspace_root = workspace_root
        self.split = split
        self.img_dir = os.path.join(workspace_root, 'multispectral/aligned', 'train' if split == 'train' else 'test')
        
        # Load labels
        if split == 'train' or split == 'val':
            label_file = os.path.join(workspace_root, 'multispectral/Spectral_Images/Labels/Train_Labels_CSV.csv')
        else:
            label_file = os.path.join(workspace_root, 'multispectral/Spectral_Images/Labels/Test_labels_CSV.csv')
            
        self.labels_df = pd.read_csv(label_file)
        
        # Group annotations by image
        self.annotations = {}
        for _, row in self.labels_df.iterrows():
            img_name = Path(row['filename']).stem
            if img_name not in self.annotations:
                self.annotations[img_name] = {'boxes': [], 'labels': []}
            
            # Convert coordinates to normalized format
            xmin, ymin, xmax, ymax = map(float, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            
            # Skip invalid boxes
            if xmin >= xmax or ymin >= ymax:
                print(f"Warning: Invalid box in {img_name}: [{xmin}, {ymin}, {xmax}, {ymax}]")
                continue
                
            if xmax > 416 or ymax > 416:
                print(f"Warning: Box coordinates exceed image size in {img_name}")
                xmax = min(xmax, 416)
                ymax = min(ymax, 416)
            
            # Normalize coordinates
            xmin, xmax = xmin / 416.0, xmax / 416.0
            ymin, ymax = ymin / 416.0, ymax / 416.0
            
            # Convert to center format
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin
            
            self.annotations[img_name]['boxes'].append([cx, cy, w, h])
            self.annotations[img_name]['labels'].append(1 if row['class'].lower().strip() in ['stressed', 'st'] else 0)
        
        # Get list of all images
        all_images = sorted(list(set([Path(f).stem for f in os.listdir(self.img_dir) if f.endswith('.tif')])))
        
        # Split train into train/val if needed
        if split == 'train' or split == 'val':
            np.random.seed(42)  # For reproducibility
            total_images = len(all_images)
            train_size = int(0.7 * total_images)
            print(f"Total images: {total_images}, Train size: {train_size}, Val size: {total_images - train_size}")
            
            # Shuffle all images first
            all_images = np.array(all_images)
            np.random.shuffle(all_images)
            
            # Split into train and val
            self.image_ids = all_images[:train_size] if split == 'train' else all_images[train_size:]
        else:
            self.image_ids = all_images
        
        print(f"Loaded {len(self.image_ids)} images for {split} split")
        
        # Print some statistics
        num_boxes = sum(len(self.annotations[img_id]['boxes']) for img_id in self.image_ids)
        print(f"Total number of boxes: {num_boxes}")
        print(f"Average boxes per image: {num_boxes / len(self.image_ids):.1f}")
    
    def augment(self, image, boxes):
        """Simple augmentation: just horizontal flip."""
        if self.split != 'train' or len(boxes) == 0:
            return image, boxes
            
        # Random horizontal flip (50% chance)
        if torch.rand(1) > 0.5:
            image = torch.flip(image, [2])  # Flip width dimension
            boxes[:, 0] = 1 - boxes[:, 0]  # Flip x-coordinate of center
        
        return image, boxes
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.tif")
        
        # Load image and normalize to [0,1]
        img = tifffile.imread(img_path)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        
        # Get annotations
        ann = self.annotations[img_id]
        boxes = torch.tensor(ann['boxes'], dtype=torch.float32)
        labels = torch.tensor(ann['labels'], dtype=torch.long)
        
        # Apply augmentations if in training
        if self.split == 'train':
            img, boxes = self.augment(img, boxes)
        
        return img, boxes, labels

def build_rtdetr(num_classes=2):
    # Initialize backbone with some layers frozen
    backbone = PResNet(
        depth=18,
        variant='d',
        num_stages=4,
        return_idx=[1, 2, 3],
        freeze_at=1,  # Freeze first stage
        freeze_norm=True,  # Freeze batch norm
        pretrained=False
    )
    
    # Modify first conv layer for 4 channels
    conv_def = [
        [4, 32, 3, 2, "conv1_1"],
        [32, 32, 3, 1, "conv1_2"],
        [32, 64, 3, 1, "conv1_3"],
    ]
    backbone.conv1 = nn.Sequential(OrderedDict([
        (_name, ConvNormLayer(c_in, c_out, k, s, act='relu')) 
        for c_in, c_out, k, s, _name in conv_def
    ]))
    
    # For ResNet18 with return_idx=[1,2,3], the output channels are [128, 256, 512]
    encoder = HybridEncoder(
        in_channels=[128, 256, 512],  # Ascending order of channels
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        expansion=0.5,
        use_encoder_idx=[2]  # Only use encoder on the last feature map
    )
    
    # Initialize decoder with same hidden_dim as encoder
    decoder = RTDETRTransformer(
        hidden_dim=256,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        num_queries=300,
        num_levels=3,  # Matches number of feature levels from backbone
        num_decoder_points=4,
        num_classes=num_classes,
        feat_channels=[512, 256, 128],  # Descending order for decoder's input projection
        feat_strides=[32, 16, 8]  # Corresponding strides
    )
    
    # Build full model
    class RTDETRWrapper(RTDETR):
        def forward(self, x, targets=None):
            if self.multi_scale and self.training:
                sz = np.random.choice(self.multi_scale)
                x = F.interpolate(x, size=[sz, sz])
            
            # Get backbone features
            feats = self.backbone(x)
            
            # Apply encoder
            enc_feats = self.encoder(feats)
            
            # Reverse feature order for decoder
            feats = feats[::-1]  # Now [512, 256, 128]
            
            # Pass features and targets to decoder
            out = self.decoder(feats, targets)
            
            # Convert logits to float32 for focal loss
            if 'pred_logits' in out:
                out['pred_logits'] = out['pred_logits'].float()
            
            return out
    
    model = RTDETRWrapper(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder
    )
    
    return model

def collate_fn(batch):
    """Custom collate function to handle variable number of boxes and labels."""
    images = torch.stack([item[0] for item in batch])
    targets = []
    for _, boxes, labels in batch:
        targets.append({
            'boxes': boxes,
            'labels': labels  # Keep as long for indexing
        })
    return images, targets

def train_one_epoch(model, criterion, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Pass both images and targets to model
        outputs = model(images, targets)
        loss_dict = criterion(outputs, targets)
        
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() & weight_dict.keys())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        if batch_idx % 10 == 0:
            print(f'Batch [{batch_idx}/{len(data_loader)}], Loss: {losses.item():.4f}')
    
    return total_loss / len(data_loader)

class CustomCriterion(SetCriterion):
    def loss_labels_focal(self, outputs, targets, indices, num_boxes, **kwargs):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                  dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1].float()
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

def evaluate(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images, targets)
            loss_dict = criterion(outputs, targets)
            
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() & weight_dict.keys())
            total_loss += losses.item()
    
    return total_loss / len(data_loader)

def draw_predictions(image, boxes, labels, scores, targets, output_path):
    """Draw predicted and ground truth boxes on image and save it side by side."""
    # Convert to uint8 if float
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Create three images side by side (original, GT, predictions)
    h, w = image.shape
    combined_img = np.zeros((h, w * 3, 3), dtype=np.uint8)
    
    # Original image (left panel)
    original = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    combined_img[:, :w] = original
    
    # Ground truth visualization (middle panel)
    gt_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gt_boxes = targets['boxes'].cpu().numpy()
    gt_labels = targets['labels'].cpu().numpy()
    
    for box, label in zip(gt_boxes, gt_labels):
        # Convert from normalized [cx, cy, w, h] to pixel coordinates
        cx, cy, w_box, h_box = box
        xmin = int((cx - w_box/2) * 416)
        ymin = int((cy - h_box/2) * 416)
        xmax = int((cx + w_box/2) * 416)
        ymax = int((cy + h_box/2) * 416)
        
        # Color based on class
        color = (50, 255, 50) if label == 0 else (50, 50, 255)  # Green for healthy, Red for stressed
        
        # Draw dashed rectangle for ground truth
        for i in range(0, 416, 10):  # Draw dashed lines
            if xmin + i < xmax:
                cv2.line(gt_vis, (xmin + i, ymin), (min(xmin + i + 5, xmax), ymin), color, 2)
                cv2.line(gt_vis, (xmin + i, ymax), (min(xmin + i + 5, xmax), ymax), color, 2)
            if ymin + i < ymax:
                cv2.line(gt_vis, (xmin, ymin + i), (xmin, min(ymin + i + 5, ymax)), color, 2)
                cv2.line(gt_vis, (xmax, ymin + i), (xmax, min(ymin + i + 5, ymax)), color, 2)
        
        # Add GT label with black outline for visibility
        label_text = f"GT: {'healthy' if label == 0 else 'stressed'}"
        # Draw black outline
        cv2.putText(gt_vis, label_text, (xmin, ymin-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        # Draw text in color
        cv2.putText(gt_vis, label_text, (xmin, ymin-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    combined_img[:, w:2*w] = gt_vis
    
    # Predictions visualization (right panel)
    pred_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # First sort predictions by confidence
    if len(boxes) > 0:
        indices = np.argsort(scores)[::-1]  # Sort in descending order
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
    
    for box, label, score in zip(boxes, labels, scores):
        if score < 0.25:  # Skip low confidence predictions
            continue
            
        # Box is already in absolute coordinates [xmin, ymin, xmax, ymax]
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])
        
        # Ensure coordinates are within image bounds
        xmin = max(0, min(xmin, w-1))
        ymin = max(0, min(ymin, h-1))
        xmax = max(0, min(xmax, w-1))
        ymax = max(0, min(ymax, h-1))
        
        # Color based on class with high intensity
        color = (50, 255, 50) if label == 0 else (50, 50, 255)  # Bright green for healthy, bright red for stressed
        
        # Draw solid rectangle with thick border
        cv2.rectangle(pred_vis, (xmin, ymin), (xmax, ymax), (0, 0, 0), 4)  # Black outline
        cv2.rectangle(pred_vis, (xmin, ymin), (xmax, ymax), color, 2)  # Colored inner line
        
        # Add prediction label and score with black outline for visibility
        label_text = f"Pred: {'healthy' if label == 0 else 'stressed'} {score:.2f}"
        # Draw black outline
        cv2.putText(pred_vis, label_text, (xmin, ymin-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        # Draw text in color
        cv2.putText(pred_vis, label_text, (xmin, ymin-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    combined_img[:, 2*w:] = pred_vis
    
    # Add titles with black outline for visibility
    title_y = 30
    font_scale = 1
    thickness = 2
    
    # Function to draw outlined text
    def draw_outlined_text(img, text, pos, font_scale, color):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    draw_outlined_text(combined_img, "Original", (w//3, title_y), font_scale, (255, 255, 255))
    draw_outlined_text(combined_img, "Ground Truth", (w + w//3, title_y), font_scale, (255, 255, 255))
    draw_outlined_text(combined_img, "Predictions", (2*w + w//3, title_y), font_scale, (255, 255, 255))
    
    # Save combined image
    cv2.imwrite(output_path, combined_img)

def visualize_dataset(dataset, output_dir, num_samples=10):
    """Visualize random samples from the dataset with their ground truth boxes."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get random indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in indices:
        img, boxes, labels = dataset[idx]
        img_id = dataset.image_ids[idx]
        
        # Get green channel and convert to uint8
        green_channel = (img[0].numpy() * 255).astype(np.uint8)
        
        # Convert to 3-channel grayscale for OpenCV
        green_channel = cv2.cvtColor(green_channel, cv2.COLOR_GRAY2BGR)
        
        # Draw each ground truth box
        for box, label in zip(boxes, labels):
            # Convert from [cx, cy, w, h] to pixel coordinates
            cx, cy, w, h = box.numpy()
            xmin = int((cx - w/2) * 416)
            ymin = int((cy - h/2) * 416)
            xmax = int((cx + w/2) * 416)
            ymax = int((cy + h/2) * 416)
            
            # Draw rectangle in white
            cv2.rectangle(green_channel, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
            
            # Add label in white
            label_text = 'healthy' if label.item() == 0 else 'stressed'
            cv2.putText(green_channel, label_text, (xmin, ymin-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Convert back to grayscale for saving
        green_channel = cv2.cvtColor(green_channel, cv2.COLOR_BGR2GRAY)
        
        # Save visualization
        output_path = os.path.join(output_dir, f'{img_id}_gt.jpg')
        cv2.imwrite(output_path, green_channel)
        print(f"Saved visualization for {img_id}")
        
        # Also print the boxes and labels for verification
        print(f"Image {img_id} has {len(boxes)} boxes:")
        for box, label in zip(boxes, labels):
            cx, cy, w, h = box.numpy()
            label_text = 'healthy' if label.item() == 0 else 'stressed'
            print(f"- {label_text}: center=({cx:.3f}, {cy:.3f}), size=({w:.3f}, {h:.3f})")

def apply_nms(boxes, scores, labels, iou_threshold=0.45):
    """Apply Non-Maximum Suppression to remove overlapping boxes."""
    # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] if needed
    if boxes.shape[-1] == 4:
        if boxes.size > 0:
            if boxes[0][2] < 1:  # Check if first box width is normalized
                # If normalized coordinates, convert to absolute first
                boxes = boxes * 416
            
            # If center format, convert to corner format
            if boxes[0][2] < boxes[0][0]:  # Check if width < center_x (indicates center format)
                cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                boxes = np.column_stack([
                    cx - w/2, cy - h/2,
                    cx + w/2, cy + h/2
                ])
    
    # Apply NMS per class
    keep_boxes = []
    keep_scores = []
    keep_labels = []
    
    for class_id in np.unique(labels):
        class_mask = labels == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        
        # Compute areas
        x1 = class_boxes[:, 0]
        y1 = class_boxes[:, 1]
        x2 = class_boxes[:, 2]
        y2 = class_boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort boxes by score
        order = class_scores.argsort()[::-1]
        keep_indices = []
        
        while order.size > 0:
            i = order[0]
            keep_indices.append(i)
            
            if order.size == 1:
                break
                
            # Compute IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        
        keep_boxes.extend(class_boxes[keep_indices])
        keep_scores.extend(class_scores[keep_indices])
        keep_labels.extend([class_id] * len(keep_indices))
    
    return np.array(keep_boxes), np.array(keep_scores), np.array(keep_labels)

def main():
    parser = argparse.ArgumentParser()
    # Dataset creation arguments
    parser.add_argument('--create-aligned', action='store_true', help='Create aligned dataset for both train and test splits')
    
    # Training arguments
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--visualize-train', action='store_true', help='Visualize training data annotations')
    parser.add_argument('--vis-samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    # Model parameters
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--model-path', type=str, default='multispectral/best.pth', help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='multispectral/output', help='Output directory for results')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    workspace_root = '/home/appuser/object_detection'
    
    # Create aligned dataset if requested
    if args.create_aligned:
        print("Creating aligned datasets...")
        print("Processing training split...")
        create_aligned_dataset(workspace_root, split='train')
        print("\nProcessing test split...")
        create_aligned_dataset(workspace_root, split='test')
        return
        
    if args.visualize_train:
        print("Visualizing training data...")
        train_dataset = MultispectralDataset(workspace_root, split='train')
        output_dir = os.path.join(workspace_root, args.output_dir, 'train_vis')
        visualize_dataset(train_dataset, output_dir, args.vis_samples)
        return
        
    # Initialize datasets
    if args.train:
        train_dataset = MultispectralDataset(workspace_root, split='train')
        val_dataset = MultispectralDataset(workspace_root, split='val')
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )
        
        print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images")
        
        # Initialize model with frozen layers
        model = build_rtdetr(num_classes=args.num_classes)
        model = model.to(device)
        
        # Modified weight dictionary with balanced weights
        weight_dict = {
            'cost_class': 4.0,  # Increased class cost
            'cost_bbox': 5.0,
            'cost_giou': 2.0,
            'loss_focal': 4.0,  # Increased focal loss weight
            'loss_bbox': 5.0,
            'loss_giou': 2.0
        }
        
        matcher = HungarianMatcher(
            weight_dict=weight_dict,
            use_focal_loss=True
        )
        
        criterion = CustomCriterion(
            num_classes=args.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=['focal', 'boxes'],
            alpha=0.25,  # Focal loss alpha (balance between classes)
            gamma=2.0    # Focal loss gamma (focus on hard examples)
        )
        criterion = criterion.to(device)
        
        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.01  # Increased weight decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(args.epochs):
            print(f'Epoch [{epoch+1}/{args.epochs}]')
            
            # Train
            train_loss = train_one_epoch(
                model, criterion, optimizer, train_loader, device
            )
            
            # Validate
            val_loss = evaluate(model, criterion, val_loader, device)
            
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss
                }
                torch.save(
                    checkpoint,
                    os.path.join(workspace_root, args.model_path)
                )
                print(f'New best model saved! Val Loss: {val_loss:.4f}')
            else:
                patience_counter += 1
                
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss
                }
                torch.save(
                    checkpoint,
                    os.path.join(workspace_root, 'multispectral', f'rtdetr_epoch_{epoch+1}.pth')
                )
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f'Early stopping triggered! No improvement for {args.patience} epochs')
                print(f'Best validation loss was {best_val_loss:.4f} at epoch {best_epoch+1}')
                break
    
    if args.test:
        print("Running inference...")
        # Create output directory
        output_dir = os.path.join(workspace_root, args.output_dir, 'predictions')
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        model = build_rtdetr(num_classes=args.num_classes)
        checkpoint = torch.load(os.path.join(workspace_root, args.model_path))
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        model.eval()
        
        # Initialize test dataset
        test_dataset = MultispectralDataset(workspace_root, split='test')
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Initialize post-processor
        postprocessor = RTDETRPostProcessor(
            num_classes=args.num_classes,
            use_focal_loss=True,
            num_top_queries=300
        )
        
        # Run inference
        total_predictions = 0
        predictions_above_threshold = 0
        
        for idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            img_id = test_dataset.image_ids[idx]
            
            with torch.no_grad():
                outputs = model(images)
                results = postprocessor(outputs, torch.tensor([[416.0, 416.0]]).to(device))
                
                for img_results in results:
                    pred_boxes = img_results['boxes'].cpu().numpy()
                    pred_labels = img_results['labels'].cpu().numpy()
                    scores = img_results['scores'].cpu().numpy()
                    
                    # Get all predictions and apply NMS
                    all_predictions = list(zip(pred_boxes, pred_labels, scores))
                    all_predictions.sort(key=lambda x: x[2], reverse=True)  # Sort by confidence
                    
                    if len(all_predictions) > 0:
                        pred_boxes, pred_labels, scores = zip(*all_predictions)
                        pred_boxes = np.array(pred_boxes)
                        pred_labels = np.array(pred_labels)
                        scores = np.array(scores)
                        
                        # Apply NMS to remove overlapping boxes
                        pred_boxes, scores, pred_labels = apply_nms(pred_boxes, scores, pred_labels)
                    else:
                        pred_boxes = np.array([])
                        pred_labels = np.array([])
                        scores = np.array([])
                    
                    total_predictions += len(scores)
                    predictions_above_threshold += np.sum(scores > 0.25)
                    
                    # Get green channel from original image
                    green_channel = images[0, 0].cpu().numpy()
                    
                    # Draw predictions and ground truth boxes
                    output_path = os.path.join(output_dir, f'{img_id}_pred.jpg')
                    draw_predictions(green_channel, pred_boxes, pred_labels, scores, targets[0], output_path)
                    
                    print(f"\nPredictions for image {img_id}:")
                    print(f"Found {len(pred_boxes)} predictions, {np.sum(scores > 0.25)} above threshold")
                    
                    # Print predictions above threshold
                    for box, label, score in zip(pred_boxes, pred_labels, scores):
                        if score > 0.25:
                            class_name = 'healthy' if label == 0 else 'stressed'
                            print(f"Class: {class_name}, Score: {score:.3f}, Box: {box}")
        
        print("\nOverall Statistics:")
        print(f"Total predictions: {total_predictions}")
        print(f"Predictions above threshold (0.25): {predictions_above_threshold}")
        print(f"Average predictions per image: {predictions_above_threshold/len(test_dataset):.1f}")

if __name__ == '__main__':
    main()


