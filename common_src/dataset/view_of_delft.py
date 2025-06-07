import os
import sys
import os.path as osp
root = os.path.abspath(os.path.join(os.getcwd()))
if root not in sys.path:
    sys.path.insert(0, root)
    
import numpy as np
from common_src.model.utils import LiDARInstance3DBoxes

import torch
from torch.utils.data import Dataset

from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix, homogeneous_transformation

from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_mobilenet_v3_large
from torchvision import transforms

from PIL import Image
import time

# Added for testing
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

class ViewOfDelft(Dataset):
    CLASSES = ['Car', 
               'Pedestrian', 
               'Cyclist',]
            #    'rider', 
            #    'unused_bicycle', 
            #    'bicycle_rack', 
            #    'human_depiction', 
            #    'moped_or_scooter', 
            #    'motor',
            #    'truck',
            #    'other_ride',
            #    'other_vehicle',
            #    'uncertain_ride'
    
    LABEL_MAPPING = {
        'class': 0, # Describes the type of object: 'Car', 'Pedestrian', 'Cyclist', etc.
        'truncated': 1, # Not used, only there to be compatible with KITTI format.
        'occluded': 2, # Integer (0,1,2) indicating occlusion state 0 = fully visible, 1 = partly occluded 2 = largely occluded.
        'alpha': 3, # Observation angle of object, ranging [-pi..pi]
        'bbox2d': slice(4,8),
        'bbox3d_dimensions': slice(8,11), # 3D object dimensions: height, width, length (in meters).
        'bbox3d_location': slice(11,14), # 3D object location x,y,z in camera coordinates (in meters).
        'bbox3d_rotation': 14, # Rotation around -Z-axis in LiDAR coordinates [-pi..pi].
    }
    
    def __init__(self, 
                 data_root = 'data/view_of_delft', 
                 sequential_loading=False,
                 split = 'train',
                 segmentation_generation=False):
        super().__init__()
        
        self.data_root = data_root
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}. Must be one of ['train', 'val', 'test']"
        self.split = split
        self.segmentation_generation = segmentation_generation
        split_file = os.path.join(data_root, 'lidar', 'ImageSets', f'{split}.txt')

        with open(split_file, 'r') as f:
            lines = f.readlines()
            self.sample_list = [line.strip() for line in lines]
        
        self.vod_kitti_locations = KittiLocations(root_dir = data_root)

        # Loading the segmentation model
        self.seg_model = deeplabv3_mobilenet_v3_large(pretrained=True).eval()
        self.seg_transform = transforms.Compose([
            transforms.Resize((512, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std =[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        num_frame = self.sample_list[idx]
        vod_frame_data = FrameDataLoader(kitti_locations=self.vod_kitti_locations,
                                        frame_number=num_frame)
        local_transforms = FrameTransformMatrix(vod_frame_data)

        lidar_data = vod_frame_data.lidar_data
        image_data = vod_frame_data.image

        # Get segmentation scores
        if self.split == "train" and not self.segmentation_generation:
            seg_path = os.path.join("common_src/dataset/sem_cache", f"{num_frame}.npz")
            raw = np.load(seg_path)
            if "sem_scores" in raw:
                packed_scores = raw["sem_scores"]  # shape: (H, W), dtype=uint16
                sem_scores = decode_sem_scores_compressed(packed_scores)  # shape: (H, W, 4)
            elif "sem_ids" in raw:
                sem_ids = raw["sem_ids"]  # shape: (H, W)
                sem_scores = np.eye(4, dtype=np.float32)[sem_ids]  # one-hot
            else:
                raise ValueError("Neither 'sem_scores' nor 'sem_ids' found in npz")
        else:
            sem_scores = self.get_segmentation(image_data)

        # Painting the pointcloud
        transforms = FrameTransformMatrix(vod_frame_data)
        t_camera_lidar = transforms.t_camera_lidar
        P = transforms.camera_projection_matrix
        painted_lidar = self.paint_lidar_points(lidar_data, sem_scores, t_camera_lidar, P)

        gt_labels_3d_list = []
        gt_bboxes_3d_list = []
        if self.split != 'test':
            raw_labels = vod_frame_data.raw_labels
            for idx, label in enumerate(raw_labels):
                label = label.split(' ')

                if label[self.LABEL_MAPPING['class']] in self.CLASSES:
                    gt_labels_3d_list.append(int(self.CLASSES.index(label[self.LABEL_MAPPING['class']])))

                    bbox3d_loc_camera = np.array(label[self.LABEL_MAPPING['bbox3d_location']])
                    trans_homo_cam = np.ones((1,4))
                    trans_homo_cam[:, :3] = bbox3d_loc_camera
                    bbox3d_loc_lidar = homogeneous_transformation(trans_homo_cam, local_transforms.t_lidar_camera)

                    bbox3d_locs = np.array(bbox3d_loc_lidar[0,:3], dtype=np.float32)
                    bbox3d_dims = np.array(label[self.LABEL_MAPPING['bbox3d_dimensions']], dtype=np.float32)[[2, 1, 0]]
                    bbox3d_rot = np.array([label[self.LABEL_MAPPING['bbox3d_rotation']]], dtype=np.float32)

                    gt_bboxes_3d_list.append(np.concatenate([bbox3d_locs, bbox3d_dims, bbox3d_rot], axis=0))

        painted_lidar = torch.tensor(painted_lidar)

        if gt_bboxes_3d_list == []:
            gt_labels_3d = np.array([0])
            gt_bboxes_3d = np.zeros((1,7))
        else:
            gt_labels_3d = np.array(gt_labels_3d_list, dtype=np.int64)
            gt_bboxes_3d = np.stack(gt_bboxes_3d_list, axis=0)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0))

        gt_labels_3d = torch.tensor(gt_labels_3d)

        return dict(
            lidar_data = painted_lidar,
            gt_labels_3d = gt_labels_3d,
            gt_bboxes_3d = gt_bboxes_3d,
            meta = dict(num_frame = num_frame),
            sem_scores = sem_scores,
            image = image_data
        )

    def get_segmentation(self, image_array):
        image = Image.fromarray(image_array)
        # Getting the segmentation
        input_tensor = self.seg_transform(image).unsqueeze(0)
        with torch.no_grad():
            sem_scores = self.seg_model(input_tensor)['out'].softmax(dim=1)  # [1, C, H, W]
        sem_scores = sem_scores.squeeze(0).permute(1, 2, 0).cpu().numpy()     # [H, W, C]

        # We are only interested in cars (7), pedestrians(15), and bicycles(2), everything else is background (0)
        bg = 1.0 - sem_scores[:, :, 2] - sem_scores[:, :, 7] - sem_scores[:, :, 15]
        sem_scores_reduced = np.stack([
            bg,
            sem_scores[:, :, 2],   # bicycle
            sem_scores[:, :, 7],   # car
            sem_scores[:, :, 15],  # person
        ], axis=-1)  # final shape: [H, W, 4]
        return sem_scores_reduced
    
    def paint_lidar_points(self, lidar_points, sem_scores, transform_matrix, P):
        """
        lidar_points: (N, 4) - [x, y, z, intensity]
        sem_scores: (H, W, C)
        transform_matrix: (4, 4)
        P: (3, 4)
        """
        H, W, C = sem_scores.shape

        # Step 1: Transform to camera frame
        lidar_points = lidar_points.copy()
        coords = lidar_points[:, :3]
        intensity = lidar_points[:, 3:4]  # (N, 1)
        
        lidar_hom = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)  # (N, 4)
        points_cam = (transform_matrix @ lidar_hom.T).T  # (N, 4)

        # Step 2: Project to image
        pixels = (P @ points_cam.T).T  # (N, 3)
        z = pixels[:, 2]
        valid_mask = z > 0

        u = (pixels[:, 0] / z).astype(int)
        v = (pixels[:, 1] / z).astype(int)

        # Step 3: Rescale u,v to segmentation map size
        orig_H = 1216
        orig_W = 1936
        u_scaled = (u * W / orig_W).astype(int)
        v_scaled = (v * H / orig_H).astype(int)

        # Step 4: Filter valid points
        in_bounds = (u_scaled >= 0) & (u_scaled < W) & (v_scaled >= 0) & (v_scaled < H)
        final_mask = valid_mask & in_bounds

        # Step 5: Retrieve segmentation scores
        seg = np.zeros((lidar_points.shape[0], C), dtype=np.float32)
        seg[final_mask] = sem_scores[v_scaled[final_mask], u_scaled[final_mask], :]

        # Step 6: Concatenate original features + segmentation
        painted = np.hstack([coords, intensity, seg])  # (N, 4 + C)

        return painted

    
def save_image(image_np, id = 0, output_dir="outputs"):
    plt.imsave(os.path.join(output_dir, f"image_{id}.png"), image_np)

def save_segmentation_map(sem_scores, id=0, output_dir="outputs"):
    """
    Saves the segmentation map as an RGB image.
    
    Args:
        sem_scores: np.ndarray of shape (H, W, C) with class probabilities
        id: identifier for the saved file
        output_dir: directory to save the image
    """

    # Get predicted class per pixel
    seg_map = np.argmax(sem_scores, axis=-1)  # (H, W)

    # Define a colormap (change as needed)
    class_colors = np.array([
        [128, 128, 128],  # grey (background)
        [255, 0, 0],      # red (bicycle)
        [0, 255, 0],      # green (car)
        [0, 0, 255],      # blue (person)
    ], dtype=np.uint8)

    # Map indices to RGB colors
    color_seg_map = class_colors[seg_map]  # (H, W, 3)

    plt.imsave(os.path.join(output_dir, f"segmentation_{id}.png"), color_seg_map)

def save_painted_projection(painted_lidar, id=0, projection_axis='xy'):
    """
    Visualize and save a top-down or front view of painted point cloud.

    Args:
        painted_lidar: (N, 4+C) array — original LiDAR + semantic scores
        id: identifier for saving
        projection_axis: 'xy', 'xz', or 'yz' for different views
    """

    coords = painted_lidar[:, :3]
    sem_scores = painted_lidar[:, 4:]  # skip intensity
    num_points = coords.shape[0]

    # Determine dominant semantic label
    dominant_class = np.argmax(sem_scores, axis=1)
    is_painted = sem_scores.sum(axis=1) > 0.001

    # Assign color per point
    color_map = np.array([
        [0.5, 0.5, 0.5],  # 0: background → gray
        [1.0, 0.0, 0.0],  # 1: person     → red
        [0.0, 1.0, 0.0],  # 2: car        → green
        [0.0, 0.0, 1.0],  # 3: bicycle    → blue
    ])

    # Default to black for unpainted
    point_colors = np.zeros((num_points, 3))  # black
    point_colors[is_painted] = color_map[dominant_class[is_painted]]

    # Select projection
    if projection_axis == 'xy':
        x, y = coords[:, 0], coords[:, 1]
    elif projection_axis == 'xz':
        x, y = coords[:, 0], coords[:, 2]
    elif projection_axis == 'yz':
        x, y = coords[:, 1], coords[:, 2]
    else:
        raise ValueError("Invalid projection_axis")

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, c=point_colors, s=0.5)
    plt.axis('equal')
    plt.title(f"Painted Point Cloud Projection ({projection_axis}-view)")
    plt.xlabel(projection_axis[0])
    plt.ylabel(projection_axis[1])
    plt.savefig(f"outputs/painted_pc_{projection_axis}_{id}.png", dpi=300)
    plt.close()


def visualize_painted_pointcloud(painted_lidar, class_names=["bg", "person", "car", "bike"]):
    # painted_lidar shape: (N, 4 + C), where first 3 are x,y,z and last C are semantic scores
    xyz = painted_lidar[:, :3]
    seg_scores = painted_lidar[:, 4:]  # skip [x, y, z, r]

    labels = np.argmax(seg_scores, axis=1)
    confidence = np.max(seg_scores, axis=1)

    # If all scores are zero, assign label -1 (for unpainted)
    labels[confidence == 0] = -1

    # Define colors
    color_map = {
        -1: [0, 0, 0],         # black for unpainted
         0: [0.5, 0.5, 0.5],   # grey for background
         1: [1.0, 0, 0],       # red for person
         2: [0, 1.0, 0],       # green for car
         3: [0, 0, 1.0],       # blue for bike
    }

    colors = np.array([color_map[label] for label in labels])

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Painted Point Cloud Visualization')
    plt.tight_layout()
    plt.savefig("painted_pointcloud.png")
    plt.close()

def save_sem_scores(dataset, output_dir="common_src/dataset/sem_cache"):
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(len(dataset)):
        frame_data = dataset[idx] 
        sem_scores = frame_data["sem_scores"]  # shape (H, W, 4), float32
        frame_id = frame_data["meta"]["num_frame"]

        # Quantize and encode the entire softmax map into (H, W) uint16
        quantized = np.round(sem_scores * 10).astype(np.uint16)  # shape (H, W, 4)
        compressed_score = (
            (quantized[:, :, 0] << 12) |
            (quantized[:, :, 1] << 8) |
            (quantized[:, :, 2] << 4) |
            (quantized[:, :, 3])
        ).astype(np.uint16)  # shape (H, W)

        save_path = os.path.join(output_dir, f"{frame_id}.npz")
        np.savez_compressed(save_path, sem_scores=compressed_score)

        if idx % 50 == 0:
            print(f"Saved {idx+1}/{len(dataset)} segmentations")


def save_argmax_segmentation(dataset, output_dir="common_src/dataset/sem_cache"):
     # shape (H, W)
    os.makedirs(output_dir, exist_ok=True)
    for idx in range(len(dataset)):
        frame_data = dataset[idx] 
        sem_scores = frame_data["sem_scores"].astype(np.float16)
        frame_id = frame_data["meta"]["num_frame"]
        seg_map = np.argmax(sem_scores, axis=-1).astype(np.uint16) 
        np.savez_compressed(os.path.join(output_dir, f"{frame_id}.npz"), sem_ids=seg_map)
        if idx % 50 == 0:
            print(f"Saved {idx+1}/{len(dataset)} segmentations")

def encode_softmax_to_uint16(softmax_vector):
    # Assume input is 4 values between 0 and 1
    quantized = (np.round(softmax_vector * 10)).astype(np.uint16)  # values 0–10
    # Pack into 16-bit int: 4 x 4-bit chunks
    packed = (quantized[0] << 12) | (quantized[1] << 8) | (quantized[2] << 4) | quantized[3]
    return packed

def decode_sem_scores(packed_scores):
    # packed_scores: shape (H, W), dtype=uint16
    q0 = (packed_scores >> 12) & 0xF
    q1 = (packed_scores >> 8) & 0xF
    q2 = (packed_scores >> 4) & 0xF
    q3 = packed_scores & 0xF

    decoded = np.stack([q0, q1, q2, q3], axis=-1).astype(np.float32) / 10.0  # shape (H, W, 4)
    return decoded

def save_sem_scores_compressed(dataset, output_dir="common_src/dataset/sem_cache"):
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(len(dataset)):
        frame_data = dataset[idx]
        sem_scores = frame_data["sem_scores"]  # shape: (H, W, 4)
        frame_id = frame_data["meta"]["num_frame"]

        # Extract only class channels (car, pedestrian, cyclist), skip background
        # Assume: sem_scores[..., 1] = car, [2] = pedestrian, [3] = cyclist
        selected = sem_scores[..., 1:]  # shape: (H, W, 3)

        # Quantize to 0–255
        quantized = np.clip(np.round(selected * 255), 0, 255).astype(np.uint8)  # (H, W, 3)

        # Save
        save_path = os.path.join(output_dir, f"{frame_id}.npz")
        np.savez_compressed(save_path, sem_scores=quantized)

        if idx % 50 == 0:
            print(f"Saved {idx+1}/{len(dataset)} compressed segmentations")

def decode_sem_scores_compressed(quantized):
    # quantized: shape (H, W, 3), dtype=uint8

    probs = quantized.astype(np.float32) / 255.0  # (H, W, 3)
    sum_ = probs.sum(axis=-1, keepdims=True)  # (H, W, 1)
    bg = np.clip(1.0 - sum_, 0.0, 1.0)  # background = 1 - sum of rest

    full = np.concatenate([bg, probs], axis=-1)  # (H, W, 4)
    return full

if __name__ == "__main__":
    # Test if Segmentation works
    dataset = ViewOfDelft(segmentation_generation=False)
    id = 658

    #save_sem_scores_compressed(dataset)

    ### Timing Segmentation
    # start = time.time()
    data_658 = dataset[id]
    # end = time.time()
    # print(f"Time to load sample {id}: {end - start:.2f} seconds")

    ### Saving visualizations
    # image = data_658["image"]
    sem_scores = data_658["sem_scores"]
    painted_pc = data_658["lidar_data"]
    # 
    # save_painted_projection(painted_pc, id, "xy")
    # The images get saved under outputs/
    # save_segmentation_map(sem_scores, id)
    # save_image(image, id=id)