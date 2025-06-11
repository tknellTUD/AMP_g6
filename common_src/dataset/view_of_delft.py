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
                 split = 'train'):
        super().__init__()
        
        self.data_root = data_root
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}. Must be one of ['train', 'val', 'test']"
        self.split = split
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
        # Get segmentations scores
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
                    bbox3d_dims = np.array(label[self.LABEL_MAPPING['bbox3d_dimensions']], dtype=np.float32)[[2, 1, 0]] # hwl -> lwh
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
            meta = dict(
                num_frame = num_frame 
            ),
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
        lidar_points: (N, 4)
        sem_scores: (H, W, C)
        transform_matrix: (4, 4)
        P: (3, 4)
        """
        H, W, C = sem_scores.shape

        # Step 1: Transform to camera frame
        lidar_points = lidar_points.copy()
        lidar_points[:,3] = 1.0
        points_cam = (transform_matrix @ lidar_points.T).T  # shape: (N, 4)

        # Step 2: Project to image using camera matrix
        pixels = (P @ points_cam.T).T  # (N, 3)
        z = pixels[:, 2]
        valid_mask = z > 0

        u = (pixels[:, 0] / z).astype(int)
        v = (pixels[:, 1] / z).astype(int)

        # Rescale u and w to fit to different size segmentation
        orig_H = 1216
        orig_W = 1936
        H, W, _ = sem_scores.shape  # resized seg map shape (e.g., 512, 1024)

        # Rescale u and v from original image scale to segmentation map scale
        u_scaled= (u * W / orig_W).astype(int)
        v_scaled = (v * H / orig_H).astype(int)

        # Step 3: Check image bounds
        in_bounds = (u_scaled >= 0) & (u_scaled < W) & (v_scaled >= 0) & (v_scaled < H)
        final_mask = valid_mask & in_bounds

        # Step 4: Retrieve segmentation scores
        seg = np.zeros((lidar_points.shape[0], C), dtype=np.float32)
        seg[final_mask] = sem_scores[v_scaled[final_mask], u_scaled[final_mask], :]  # fast lookup

        # Step 5: Concatenate painted features
        painted = np.hstack([lidar_points, seg])  # (N, 4 + C)

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

if __name__ == "__main__":
    # Test if Segmentation works
    dataset = ViewOfDelft()
    id = 658
    start = time.time()
    data_658 = dataset[id]
    end = time.time()
    print(f"Time to load sample {id}: {end - start:.2f} seconds")
    image = data_658["image"]
    sem_scores = data_658["sem_scores"]
    painted_pc = data_658["lidar_data"]
    # 
    # save_painted_projection(painted_pc, id, "xy")
    # The images get saved under outputs/
    save_segmentation_map(sem_scores, id)
    # save_image(image, id=id)