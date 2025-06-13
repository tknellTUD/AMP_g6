import os
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
from PIL import Image as PILImage
from random import randint
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from numba import njit

import torch.multiprocessing as mp

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
                 segmentation_generation=False,
                 seg_model = False,
                 device = 'cuda'): ### REMEMBER TO CHANGE THIS BACK TO FALSE
        
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
        self.seg_model = seg_model
        
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
            sem_scores = polish_segmentation(sem_scores, save_visualization=False, device='cuda')
        # Painting the pointcloud
        transforms = FrameTransformMatrix(vod_frame_data)
        t_camera_lidar = transforms.t_camera_lidar
        P = transforms.camera_projection_matrix
        painted_lidar = self.paint_lidar_points(lidar_data, sem_scores, t_camera_lidar, P, image_data=image_data, num_frame=num_frame)

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
        # print(f"painted_lidar is a {'Tensor' if isinstance(painted_lidar, torch.Tensor) else 'NumPy array'}")
        # print(f"sem_scores is a {'Tensor' if isinstance(sem_scores, torch.Tensor) else 'NumPy array'}")
        return dict(
            lidar_data = painted_lidar,
            gt_labels_3d = gt_labels_3d,
            gt_bboxes_3d = gt_bboxes_3d,
            meta = dict(num_frame = num_frame),
            sem_scores = sem_scores,
            image = image_data)

    def get_segmentation(self, image_array):
        start_time = time.time()
        image = Image.fromarray(image_array)
        # Getting the segmentation
        input_tensor = self.seg_transform(image).unsqueeze(0).to('cuda')
        with torch.no_grad():
            sem_scores = self.seg_model(input_tensor)['out'].softmax(dim=1)  # [1, C, H, W]
        sem_scores = sem_scores.squeeze(0).permute(1, 2, 0).cpu().numpy()     # [H, W, C]

        # We are only interested in cars (7), pedestrians(15), and cyclists(2), everything else is background (0)
        bg = 1.0 - sem_scores[:, :, 2] - sem_scores[:, :, 7] - sem_scores[:, :, 15]
        sem_scores_reduced = np.stack([
            bg,
            sem_scores[:, :, 2],   # cyclist
            sem_scores[:, :, 7],   # car
            sem_scores[:, :, 15],  # person
        ], axis=-1)  # final shape: [H, W, 4]
        sem_time = time.time()
        # print(f"Segmentation time: {sem_time - start_time:.2f} seconds")
        #sem_scores_reduced = polish_segmentation(sem_scores_reduced, save_visualization=False, device='cuda')
        return sem_scores_reduced
    
    def paint_lidar_points(self, lidar_points, sem_scores, transform_matrix, P, image_data=None, num_frame=0):
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
        # intensity = lidar_points[:, 3:4]  # (N, 1)
        intensity = np.ones((lidar_points.shape[0], 1), dtype=np.float32) # Removing intensity to test if this fixes performance issues
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

        # # Step 4.1: Project valid lidar points onto the image
        # projected_image = image_data.copy()
        # for i in range(len(u)):
        #     if final_mask[i]:
        #         u_img, v_img = u[i], v[i]
        #         if 0 <= u_img < orig_W and 0 <= v_img < orig_H:
        #             # Draw a 3x3 red square around the pixel
        #             for dx in range(-1, 2):
        #                 for dy in range(-1, 2):
        #                     nx, ny = u_img + dx, v_img + dy
        #                     if 0 <= nx < orig_W and 0 <= ny < orig_H:
        #                         # Use the segmentation color for the lidar point
        #                         seg_color = sem_scores[v_scaled[i], u_scaled[i], :]
        #                         # Map segmentation scores to RGB colors
        #                         seg_color_rgb = (seg_color[:3] * 255).astype(np.uint8)  # Scale to 0-255
        #                         projected_image[ny, nx] = seg_color_rgb

        # # Save the projected image
        # output_path = os.path.join("outputs", f"projected_lidar_{num_frame}.png")
        # Image.fromarray(projected_image).save(output_path)

        # Step 6: Concatenate original features + segmentation
        painted = np.hstack([coords, intensity, seg])  # (N, 4 + C)

        return torch.tensor(painted, device='cuda')

    
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
        [1.0, 0.0, 0.0],  # 1: bicycle     → red
        [0.0, 1.0, 0.0],  # 2: car        → green
        [0.0, 0.0, 1.0],  # 3: Person    → blue
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

    # # Plot
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=1)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Painted Point Cloud Visualization')
    # plt.tight_layout()
    # plt.savefig("painted_pointcloud.png")
    # plt.close()

def save_sem_scores(dataset, output_dir="common_src/dataset/sem_cache"):
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(len(dataset)):
        frame_data = dataset[idx] 
        sem_scores = frame_data["sem_scores"]  # shape (H, W, 4), float32
        frame_id = frame_data["meta"]["num_frame"]

        # Quantize and encode the entire softmax map into (H, W) uint16
        quantized = np.round(sem_scores * 10).astype(np.uint16)  # shape (H, W, 4)
        quantized = polish_segmentation(quantized, save_visualization=False)  # polish the segmentation
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
    avg_time = 0.0
    for idx in range(len(dataset)):
        img_time = time.time()
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

        if idx % 50 == 0:
                seg_map = np.argmax(quantized, axis=-1).astype(np.uint16)
                np.savez_compressed(os.path.join(output_dir, f"argmax_segmentation_{idx}.npz"), seg_map=seg_map)
                save_segmentation_map(quantized, idx)

        avg_time += time.time() - img_time
        print(f"Average processing time per frame: {avg_time / (idx+1)} seconds")
def decode_sem_scores_compressed(quantized):
    # quantized: shape (H, W, 3), dtype=uint8

    probs = quantized.astype(np.float32) / 255.0  # (H, W, 3)
    sum_ = probs.sum(axis=-1, keepdims=True)  # (H, W, 1)
    bg = np.clip(1.0 - sum_, 0.0, 1.0)  # background = 1 - sum of rest

    full = np.concatenate([bg, probs], axis=-1)  # (H, W, 4)
    return full

def show_diff_classification(full_seg, reconst_seg, feature=0):
    diff = np.abs(full_seg[:, :, feature] - reconst_seg[:, :, feature])

    # plt.imshow(diff, cmap='hot')
    # plt.colorbar(label='|Original - Reconstructed| (Background)')
    # plt.title(f"Error in Reconstructed Background Score (Class {feature})")
    # plt.tight_layout()
    # plt.savefig(f"outputs/background_diff_heatmap_class{feature}.png")
    # plt.close()

import torch

import torch

import matplotlib.pyplot as plt
try:
    # Available in torchvision 0.14+ (CPU) and 0.19-dev (CUDA)
    from torchvision._C import _connected_components as _cc_torchvision  # type: ignore
except ImportError:  # pragma: no cover – keep runtime dependency optional
    _cc_torchvision = None

def polish_segmentation(sem_scores, *, save_visualization=False, device="cuda"):
    """
    Post-process a per-pixel semantic-scores tensor (H × W × C) so that
    pedestrians riding bicycles are re-labelled as cyclists.

    ── Class layout (index in last dim) ──
        0 → background
        1 → bicycle / cyclist
        2 → car                     (only used for a debug overlay)
        3 → pedestrian
    """
    # ------------------------------------------------------------------ #
    # 0.  Torch bookkeeping
    # ------------------------------------------------------------------ #
    polish_start = time.time()
    sem_scores_tensor = torch.as_tensor(sem_scores, device=device, dtype=torch.float32)
    # print(f"Polishing segmentation on device: {device}")    
    sem_scores_copy   = sem_scores_tensor.clone()

    # Binary masks for the classes we care about
    bicycle_mask    = sem_scores_copy[:, :, 1].cpu().numpy() > 0.50
    pedestrian_mask = sem_scores_copy[:, :, 3].cpu().numpy() > 0.50

    height, width = bicycle_mask.shape   # shape = (rows / y, columns / x)
    # print(f"bicycle_mask shape = (H={height}, W={width})")

    ## Connected components labelling
    labels, K = connected_components(pedestrian_mask)
    # print(f"Found {K} pedestrian blobs")
    # print(labels.shape)
    # bounding boxes
    ped_bboxes = []
    for k in range(1, K+1):
        ys, xs = np.nonzero(labels == k)
        ped_bboxes.append((ys.min(), xs.min(), ys.max(), xs.max()))
    # print(f"Bounding boxes for {K} pedestrian blobs: {ped_bboxes}")
    #print(labels)
    # ------------------------------------------------------------------ #
    # 2.  For each pedestrian blob, look for bicycle pixels directly below
    # ------------------------------------------------------------------ #
    cyclist_pixels = []                       # global collection of all accepted bicycle pixels

    overlap_mask = np.zeros_like(bicycle_mask, dtype=bool)  # Mask to put all correct cyclist pixels
    cumulative_blob_mask = np.zeros_like(labels, dtype=bool)

    for k in range(1, K+1):
        ys, xs = np.nonzero(labels == k)
        min_r, min_c, max_r, max_c  = ys.min(), xs.min(), ys.max(), xs.max()


        # ------------------------------------------------------------------
        # 1.  Slice object that represents the pedestrian bounding box
        #     (+1 because Python slices are [start, stop) while your max_* is inclusive)
        # ------------------------------------------------------------------
        # Search window: a slim rectangle just below the pedestrian blob
        blob_height = max_r - min_r + 1
        blob_width  = max_c - min_c + 1
        if blob_height/ blob_width < 2:  # too narrow, skip
            centre_c = (min_c + max_c) // 2

            search_min_r = max_r - blob_height // 3  # start at the blob’s bottom row
            search_max_r = min(height - 1, max_r + blob_height // 2)  # 25% extra below

            search_min_c = max(0, centre_c - blob_height // 2)
            search_max_c = min(width - 1, centre_c + blob_height // 2)

            # Draw a green line (car class) on the bounding box
            # sem_scores_copy[search_min_r:search_max_r + 1, search_min_c, 2] = 1.0  # left vertical line
            # sem_scores_copy[search_min_r:search_max_r + 1, search_max_c, 2] = 1.0  # right vertical line
            # sem_scores_copy[search_min_r, search_min_c:search_max_c + 1, 2] = 1.0  # top horizontal line
            # sem_scores_copy[search_max_r, search_min_c:search_max_c + 1, 2] = 1.0  # bottom horizontal line

            bbox_sl = np.s_[search_min_r:search_max_r + 1, search_min_c:search_max_c + 1]
            
            # ------------------------------------------------------------------
            # 2.  Boolean mask of bicycle pixels *inside* the pedestrian box
            #     (True where bicycle, False elsewhere, shape == (box-height, box-width))
            # ------------------------------------------------------------------
            bike_in_box = bicycle_mask[bbox_sl]

            # ──►  If you also want to ignore any pixels that belong to the pedestrian itself:
            # bike_in_box = bicycle_mask[bbox_sl] & ~pedestrian_mask[bbox_sl]

            # ------------------------------------------------------------------
            # 3.  (optional) Same-size-as-image mask with True only for the overlap
            # ------------------------------------------------------------------
            overlap_mask[bbox_sl] = bike_in_box       # or bike_in_box after the ped filter


            # bike_coords is an (N, 2) array of (r, c)
            # Visualize the overlap mask with teal pixels
            # Visualize the overlap mask with pink pixels
            if overlap_mask[bbox_sl].any():
                # Create a mask for the current pedestrian blob
                blob_mask = (labels == k)
                # Add the current blob mask to the cumulative mask
                cumulative_blob_mask |= blob_mask
            
    # Swap bike and pedestrian scores for cumulative blob mask
    temp = sem_scores_copy[..., 1][cumulative_blob_mask].clone()
    sem_scores_copy[..., 1][cumulative_blob_mask] = sem_scores_copy[..., 3][cumulative_blob_mask]
    sem_scores_copy[..., 3][cumulative_blob_mask] = temp

    # Move isolated cyclist pixels to background
    isolated_cyclist_mask = bicycle_mask & ~overlap_mask
    temp = sem_scores_copy[..., 0][isolated_cyclist_mask].clone()
    sem_scores_copy[..., 0][isolated_cyclist_mask] = sem_scores_copy[..., 1][isolated_cyclist_mask]
    sem_scores_copy[..., 1][isolated_cyclist_mask] = temp

    polished_segmentation = torch.nan_to_num(sem_scores_copy, nan=0.0).cpu().numpy()

    # ------------------------------------------------------------------ #
    # 5.  Optional visual comparison
    # ------------------------------------------------------------------ #
    if save_visualization:
        plt.figure(figsize=(12, 6), tight_layout=True)

        plt.subplot(1, 2, 1)
        plt.imshow(np.argmax(sem_scores, axis=-1), cmap="viridis")
        plt.title("Original segmentation")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(np.argmax(polished_segmentation, axis=-1), cmap="viridis")
        plt.title("Polished segmentation")
        plt.colorbar()

        plt.savefig("outputs/segmentation_comparison.png")
        plt.close()

    # print(f"Polishing took {time.time() - polish_start:.2f} seconds")
    return polished_segmentation

@njit
def connected_components(mask):
    """
    8-connected labelling.
    Parameters
    ----------
    mask : 2-D bool array
    Returns
    -------
    labels : 2-D int32, same size as mask, 0 = background, 1…K = blob id
    K      : number of blobs
    """
    H, W = mask.shape
    labels = np.zeros((H, W), np.int32)

    # Union-Find helpers -------------------------------------------------
    parent = np.arange(H*W, dtype=np.int32)
    rank   = np.zeros(H*W,  dtype=np.int8)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]     # path compression
            x = parent[x]
        return x

    def union(x, y):
        xr, yr = find(x), find(y)
        if xr == yr:
            return
        if rank[xr] < rank[yr]:
            parent[xr] = yr
        elif rank[xr] > rank[yr]:
            parent[yr] = xr
        else:
            parent[yr] = xr
            rank[xr]  += 1

    # Pass 1 – build equivalence graph ----------------------------------
    for r in range(H):
        for c in range(W):
            if not mask[r, c]:
                continue
            idx = r*W + c
            # check north & west neighbours (8-connectivity → NW, N, W, NE)
            if r > 0 and mask[r-1, c]:              union(idx, (r-1)*W + c)
            if c > 0 and mask[r, c-1]:              union(idx, r*W + (c-1))
            if r > 0 and c > 0 and mask[r-1, c-1]:  union(idx, (r-1)*W + c-1)
            if r > 0 and c < W-1 and mask[r-1, c+1]:union(idx, (r-1)*W + c+1)

    # Pass 2 – assign compact labels ------------------------------------
    root2lbl = {}
    lbl = 1
    for r in range(H):
        for c in range(W):
            if not mask[r, c]:
                continue
            root = find(r*W + c)
            if root not in root2lbl:
                root2lbl[root] = lbl
                lbl += 1
            labels[r, c] = root2lbl[root]

    return labels, lbl-1

def save_combined_visualization(image, original_scores, updated_scores=None, id=0, output_dir="outputs"):
    # Convert original and updated segmentation maps to RGB
    original_seg_map = np.argmax(original_scores, axis=-1)
    updated_seg_map = np.argmax(updated_scores, axis=-1)

    class_colors = np.array([
        [128, 128, 128],  # grey (background)
        [255, 0, 0],      # red (bicycle)
        [0, 255, 0],      # green (car)
        [0, 0, 255],      # blue (person)
    ], dtype=np.uint8)

    original_rgb = class_colors[original_seg_map]
    updated_rgb = class_colors[updated_seg_map]

    # Convert to PIL images
    image_pil = PILImage.fromarray(image)
    original_pil = PILImage.fromarray(original_rgb)
    updated_pil = PILImage.fromarray(updated_rgb)

    # Resize all images to the same width
    target_width = 1024  # Set a target width for resizing
    image_pil = image_pil.resize((target_width, int(image_pil.height * target_width / image_pil.width)))
    original_pil = original_pil.resize((target_width, int(original_pil.height * target_width / original_pil.width)))
    updated_pil = updated_pil.resize((target_width, int(updated_pil.height * target_width / updated_pil.width)))

    # Combine images vertically
    total_height = image_pil.height + original_pil.height + updated_pil.height
    combined_image = PILImage.new("RGB", (target_width, total_height))
    combined_image.paste(image_pil, (0, 0))
    combined_image.paste(original_pil, (0, image_pil.height))
    combined_image.paste(updated_pil, (0, image_pil.height + original_pil.height))

    # Save combined image as JPEG
    combined_image.save(os.path.join(output_dir, f"combined_visualization_{id}.jpeg"), "JPEG")

if __name__ == "__main__":
 
     # Set the multiprocessing start method to 'spawn'
    mp.set_start_method('spawn', force=True)

    # Test if Segmentation works
    seg_model = deeplabv3_resnet101(pretrained=True).eval().to('cuda')
    dataset = ViewOfDelft(segmentation_generation=True, seg_model=seg_model, device='cuda')
    
    # # Find an image with both bicycles and pedestrians
    # id = -1
    # for _ in range(len(dataset)):
    #     idx = randint(0, len(dataset) - 1)
    #     sem_scores = dataset[idx]["sem_scores"]
    #     has_cyclist = np.any(np.argmax(sem_scores, axis=-1) == 1)  # Cyclist class index
    #     has_pedestrian = np.any(np.argmax(sem_scores, axis=-1) == 3)  # Pedestrian class index
    #     if has_cyclist and has_pedestrian:
    #         id = idx
    #         break

    # if id == -1:
    #     raise ValueError("No image with both cyclists and pedestrians found in the dataset.")
    # id = 640
    
    # save_sem_scores_compressed(dataset)

    # ## Timing Segmentation
    # start = time.time()
    # data_658 = dataset[id]
    # end = time.time()
    # print(f"Time to load sample {id}: {end - start:.2f} seconds")

    # ## Saving visualizations
    # image = data_658["image"]
    # sem_scores = data_658["sem_scores"]
    # painted_pc = data_658["lidar_data"]
    
    # save_painted_projection(painted_pc, id, "xy")
    # # The images get saved under outputs/
    # save_segmentation_map(sem_scores, id)
    # #save_image(image, id=id, output_dir="outputs")

    # updated_scores = polish_segmentation(sem_scores)
    # # Combine the image, original segmentation, and updated segmentation into one JPEG

    # # Save the combined visualization
    # save_combined_visualization(image, sem_scores, updated_scores=updated_scores, id=id, output_dir="outputs")

    # sem_scores = decode_sem_scores_compressed(np.load("common_src/dataset/sem_cache/09641.npz")["sem_scores"])
    # save_segmentation_map(sem_scores, id=0, output_dir="outputs")