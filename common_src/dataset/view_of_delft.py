import os
import sys

# Add the parent directory to the Python path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import numpy as np
from common_src.model.utils import LiDARInstance3DBoxes

import torch
from torch.utils.data import Dataset

from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix, homogeneous_transformation

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
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # print(f"IDX: {idx}")
        num_frame = self.sample_list[idx]
        vod_frame_data = FrameDataLoader(kitti_locations=self.vod_kitti_locations,
                                         frame_number=num_frame)
        local_transforms = FrameTransformMatrix(vod_frame_data)
        
        lidar_data = torch.tensor(vod_frame_data.lidar_data)


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
        
        gt_labels_3d = torch.tensor(gt_labels_3d, dtype=torch.float32)  # Ensure float32 data type
        
        stereo_camera = vod_frame_data.image #[H, W, C]
        # print(f"Stereo camera shape: {stereo_camera.shape}")
        stereo_camera = torch.tensor(stereo_camera, dtype=torch.float32)  # Ensure float32 data type  # [1, C, H, W] for num cams 1
        # print(f"Stereo camera tensor shape: {stereo_camera.shape}")
        return dict(
            lidar_data = lidar_data,
            stereo_camera = stereo_camera,
            gt_labels_3d = gt_labels_3d,
            gt_bboxes_3d = gt_bboxes_3d,
            meta = dict(
            num_frame = num_frame 
            )
        )


def forward_projection(self, img_feats_fused):
    """
    Dummy version: Averages image features and reshapes into BEV.
    Replace with a real view-transformer (e.g. Lift-Splat-Shoot) for production.
    """
    B, N, C, H, W = img_feats_fused.shape
    img_feats = img_feats_fused.view(B, N * H, W, C).permute(0, 3, 1, 2)  # [B, C, H*N, W]
    bev_feats = torch.mean(img_feats, dim=2, keepdim=False)  # Collapse height

    return bev_feats  # [B, C, H_bev, W_bev]

def prepare_lidar_feats_for_hat(points, lidar2cam, cam_intrinsics, batch_size, num_cams, depth_bins, img_width, voxel_feature_dim=6, max_points_per_bin=10):
    """
    Constructs depth-aware radar features for HAT using raw radar/LiDAR points.

    Args:
        points (List[Tensor]): List of [B] tensors, each [N_i, 4] with (x, y, z, intensity)
        lidar2cam (Tensor): [B, N, 4, 4] transformation from lidar to each camera
        cam_intrinsics (Tensor): [B, N, 3, 3] camera intrinsics
        batch_size (int): Number of scenes
        num_cams (int): Number of cameras
        depth_bins (int): Number of depth bins D
        img_width (int): Feature map width W
        voxel_feature_dim (int): Input feature size (e.g., 4 or 6)
        max_points_per_bin (int): Optional max for sparsity control

    Returns:
        lidar_feats: [B, N, 1, W, D, C]
    """

    C = voxel_feature_dim
    lidar_feats = torch.zeros(batch_size, num_cams, 1, img_width, depth_bins, C)
    bin_counts = torch.zeros(batch_size, num_cams, 1, img_width, depth_bins)

    for b in range(batch_size):
        for n in range(num_cams):
            pts = points[b]  # [N, 4]
            if pts.shape[0] == 0:
                continue

            P = lidar2cam[b, n]      # [4, 4]
            K = cam_intrinsics[b, n]  # [3, 3]

            N_pts = pts.shape[0]
            pts_h = torch.cat([pts[:, :3], torch.ones(N_pts, 1, device=pts.device)], dim=1)  # [N, 4]
            cam_pts = (P @ pts_h.T).T[:, :3]  # [N, 3]

            valid = cam_pts[:, 2] > 0
            cam_pts = cam_pts[valid]
            raw_feats = pts[valid][:, :voxel_feature_dim]

            img_pts = (K @ cam_pts.T).T  # [N, 3]
            img_pts = img_pts[:, :2] / cam_pts[:, 2:3]  # [N, 2]
            x_img = img_pts[:, 0].long()
            z_cam = cam_pts[:, 2]

            z_max = z_cam.max().item() + 1e-5
            for i in range(cam_pts.shape[0]):
                x = x_img[i].item()
                d = int(z_cam[i].item() / (z_max / depth_bins))
                if 0 <= x < img_width and 0 <= d < depth_bins:
                    if bin_counts[b, n, 0, x, d] < max_points_per_bin:
                        lidar_feats[b, n, 0, x, d] += raw_feats[i]
                        bin_counts[b, n, 0, x, d] += 1

    nonzero = bin_counts > 0
    lidar_feats[nonzero] /= bin_counts[nonzero].unsqueeze(-1)

    return lidar_feats

def cp_tasks():
    
        # 1. Extract image features
        img_feats = self.image_backbone(img_data)  # expected shape: [B, N, H, W, C]

        # 2. Apply HAT: fuse image features with LiDAR features
        # reshape voxel_features to match image perspective format if needed
        lidar_feats_for_hat = self.prepare_lidar_feats_for_hat(voxel_features, coors, bs)  # implement this
        img_feats_fused = self.hat(img_feats, lidar_feats_for_hat)

        # 3. Project to BEV
        bev_feats_img = self.forward_projection(img_feats_fused)  # shape: [B, C, H_bev, W_bev]

        # 4. Fuse LiDAR + Image BEV features
        fused_bev = torch.cat([bev_feats_lidar, bev_feats_img], dim=1)
        fused_bev = self.se_fusion(fused_bev)

        # 5. Refine with backprojection guided by LiDAR
        fused_bev = self.ldc(fused_bev, bev_feats_img)

def extract_lidar_uvz_features(lidar_pc_lidar, transform_matrix, projection_matrix, image_shape, feature_dim=4, img_data=None):
    """
    From raw LiDAR points, return valid projected pixel x-coords (u), depth (z), and point features.

    Args:
        lidar_pc_lidar (np.ndarray): shape (N, 4+C), where first 4 = (x, y, z, intensity)
        transform_matrix (np.ndarray): (4, 4) LiDAR-to-camera transform
        projection_matrix (np.ndarray): (3, 4) camera projection matrix
        image_shape (tuple): (H, W)
        feature_dim (int): Number of features to keep (default: 4)

    Returns:
        output (np.ndarray): shape (N_valid, 1 + 1 + C), i.e., [u, z, features]
    """
    # print(f"T_camera_LiDAR:\n{transform_matrix}")
    print(f"P_camera:\n{projection_matrix}")
    H, W = 1216,1936
    N = lidar_pc_lidar.shape[0]

    # Ensure the fourth feature of lidar_pc_camera is set to 1
    intensities = lidar_pc_lidar[:, 3].copy()
    lidar_pc_lidar[:, 3] = 1

    # Step 1: Transform to camera frame
    lidar_pc_camera = transform_matrix.dot(lidar_pc_lidar.T).T  # shape: (N, 4)  # [N, 4 + C]
    print("Min x, y, z values for lidar_pc_lidar:", lidar_pc_lidar[:, :3].min(axis=0))
    print("Max x, y, z values for lidar_pc_lidar:", lidar_pc_lidar[:, :3].max(axis=0))
    print("Min x, y, z values for lidar_pc_camera:", lidar_pc_camera[:, :3].min(axis=0))
    print("Max x, y, z values for lidar_pc_camera:", lidar_pc_camera[:, :3].max(axis=0))
    
    # Step 2: Project to image plane
    # pixels = (projection_matrix @ lidar_pc_camera.T).T  # [N, 3]
    pixels = (projection_matrix @ lidar_pc_camera.T).T
    print("Pixels shape:", pixels.shape)
    z = pixels[:, 2]  # Depth in camera frame
    valid_mask = z > 0
    u = (pixels[:, 0] / z).astype(int)
    v = (pixels[:, 1] / z).astype(int)

    # Step 3: Check image bounds
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    final_mask = valid_mask & in_bounds

    lidar_pc_camera[:, 3] = intensities
    feats_valid = lidar_pc_camera[final_mask, :feature_dim]  # [N_valid, C]

    u_valid = u[final_mask].reshape(-1, 1)  # Extract valid u values
    z_valid = z[final_mask].reshape(-1, 1)  # Extract valid z values

    num_W_bins = image_shape[1]  # W

    num_D_bins = 64  # D
    z_max = 100
    depth_bin_size = z_max / num_D_bins

    # Clip depth values to the fixed range
    z_valid = np.clip(z_valid, 0, z_max)

    # Step 4: Bin the valid points width and depth wise
    binned_features = np.zeros((1, 1, num_W_bins, num_D_bins, feature_dim), dtype=np.float32)
    bin_counts = np.zeros((1, 1, num_W_bins, num_D_bins), dtype=np.int32)

    depth_bin_size = z_max / num_D_bins
    # Overlay valid points on the image
    import matplotlib.pyplot as plt

    # Extract valid u and v coordinates
    v_valid = v[final_mask].reshape(-1, 1)

    # Load the image for visualization
    
    # Bin the valid points width and depth-wise
    for i in range(u_valid.shape[0]):
        u_bin = u_valid[i, 0]
        d_bin = int(z_valid[i, 0] / depth_bin_size)

        if 0 <= u_bin < num_W_bins and 0 <= d_bin < num_D_bins:
            if bin_counts[0, 0, u_bin, d_bin] < 10:  # Limit max points per bin
                binned_features[0, 0, u_bin, d_bin] += feats_valid[i]
                bin_counts[0, 0, u_bin, d_bin] += 1

    nonzero_mask = bin_counts > 0
    binned_features[nonzero_mask] /= bin_counts[nonzero_mask].reshape(-1, 1)
    
    output = np.concatenate([u_valid, z_valid, feats_valid], axis=1)  # [N_valid, 1 + 1 + C]

    return output, binned_features, bin_counts


if __name__ == '__main__':
    dataset = ViewOfDelft(data_root='data/view_of_delft', split='train')
    print(f"Dataset length: {len(dataset)}")

    from common_src.model.middle_encoders.HyDRa.height_association_transformer import HeightAssociationTransformer
    from common_src.model.middle_encoders.HyDRa.SE_fusion import SEFusion
    from common_src.model.middle_encoders.HyDRa.back_projection import LidarDepthRefiner
    from common_src.model.backbones.img_backbone import ImageBackbone
    import yaml
    import open3d as o3d
    # Load the configuration from a YAML file
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config/model/centerpoint.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract HAT configuration
    hat_config = config.get('hat', None)
    se_fusion_config = config.get('se_fusion', None)
    ldc_config = config.get('ldc', None)
    image_backbone = ImageBackbone(out_channels=128)
    print("HAT Config:", config.get('hat', None))
    hat = HeightAssociationTransformer(hat_config)  
    print(se_fusion_config)
    se_fusion = SEFusion(se_fusion_config)
    print(ldc_config)
    ldc = LidarDepthRefiner(ldc_config)
        # Prepare LiDAR features for HAT
    vod_frame_data = FrameDataLoader(kitti_locations=dataset.vod_kitti_locations, 
                                     frame_number='01545')
    
    img_data = vod_frame_data.image  # [H, W, C]
    img_data = torch.tensor(img_data, dtype=torch.float32).permute(2, 0, 1)  # [C, H, W]
    img_data = img_data.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W] for batch size 1 and num cams 1
    print(f"Image data shape: {img_data.shape}")
    img_feats = image_backbone(img_data)  # [B, N, H, W, C]
    print(f"Image features shape: {img_feats.shape}")
    # Reshape to [B, N, H, W, C] where B=1, N=1 (single camera), H=H, W=W, C=C

    B, N, H, W, C = img_feats.shape
    height_queries = img_feats.view(B * N * W, H, C)
    print(f"Height queries shape: {height_queries.shape}")

                                     
    transforms = FrameTransformMatrix(vod_frame_data)
    t_camera_lidar = transforms.t_camera_lidar
    P = transforms.camera_projection_matrix
    lidar_data = vod_frame_data.lidar_data
    print(f"Lidar data shape: {lidar_data.shape}")
    # Convert img_data from [1, 1, C, H, W] to [H, W, C]
    img_data = img_data.squeeze(0).squeeze(0).permute(1, 2, 0).numpy()

    output, binned_features, bin_counts = extract_lidar_uvz_features(
        lidar_pc_lidar=lidar_data,           # [N, 4 + semantic scores]
        transform_matrix=t_camera_lidar,      # [4, 4]
        projection_matrix=P,                  # [3, 4]
        image_shape=(H, W),                   # size of sem_scores or image
        feature_dim=lidar_data.shape[1],    # preserve full painted features
        img_data=img_data)


    print(f"binned_features shape: {binned_features.shape}")
    # Perform positional encoding
    B, N, W, D, C = binned_features.shape
    binned_features = torch.tensor(binned_features)  # If it's NumPy
    lidar_seq = binned_features.view(B * N * W, D, C)
    print(lidar_seq.shape)
    print(height_queries.shape)
    depth_pos_encoding = nn.Parameter(torch.randn(D, C))
    lidar_seq += depth_pos_encoding.unsqueeze(0).unsqueeze(0)  # Add depth positional encoding

    # import matplotlib.pyplot as plt

    # # Sum the bin contents along the feature dimension to get a scalar value for each bin
    # bin_sums = binned_features.sum(axis=-1).squeeze()  # [1, 1, 1, W, D] -> [W, D]
    # # Extract x, y coordinates from the LiDAR point cloud
    # x_coords = lidar_pc_camera[:, 0]
    # y_coords = lidar_pc_camera[:, 1]

    # # Create a scatter plot for the top-down view
    # plt.figure(figsize=(10, 8))
    # plt.scatter(x_coords, y_coords, s=1, c='blue', alpha=0.5)
    # plt.xlabel('X (meters)')
    # plt.ylabel('Y (meters)')
    # plt.title('Top-Down View of LiDAR Point Cloud')
    # plt.axis('equal')
    # plt.show()

    # # Create a top-down view plot
    # plt.figure(figsize=(10, 8))
    # plt.imshow(bin_sums.T, cmap='viridis', origin='lower', aspect='auto')
    # plt.colorbar(label='Bin Content Sum')
    # plt.xlabel('Image Width Bins')
    # plt.ylabel('Depth Bins')
    # plt.title('Top-Down View of Binned Features')
    # plt.show()








### May be worthwile to change depth from z value to distance from camera