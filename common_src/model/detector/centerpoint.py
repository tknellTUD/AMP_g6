import os
import tempfile
import pickle
from datetime import datetime

import numpy as np 
import torch
import torch.nn.functional as F
from collections import OrderedDict

from vod.evaluation import Evaluation
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix, homogeneous_transformation

import lightning as L
import torch.distributed as dist

from common_src.ops import Voxelization
from common_src.model.voxel_encoders import PillarFeatureNet
from common_src.model.middle_encoders import PointPillarsScatter
from common_src.model.backbones import SECOND
from common_src.model.necks import SECONDFPN
from common_src.model.heads import CenterHead
from common_src.model.middle_encoders.HyDRa.height_association_transformer import HeightAssociationTransformer
from common_src.model.middle_encoders.HyDRa.SE_fusion import SEFusion
from common_src.model.middle_encoders.HyDRa.back_projection import LidarDepthRefiner
from common_src.model.middle_encoders.HyDRa.zeroinit import ZeroInitLayerNorm
from common_src.model.backbones.img_backbone import ImageBackbone

class CenterPoint(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
    
        self.img_shape = torch.tensor([1936 , 1216])
        self.data_root = config.get('data_root', None)
        self.class_names = config.get('class_names', None)
        self.output_dir = config.get('output_dir', None)
        self.pc_range = torch.tensor(config.get('point_cloud_range', None))
        
        voxel_layer_config = config.get('pts_voxel_layer', None)
        voxel_encoder_config = config.get('voxel_encoder', None)
        middle_encoder_config = config.get('middle_encoder', None)

        #Added
        hat_config = config.get('hat', None)
        se_fusion_config = config.get('se_fusion', None)
        ldc_config = config.get('ldc', None)

        backbone_config = config.get('backbone', None)
        neck_config = config.get('neck', None)
        head_config = config.get('head', None)
        
        self.voxel_layer = Voxelization(**voxel_layer_config)
        self.voxel_encoder = PillarFeatureNet(**voxel_encoder_config)
        self.middle_encoder = PointPillarsScatter(**middle_encoder_config)
        self.backbone = SECOND(**backbone_config)
        self.image_backbone = ImageBackbone(out_channels=64)
        self.lidar_channel_expansion = torch.nn.Linear(4, 64)  # Expand LiDAR features to 64 channels
        self.zero_init_layer_norm = ZeroInitLayerNorm(64)  # Initialize LayerNorm with zero weights

        self.depth_pos_encoding = torch.nn.Parameter(torch.randn(76, 4))  # Example depth encoding, adjust as needed
        self.neck = SECONDFPN(**neck_config)
        self.head = CenterHead(**head_config)
        
        self.optimizer_config = config.get('optimizer', None)
        
        self.vod_kitti_locations = KittiLocations(root_dir = self.data_root, 
                                     output_dir= self.output_dir,
                                     frame_set_path='',
                                     pred_dir='',)
        self.inference_mode = config.get('inference_mode', 'val')
        self.save_results = config.get('save_preds_results', False)
        self.val_results_list =[]

        print("HAT Config:", config.get('hat', None))
        self.hat = HeightAssociationTransformer(hat_config)
        print(se_fusion_config)
        self.se_fusion = SEFusion(se_fusion_config)
        print(ldc_config)
        self.ldc = LidarDepthRefiner(ldc_config)

        
    ## Voxelization
    def voxelize(self, points):
        voxel_dict = dict()
        voxels, coors, num_points = [], [], []
        for i, res in enumerate(points):
            res_voxels, res_coors, res_num_points = self.voxel_layer(res.cuda())
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)

        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors = torch.cat(coors, dim=0)

        voxel_dict['voxels'] = voxels
        voxel_dict['num_points'] = num_points
        voxel_dict['coors'] = coors

        return voxel_dict
    
    def _model_forward(self, lidar_data=None, img_data=None, frame_idx=None):

        #voxel_dict = self.voxelize(pts_data)
    
        # voxels = voxel_dict['voxels']
        # num_points = voxel_dict['num_points']
        # coors = voxel_dict['coors']
    
        # voxel_features = self.voxel_encoder(voxels, num_points, coors)
        # bs = coors[-1,0].item() + 1
        # bev_feats_lidar = self.middle_encoder(voxel_features, coors, bs)

        
        # Image path (skip if img_data is None)
        # if img_data is not None:
        # 1. Extract image features
        print(f"Image data shape: {img_data[0].shape}")

        img_data = torch.stack(img_data, dim=0)  # Stack the list of image tensors along a new dimension

        print(f"Image data shape after stacking: {img_data.shape}")  # [B, N, C, H, W]
        image_queries, H_feat, W_feat = self.img_features(img_data)  # [B, N, H, W, C] -> [B*N*H, W, C]
        print(f"Image queries shape: {image_queries.shape}")  # [B*N*H, W, C]
        W = image_queries[2]
        lidar_features= []
        for pc in lidar_data:
            # 2. Extract LiDAR features
            lidar_pc_lidar = pc[:, :4]
            lidar_features.append(self.extract_lidar_uvz_features_torch(lidar_pc_lidar,
                frame_idx=frame_idx,
                image_shape=(H_feat, W_feat),
                feature_dim=4,
                z_max=100.0,
                num_D_bins=76))
            print(f"Extracted LiDAR features shape: {lidar_features[-1].shape}")  # [1, 1, 1, W, D, C]
        # lidar_features = self.extract_lidar_uvz_features_torch(lidar_pc_lidar, frame_idx, (H_feat, W_feat), feature_dim=4)

        binned_lidar = torch.stack(lidar_features, dim=0)  # [B, W, D, C]
        B_lidar, W_lidar, D_lidar, C_lidar = binned_lidar.shape
        lidar_seq = binned_lidar.view(B_lidar*W_lidar, D_lidar, C_lidar)
        print(f"Binned LiDAR shape: {binned_lidar.shape}")

        # Perform positional encoding on LiDAR features
        lidar_seq += self.depth_pos_encoding  # Add depth positional encoding


        # Apply the layer to expand LiDAR features to 64 channels
        lidar_seq = self.lidar_channel_expansion(lidar_seq)
        
        # Perform cross-attention between image queries and LiDAR sequence
        fused_bev = self.hat(image_queries, lidar_seq)
        delta = self.zero_init_layer_norm(fused_bev)  # Apply zero-initialized LayerNorm
        fused_bev = image_queries + delta  # Add residual connection
        BW, H, C = fused_bev.shape
        W = 121
        B = BW // W
        fused_bev = fused_bev.view(B, C, H, W)  # Reshape to [B, C, H, W]
        print(f"Fused BEV shape: {fused_bev.shape}")  # [B, C, H, W]
        backbone_feats = self.backbone(fused_bev)
        neck_feats = self.neck(backbone_feats)
        ret_dict = self.head(neck_feats)
        return ret_dict
    
    def training_step(self, batch, batch_idx):
        print(f"Training step {batch_idx} with batch size {len(batch['pts'])}")
        lidar_pts = batch['lidar_data']
        img_data = batch['stereo_camera']
        num_frame = batch['metas'][0]['num_frame']
        gt_label_3d = batch['gt_labels_3d']
        gt_bboxes_3d = batch['gt_bboxes_3d']
        print(f"Training step batch input img_data shape; {img_data[0].shape}")
        ret_dict = self._model_forward(lidar_pts, img_data, num_frame)
        loss_input = [gt_bboxes_3d, gt_label_3d, ret_dict]
        
        losses = self.head.loss(*loss_input)
        
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            log_vars[loss_name] = loss_value.mean()
        
        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
            self.log(f'train/{loss_name}', loss_value, batch_size=1)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_config)
        return optimizer
    
    
    def validation_step(self, batch, batch_idx):
        assert len(batch['lidar_data']) == 1, 'Batch size should be 1 for validation'
        lidar_pts = batch['lidar_data']
        img_data = batch['stereo_camera']
        num_frame = batch['metas'][0]['num_frame']
        gt_label_3d = batch['gt_labels_3d']
        gt_bboxes_3d = batch['gt_bboxes_3d']
        print(f"Validation step batch input img_data shape; {img_data[0].shape}")
        print(f"Validation step batch length: {len(batch['lidar_data'])}, num_frame: {num_frame}") 
        ret_dict = self._model_forward(lidar_pts, img_data, num_frame)
        loss_input = [gt_bboxes_3d, gt_label_3d, ret_dict]
        
        bbox_list = self.head.get_bboxes(ret_dict, img_metas=metas)
        
        bbox_results = [
            dict(bboxes_3d = bboxes, 
                 scores_3d = scores, 
                 labels_3d = labels)
            for bboxes, scores, labels in bbox_list
            ]

        losses = self.head.loss(*loss_input)
        
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            log_vars[loss_name] = loss_value.mean()
        
        val_loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        log_vars['loss'] = val_loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
            self.log(f'validation/{loss_name}', loss_value, batch_size=1, sync_dist=True)
        # task0.loss_heatmap', 'task0.loss_bbox', 'task1.loss_heatmap', 'task1.loss_bbox', 'task2.loss_heatmap', 'task2.loss_bbox', 'loss'
        self.val_results_list.append(dict(
            sample_idx = batch['metas'][0]['num_frame'],
            input_batch = batch,
            bbox_results = bbox_results,
            losses = log_vars
        ))

    
    def extract_lidar_uvz_features(self, lidar_pc_lidar, frame_idx, image_shape, feature_dim=4):
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

        H, W = 1216,1936
        vod_frame_data = FrameDataLoader(kitti_locations=self.vod_kitti_locations, frame_number=frame_idx)
        local_transforms = FrameTransformMatrix(vod_frame_data)
        transform_matrix = local_transforms.t_camera_lidar
        projection_matrix = local_transforms.camera_projection_matrix

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

        feats_valid = lidar_pc_camera[final_mask, :feature_dim]  # [N_valid, C]

        u_valid = u[final_mask].reshape(-1, 1)  # Extract valid u values
        z_valid = z[final_mask].reshape(-1, 1)  # Extract valid z values

        num_W_bins = image_shape[1]  # W
        print(f"Number of width bins: {num_W_bins}")
        num_D_bins = 76  # D
        z_max = 100
        depth_bin_size = z_max / num_D_bins

        # Clip depth values to the fixed range
        z_valid = np.clip(z_valid, 0, z_max)

        # Step 4: Bin the valid points width and depth wise
        binned_features = np.zeros((num_W_bins, num_D_bins, feature_dim), dtype=np.float32)
        binned_features = torch.tensor(binned_features)  # If it's NumPy
        bin_counts = np.zeros((num_W_bins, num_D_bins), dtype=np.float32)

        depth_bin_size = z_max / num_D_bins

        # Extract valid u and v coordinates
        v_valid = v[final_mask].reshape(-1, 1)

        # Load the image for visualization
        
        # Bin the valid points width and depth-wise
        for i in range(u_valid.shape[0]):
            u_bin = u_valid[i, 0]
            d_bin = int(z_valid[i, 0] / depth_bin_size)

            if 0 <= u_bin < num_W_bins and 0 <= d_bin < num_D_bins:
                if bin_counts[0, 0, 0, u_bin, d_bin] < 10:  # Limit max points per bin
                    binned_features[0, 0, 0, u_bin, d_bin] += feats_valid[i]
                    bin_counts[0, 0, 0, u_bin, d_bin] += 1

        nonzero_mask = bin_counts > 0
        binned_features[nonzero_mask] /= bin_counts[nonzero_mask].reshape(-1, 1)

        return binned_features
    
    import torch

    def extract_lidar_uvz_features_torch(self,
        lidar_pc_lidar: torch.Tensor,           # (N, 4 + C)  float32 / float64
        frame_idx: int,
        image_shape: tuple,                     # (H, W)
        feature_dim: int = 4,
        z_max: float = 100.0,
        num_D_bins: int = 76,
    ):
        """
        Pure-PyTorch version of extract_lidar_uvz_features.

        Returns
        -------
        binned_features : torch.Tensor
            shape = (1, 1, 1, W, D, feature_dim)
        """

        H, W = image_shape  # e.g. (1216, 1936)

        # ------------------------------------------------------------------
        # 1.  Gather per-frame transforms — keep these as tensors.
        # ------------------------------------------------------------------
        vod_frame_data  = FrameDataLoader(kitti_locations=self.vod_kitti_locations,
                                        frame_number=frame_idx)

        local_transforms = FrameTransformMatrix(vod_frame_data)
        T_cam_lidar      = torch.as_tensor(local_transforms.t_camera_lidar,      # (4,4)
                                        dtype=lidar_pc_lidar.dtype,
                                        device=lidar_pc_lidar.device)
        P_cam            = torch.as_tensor(local_transforms.camera_projection_matrix,  # (3,4)
                                        dtype=lidar_pc_lidar.dtype,
                                        device=lidar_pc_lidar.device)

        # ------------------------------------------------------------------
        # 2.  Homogeneous transform LiDAR → camera.
        # ------------------------------------------------------------------
        # cat(1) keeps any extra feature channels intact
        ones = torch.ones((lidar_pc_lidar.shape[0], 1),
                        dtype=lidar_pc_lidar.dtype,
                        device=lidar_pc_lidar.device)
        pts_h = torch.cat([lidar_pc_lidar[:, :3], ones], dim=1)   # (N,4)
        pts_cam = (T_cam_lidar @ pts_h.T).T                       # (N,4)

        # keep intensity + C extra features from original tensor
        feats = lidar_pc_lidar[:, :feature_dim]

        # ------------------------------------------------------------------
        # 3.  Project to image plane.
        # ------------------------------------------------------------------
        pixels_h = (P_cam @ pts_cam.T).T                          # (N,3)
        z = pixels_h[:, 2]                                        # (N,)
        valid = z > 0

        # u = x/z, v = y/z
        u = (pixels_h[:, 0] / z).long()
        v = (pixels_h[:, 1] / z).long()

        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        keep = valid & in_bounds

        u_valid = u[keep]
        z_valid = torch.clamp(z[keep], 0, z_max)
        feats_valid = feats[keep]

        # ------------------------------------------------------------------
        # 4.  Pre-allocate bins  (B, C, Z, W, D) = (1,1,1,W,D)
        # ------------------------------------------------------------------
        num_W_bins = W
        print(f"Number of width bins: {num_W_bins}")
        depth_bin = z_max / num_D_bins

        binned_features = torch.zeros((num_W_bins, num_D_bins, feature_dim),
                                    dtype=lidar_pc_lidar.dtype,
                                    device=lidar_pc_lidar.device)
        bin_counts = torch.zeros((num_W_bins, num_D_bins),
                                dtype=lidar_pc_lidar.dtype,
                                device=lidar_pc_lidar.device)

        # ------------------------------------------------------------------
        # 5.  Scatter-add each point into its (u_bin, d_bin).
        # ------------------------------------------------------------------
        for idx in range(u_valid.numel()):
            u_bin = u_valid[idx]
            d_bin = int(z_valid[idx] / depth_bin)

            if bin_counts[u_bin, d_bin] < 10:           # cap per-bin points
                binned_features[u_bin, d_bin] += feats_valid[idx]
                bin_counts     [u_bin, d_bin] += 1

        # ------------------------------------------------------------------
        # 6.  Average non-empty bins.
        # ------------------------------------------------------------------
        nonzero = bin_counts > 0
        binned_features[nonzero] /= bin_counts[nonzero].unsqueeze(-1)

        return binned_features

    
    def img_features(self, img_data):
        img_feats = self.image_backbone(img_data)  # [B, N, H, W, C]
        B, C_out, H_feat, W_feat = img_feats.shape
        # Reshape to [B, N, H, W, C] where B=1, N=1 (single camera), H=H, W=W, C=C

        height_queries = img_feats.view(B * W_feat, H_feat, C_out)
        print(f"Height queries shape: {height_queries.shape}")
        return height_queries, H_feat, W_feat

    

    def on_validation_epoch_end(self):
        if (not self.save_results) or self.training: 
            tmp_dir = tempfile.TemporaryDirectory()
            working_dir = tmp_dir.name
        else:
            tmp_dir = None
            working_dir = self.output_dir

        preds_dst = os.path.join(working_dir, f'{self.inference_mode}_preds')
        os.makedirs(preds_dst, exist_ok=True)
        
        outputs = self.val_results_list
        self.val_results_list = []
        results = self.format_results(outputs, results_save_path=preds_dst)
        
        if self.inference_mode =='val': 
            gt_dst = os.path.join(self.data_root, 'lidar', 'training', 'label_2')
            
            evaluation = Evaluation(test_annotation_file=gt_dst)
            results = evaluation.evaluate(result_path=preds_dst, current_class=[0, 1, 2])
            
            self.log('validation/entire_area/Car_3d', results['entire_area']['Car_3d_all'], batch_size=1, sync_dist=True)
            self.log('validation/entire_area/Pedestrian_3d', results['entire_area']['Pedestrian_3d_all'], batch_size=1, sync_dist=True)
            self.log('validation/entire_area/Cyclist_3d', results['entire_area']['Cyclist_3d_all'], batch_size=1, sync_dist=True)
            self.log('validation/entire_area/mAP', (results['entire_area']['Car_3d_all'] + results['entire_area']['Pedestrian_3d_all'] + results['entire_area']['Cyclist_3d_all']) / 3, batch_size=1, sync_dist=True)
            self.log('validation/ROI/Car_3d', results['roi']['Car_3d_all'], batch_size=1, sync_dist=True)
            self.log('validation/ROI/Pedestrian_3d', results['roi']['Pedestrian_3d_all'], batch_size=1, sync_dist=True)
            self.log('validation/ROI/Cyclist_3d', results['roi']['Cyclist_3d_all'], batch_size=1, sync_dist=True)
            self.log('validation/ROI/mAP', (results['roi']['Car_3d_all'] + results['roi']['Pedestrian_3d_all'] + results['roi']['Cyclist_3d_all']) / 3, batch_size=1, sync_dist=True)
        
            print("Results: \n"
                f"Entire annotated area: \n"
                f"Car: {results['entire_area']['Car_3d_all']} \n"
                f"Pedestrian: {results['entire_area']['Pedestrian_3d_all']} \n"
                f"Cyclist: {results['entire_area']['Cyclist_3d_all']} \n"
                f"mAP: {(results['entire_area']['Car_3d_all'] + results['entire_area']['Pedestrian_3d_all'] + results['entire_area']['Cyclist_3d_all']) / 3} \n"
                f"Driving corridor area: \n"
                f"Car: {results['roi']['Car_3d_all']} \n"
                f"Pedestrian: {results['roi']['Pedestrian_3d_all']} \n"
                f"Cyclist: {results['roi']['Cyclist_3d_all']} \n"
                f"mAP: {(results['roi']['Car_3d_all'] + results['roi']['Pedestrian_3d_all'] + results['roi']['Cyclist_3d_all']) / 3} \n"
                )
            
        if isinstance(tmp_dir, tempfile.TemporaryDirectory):
            tmp_dir.cleanup() 
        return results
        
        # detection_annotation_file = results_path
        
    def format_results(self, 
                       outputs, 
                       results_save_path=None,
                       pklfile_prefix=None):
        
        det_annos = []
        print('\nConverting prediction to KITTI format')
        print(f'Writing results to {results_save_path}')
        for result in outputs:
            sample_idx = result['sample_idx']
            res_dict = result['bbox_results']
            input_batch = result['input_batch']
            
            annos = []
            box_dict = self.convert_valid_bboxes(res_dict[0], input_batch)
            
            anno = {                 
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': [],
            }
            
            if len(box_dict['box2d']) > 0:
                box2d_preds = box_dict['box2d']
                box3d_preds_lidar = box_dict['box3d_lidar']
                box3d_location_cam = box_dict['location_cam']
                scores = box_dict['scores']
                label_preds = box_dict['label_preds']
                
                for box3d_lidar, location_cam, box2d, score, label in zip(box3d_preds_lidar, box3d_location_cam, box2d_preds, scores, label_preds):                                      
                    box2d[2:] = np.minimum(box2d[2:], self.img_shape.cpu().numpy()[:2])
                    box2d[:2] = np.maximum(box2d[:2], [0, 0])
                    anno['name'].append(self.class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    #anno['alpha'].append(limit_period(np.arctan2(location_cam[2], location_cam[0]) + box3d_lidar[6] - np.pi/2, offset=0.5, period=2*np.pi))
                    anno['alpha'].append(np.arctan2(location_cam[2], location_cam[0]) + box3d_lidar[6] - np.pi/2)
                    anno['bbox'].append(box2d)
                    anno['dimensions'].append(box3d_lidar[3:6])
                    anno['location'].append(location_cam[:3])
                    anno['rotation_y'].append(box3d_lidar[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }
                annos.append(anno)
            
            if results_save_path is not None:
                curr_file = f'{results_save_path}/{sample_idx}.txt'
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lwh -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                bbox[idx][3], dims[idx][2], dims[idx][1],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f)
                
            annos[-1]['sample_idx'] = np.array([sample_idx] * len(annos[-1]['score']), dtype=np.int64)
            det_annos += annos
        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            with open(out, "wb") as f:
                pickle.dump(det_annos, f)
            print(f'Result is saved to {out}.')
        return det_annos
        
    def convert_valid_bboxes(self, box_dict, input_batch):
        # Convert the predicted bounding boxes to the format required by the evaluation metric
        # This function should be implemented based on the specific requirements of your dataset
        box_preds = box_dict['bboxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = input_batch['metas'][0]['num_frame']
        
        vod_frame_data = FrameDataLoader(kitti_locations=self.vod_kitti_locations, frame_number=sample_idx)
        local_transforms = FrameTransformMatrix(vod_frame_data)
        
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)
        device = box_preds.tensor.device
                
        box_preds_corners_lidar = box_preds.corners
        box_preds_bottom_center_lidar = box_preds.bottom_center # box_preds.gravity_center
        # box_preds_gravity_center_lidar = box_preds.gravity_center
        
        box_preds_corners_img_list = [] 
        box_preds_bottom_center_cam_list =[]
        
        for box_pred_corners, box_pred_bottom_center in zip(box_preds_corners_lidar, box_preds_bottom_center_lidar):
            
            box_pred_corners_lidar_homo= torch.ones((8,4))
            box_pred_corners_lidar_homo[:, :3] = box_pred_corners
            box_pred_corners_cam_homo = homogeneous_transformation(box_pred_corners_lidar_homo, local_transforms.t_camera_lidar)
            box_pred_corners_img = np.dot(box_pred_corners_cam_homo, local_transforms.camera_projection_matrix.T)
            box_pred_corners_img = torch.tensor((box_pred_corners_img[:, :2].T / box_pred_corners_img[:, 2]).T, device=device)
            box_preds_corners_img_list.append(box_pred_corners_img)

            box_pred_bottom_center_lidar_homo = torch.ones((1,4))
            box_pred_bottom_center_lidar_homo[:, :3] = box_pred_bottom_center
            box_pred_bottom_center_cam_homo = homogeneous_transformation(box_pred_bottom_center_lidar_homo, local_transforms.t_camera_lidar)
            box_pred_bottom_center_cam = torch.tensor(box_pred_bottom_center_cam_homo[:,:3])
            box_preds_bottom_center_cam_list.append(box_pred_bottom_center_cam)

        if box_preds_corners_img_list != []:
            box_preds_corners_img = torch.stack(box_preds_corners_img_list, dim=0)
            assert box_preds_bottom_center_cam_list != []
            box_preds_bottom_center_cam = torch.cat(box_preds_bottom_center_cam_list, dim=0).to(device)
        
            minxy = torch.min(box_preds_corners_img, dim=1)[0]
            maxxy = torch.max(box_preds_corners_img, dim=1)[0]
            box_2d_preds = torch.cat([minxy, maxxy], dim=1)

            self.img_shape = self.img_shape.to(device)
            self.pc_range = self.pc_range.to(device)
            
            valid_cam_inds = ((box_2d_preds[:, 0] < self.img_shape[0]) & (box_2d_preds[:, 1] < self.img_shape[1]) & (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
            valid_pcd_inds = ((box_preds.center > self.pc_range[:3]) & (box_preds.center < self.pc_range[3:]))
            valid_inds = valid_cam_inds & valid_pcd_inds.all(-1)
            
            if valid_inds.sum() > 0:
                return dict(
                    box2d=box_2d_preds[valid_inds, :].cpu().numpy(),
                    location_cam=box_preds_bottom_center_cam[valid_inds].cpu().numpy(),
                    box3d_lidar=box_preds[valid_inds].tensor.cpu().numpy(),
                    scores=scores[valid_inds].cpu().numpy(),
                    label_preds=labels[valid_inds].cpu().numpy(),
                    sample_idx=sample_idx)
            else:
                return dict(
                    box2d=np.zeros([0, 4]),
                    location_cam=np.zeros([0, 3]),
                    # box3d_camera_corners=np.zeros([0, 7]),
                    box3d_lidar=np.zeros([0, 7]),
                    scores=np.zeros([0]),
                    label_preds=np.zeros([0, 4]),
                    sample_idx=sample_idx)
        else:
            return dict(
                box2d=np.zeros([0, 4]),
                location_cam=np.zeros([0, 3]),
                # box3d_camera_corners=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

# if __name__ == '__main__':
#     import sys
#     import os
#     import yaml

#     # Add the parent directory to the Python path
#     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
#     print("Yaaaaa")
#     # Load configuration from YAML file
#     config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/model/centerpoint.yaml'))
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)

#     from common_src.dataset import ViewOfDelft, FrameDataLoader
#     centerpoint = CenterPoint(config=config)

#     dataset = ViewOfDelft(data_root='data/view_of_delft', split='train')
#     vod_frame_data = FrameDataLoader(kitti_locations=dataset.vod_kitti_locations, 
#                                      frame_number='01545')
    
#     img_data = vod_frame_data.image  # [H, W, C]
#     pts_data = vod_frame_data.lidar  # [N, 4 + C]

#     ret_dict = centerpoint._model_forward(
#         pts_data=[pts_data], 
#         img_data=[img_data], 
#         batch_idx=1545
#     )