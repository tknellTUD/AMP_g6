def collate_vod_batch(batch):
    pts_list = []
    gt_labels_3d_list = []
    gt_bboxes_3d_list = []
    meta_list = []
    img_list = []
    for idx, sample in enumerate(batch):
        pts_list.append(sample['lidar_data'])
        img_list.append(sample['stereo_camera'])
        gt_labels_3d_list.append(sample['gt_labels_3d'])
        gt_bboxes_3d_list.append(sample['gt_bboxes_3d'])
        meta_list.append(sample['meta'])
    return dict(
        lidar_data = pts_list,
        stereo_camera = img_list,
        gt_labels_3d = gt_labels_3d_list,
        gt_bboxes_3d = gt_bboxes_3d_list,
        metas = meta_list
    )
    