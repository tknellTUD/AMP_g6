#include <torch/extension.h>

#include "pytorch_cpp_helper.hpp"

std::vector<torch::Tensor> dynamic_point_to_voxel_forward(
    const torch::Tensor &feats, const torch::Tensor &coors,
    const std::string &reduce_type);

void dynamic_point_to_voxel_backward(torch::Tensor &grad_feats,
                                     const torch::Tensor &grad_reduced_feats,
                                     const torch::Tensor &feats,
                                     const torch::Tensor &reduced_feats,
                                     const torch::Tensor &coors_idx,
                                     const torch::Tensor &reduce_count,
                                     const std::string &reduce_type);

void dynamic_voxelize_forward(const at::Tensor &points,
                              const at::Tensor &voxel_size,
                              const at::Tensor &coors_range, at::Tensor &coors,
                              const int NDim);

void hard_voxelize_forward(const at::Tensor &points,
                           const at::Tensor &voxel_size,
                           const at::Tensor &coors_range, at::Tensor &voxels,
                           at::Tensor &coors, at::Tensor &num_points_per_voxel,
                           at::Tensor &voxel_num, const int max_points,
                           const int max_voxels, const int NDim,
                           const bool deterministic);


int boxes_overlap_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_overlap);
int boxes_iou_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_iou);
int nms_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh, int device_id);
int nms_normal_gpu(at::Tensor boxes, at::Tensor keep,float nms_overlap_thresh, int device_id);
int points_in_boxes_gpu(at::Tensor boxes_tensor, at::Tensor pts_tensor, at::Tensor box_idx_of_points_tensor);
int points_in_boxes_batch(at::Tensor boxes_tensor, at::Tensor pts_tensor, at::Tensor box_idx_of_points_tensor);
int points_in_boxes_cpu(at::Tensor boxes_tensor, at::Tensor pts_tensor,at::Tensor pts_indices_tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dynamic_point_to_voxel_forward", &dynamic_point_to_voxel_forward,
        "dynamic_point_to_voxel_forward", py::arg("feats"), py::arg("coors"),
        py::arg("reduce_type"));
  m.def("dynamic_point_to_voxel_backward", &dynamic_point_to_voxel_backward,
        "dynamic_point_to_voxel_backward", py::arg("grad_feats"),
        py::arg("grad_reduced_feats"), py::arg("feats"),
        py::arg("reduced_feats"), py::arg("coors_idx"), py::arg("reduce_count"),
        py::arg("reduce_type"));
  m.def("hard_voxelize_forward", &hard_voxelize_forward,
        "hard_voxelize_forward", py::arg("points"), py::arg("voxel_size"),
        py::arg("coors_range"), py::arg("voxels"), py::arg("coors"),
        py::arg("num_points_per_voxel"), py::arg("voxel_num"),
        py::arg("max_points"), py::arg("max_voxels"), py::arg("NDim"),
        py::arg("deterministic"));
  m.def("dynamic_voxelize_forward", &dynamic_voxelize_forward,
        "dynamic_voxelize_forward", py::arg("points"), py::arg("voxel_size"),
        py::arg("coors_range"), py::arg("coors"), py::arg("NDim"));

  m.def("boxes_overlap_bev_gpu", &boxes_overlap_bev_gpu,"oriented boxes overlap");
  m.def("boxes_iou_bev_gpu", &boxes_iou_bev_gpu, "oriented boxes iou");
  m.def("nms_gpu", &nms_gpu, "oriented nms gpu");
  m.def("nms_normal_gpu", &nms_normal_gpu, "nms gpu");
  m.def("points_in_boxes_gpu", &points_in_boxes_gpu, "points_in_boxes_gpu forward (CUDA)");
  m.def("points_in_boxes_batch", &points_in_boxes_batch, "points_in_boxes_batch forward (CUDA)");
  m.def("points_in_boxes_cpu", &points_in_boxes_cpu,"points_in_boxes_cpu forward (CPU)");
}