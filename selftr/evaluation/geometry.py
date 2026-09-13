"""Portable geometry and depth metrics for SelfTR evaluation."""

from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def rotation_error_degrees(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Geodesic errors for corresponding rotation matrices, in degrees."""

    delta = np.asarray(reference) @ np.swapaxes(np.asarray(prediction), -1, -2)
    cosine = np.clip((np.trace(delta, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _pose_homogeneous(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape[-2:] == (4, 4):
        return poses
    if poses.shape[-2:] == (3, 4):
        bottom = np.zeros((*poses.shape[:-2], 1, 4), dtype=np.float64)
        bottom[..., 0, 3] = 1.0
        return np.concatenate((poses, bottom), axis=-2)
    raise ValueError(f"Expected Nx4x4 or Nx3x4 poses, got {poses.shape}")


def umeyama(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation, translation mapping 3xN points *x* to *y*."""

    mu_x = x.mean(axis=1, keepdims=True)
    mu_y = y.mean(axis=1, keepdims=True)
    variance = np.square(x - mu_x).sum(axis=0).mean()
    covariance = ((y - mu_y) @ (x - mu_x).T) / x.shape[1]
    left, values, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left) * np.linalg.det(right) < 0:
        correction[-1, -1] = -1
    scale = float(np.trace(np.diag(values) @ correction) / max(variance, 1e-12))
    rotation = left @ correction @ right
    translation = mu_y - scale * rotation @ mu_x
    return scale, rotation, translation


def trajectory_pose_metrics(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> dict[str, float]:
    """Compute Sim(3)-aligned trajectory and relative-pose diagnostics."""

    pred, gt = _pose_homogeneous(pred_c2w), _pose_homogeneous(gt_c2w)
    count = min(len(pred), len(gt))
    if count < 2:
        return {key: float("nan") for key in (
            "ate_rmse_m", "are_deg", "rpe_translation_rmse_m", "rpe_rotation_rmse_deg",
            "rra_30_percent", "rta_30_percent",
        )}
    pred, gt = pred[:count].copy(), gt[:count]
    scale, rotation, translation = umeyama(pred[:, :3, 3].T, gt[:, :3, 3].T)
    pred[:, :3, 3] = scale * (pred[:, :3, 3] @ rotation.T) + translation.reshape(1, 3)
    pred[:, :3, :3] = rotation[None] @ pred[:, :3, :3]
    absolute_rotation = rotation_error_degrees(gt[:, :3, :3], pred[:, :3, :3])
    ate = np.linalg.norm(pred[:, :3, 3] - gt[:, :3, 3], axis=1)
    pred_w2c, gt_w2c = np.linalg.inv(pred), np.linalg.inv(gt)
    pair_rotation, pair_translation, rpe_rotation, rpe_translation = [], [], [], []
    for first in range(count - 1):
        for second in range(first + 1, count):
            pred_relative = pred_w2c[first] @ np.linalg.inv(pred_w2c[second])
            gt_relative = gt_w2c[first] @ np.linalg.inv(gt_w2c[second])
            pair_rotation.append(float(rotation_error_degrees(
                gt_relative[:3, :3][None], pred_relative[:3, :3][None]
            )[0]))
            gt_translation, pred_translation = gt_relative[:3, 3], pred_relative[:3, 3]
            denominator = np.linalg.norm(gt_translation) * np.linalg.norm(pred_translation)
            pair_translation.append(180.0 if denominator <= 1e-15 else float(np.degrees(
                np.arccos(np.clip(abs(np.dot(gt_translation, pred_translation)) / denominator, 0.0, 1.0))
            )))
            if second == first + 1:
                rpe_rotation.append(pair_rotation[-1])
                rpe_translation.append(float(np.linalg.norm(gt_relative[:3, 3] - pred_relative[:3, 3])))
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(np.square(ate)))),
        "are_deg": float(np.mean(absolute_rotation)),
        "rpe_translation_rmse_m": float(np.sqrt(np.mean(np.square(rpe_translation)))),
        "rpe_rotation_rmse_deg": float(np.sqrt(np.mean(np.square(rpe_rotation)))),
        "rra_30_percent": float(100.0 * np.mean(np.asarray(pair_rotation) < 30.0)),
        "rta_30_percent": float(100.0 * np.mean(np.asarray(pair_translation) < 30.0)),
    }


def depth_error_metrics(prediction: np.ndarray, ground_truth: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """Pixel-weighted monocular-depth metrics after the chosen alignment."""

    prediction = np.asarray(prediction, dtype=np.float64)[valid]
    ground_truth = np.asarray(ground_truth, dtype=np.float64)[valid]
    if not len(prediction):
        raise ValueError("No valid depth pixels")
    difference = prediction - ground_truth
    ratio = np.maximum(prediction / ground_truth, ground_truth / prediction)
    return {
        "abs_rel": float(np.mean(np.abs(difference) / ground_truth)),
        "sq_rel": float(np.mean(difference**2 / ground_truth)),
        "rmse_m": float(np.sqrt(np.mean(difference**2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(prediction) - np.log(ground_truth)) ** 2))),
        "mae_m": float(np.mean(np.abs(difference))),
        "delta_1_25_percent": float(100 * np.mean(ratio < 1.25)),
        "delta_1_25_sq_percent": float(100 * np.mean(ratio < 1.25**2)),
        "delta_1_25_cu_percent": float(100 * np.mean(ratio < 1.25**3)),
        "valid_depth_pixels": float(len(prediction)),
    }


def depth_to_world_points(depth: np.ndarray, c2w: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project ``[frames, height, width]`` depth maps into world points."""

    frames, height, width = depth.shape
    ys, xs = np.mgrid[0:height, 0:width]
    points = np.empty((frames, height, width, 3), dtype=np.float32)
    for frame in range(frames):
        intrinsic, z = intrinsics[frame], depth[frame]
        camera = np.stack(((xs - intrinsic[0, 2]) * z / intrinsic[0, 0],
                           (ys - intrinsic[1, 2]) * z / intrinsic[1, 1], z), axis=-1)
        points[frame] = camera @ c2w[frame, :3, :3].T + c2w[frame, :3, 3]
    return points


def scaled_intrinsics(
    intrinsic: np.ndarray,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
    frames: int,
) -> np.ndarray:
    """Scale an intrinsic matrix after direct image/depth resizing."""

    source_height, source_width = source_hw
    target_height, target_width = target_hw
    output = np.repeat(np.asarray(intrinsic, dtype=np.float64)[None], frames, axis=0)
    output[:, 0, :] *= target_width / source_width
    output[:, 1, :] *= target_height / source_height
    output[:, 2, 2] = 1.0
    return output


def evaluate_pi3_geometry(
    pred_point_map: np.ndarray,
    gt_point_map: np.ndarray,
    valid_mask: np.ndarray,
    *,
    max_points: int = 100_000,
    icp_threshold_m: float = 0.1,
) -> dict[str, float | int]:
    """Compute deterministic Pi3-compatible ACC, Comp, NC, and Chamfer metrics."""

    valid_ids = np.flatnonzero(np.asarray(valid_mask).reshape(-1))
    if len(valid_ids) < 3:
        raise ValueError("geometry evaluation needs at least three valid points")
    selected_ids = valid_ids[np.linspace(0, len(valid_ids) - 1, min(max_points, len(valid_ids)), dtype=np.int64)]
    pred = np.asarray(pred_point_map).reshape(-1, 3)[selected_ids].astype(np.float64, copy=False)
    gt = np.asarray(gt_point_map).reshape(-1, 3)[selected_ids].astype(np.float64, copy=False)
    finite = np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
    pred, gt = pred[finite], gt[finite]
    if len(pred) < 3:
        raise ValueError("geometry evaluation needs at least three finite corresponding points")
    scale, rotation, translation = umeyama(pred.T, gt.T)
    pred = scale * (pred @ rotation.T) + translation.reshape(1, 3)
    pred_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred))
    gt_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt))
    alignment = o3d.pipelines.registration.registration_icp(
        pred_cloud, gt_cloud, icp_threshold_m, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pred_cloud.transform(alignment.transformation)
    pred_cloud.estimate_normals()
    gt_cloud.estimate_normals()
    pred, gt = np.asarray(pred_cloud.points), np.asarray(gt_cloud.points)
    pred_normals, gt_normals = np.asarray(pred_cloud.normals), np.asarray(gt_cloud.normals)
    gt_tree = cKDTree(gt)
    acc_distance, acc_index = gt_tree.query(pred, workers=-1)
    pred_tree = cKDTree(pred)
    comp_distance, comp_index = pred_tree.query(gt, workers=-1)
    normal_1 = np.abs(np.sum(pred_normals * gt_normals[acc_index], axis=-1))
    normal_2 = np.abs(np.sum(gt_normals * pred_normals[comp_index], axis=-1))
    return {
        "acc_mean_m": float(np.mean(acc_distance)), "acc_median_m": float(np.median(acc_distance)),
        "comp_mean_m": float(np.mean(comp_distance)), "comp_median_m": float(np.median(comp_distance)),
        "nc_mean": float((np.mean(normal_1) + np.mean(normal_2)) / 2),
        "nc_median": float((np.median(normal_1) + np.median(normal_2)) / 2),
        "nc1_mean": float(np.mean(normal_1)), "nc1_median": float(np.median(normal_1)),
        "nc2_mean": float(np.mean(normal_2)), "nc2_median": float(np.median(normal_2)),
        "chamfer_mean_m": float((np.mean(acc_distance) + np.mean(comp_distance)) / 2),
        "chamfer_median_m": float((np.median(acc_distance) + np.median(comp_distance)) / 2),
        "geometry_points": int(len(pred)), "sim3_scale": float(scale),
    }
