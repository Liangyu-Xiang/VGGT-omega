"""Pi3-style aligned point-map, depth and trajectory metrics."""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def rotation_error_degrees(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
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
    """Return scale, rotation and translation so scale * rotation @ x + translation approximates y."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu_x = x.mean(axis=1, keepdims=True)
    mu_y = y.mean(axis=1, keepdims=True)
    var_x = np.square(x - mu_x).sum(axis=0).mean()
    cov_xy = ((y - mu_y) @ (x - mu_x).T) / x.shape[1]
    u, d, vh = np.linalg.svd(cov_xy)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        sign[-1, -1] = -1
    scale = float(np.trace(np.diag(d) @ sign) / max(var_x, 1e-12))
    rotation = u @ sign @ vh
    translation = mu_y - scale * rotation @ mu_x
    return scale, rotation, translation


def trajectory_pose_metrics(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> dict[str, float]:
    pred = _pose_homogeneous(pred_c2w)
    gt = _pose_homogeneous(gt_c2w)
    count = min(len(pred), len(gt))
    keys = (
        "ate_rmse_m",
        "are_deg",
        "rpe_translation_rmse_m",
        "rpe_rotation_rmse_deg",
        "rra_30_percent",
        "rta_30_percent",
    )
    if count < 2:
        return {key: float("nan") for key in keys}

    pred = pred[:count].copy()
    gt = gt[:count]
    scale, rotation, translation = umeyama(pred[:, :3, 3].T, gt[:, :3, 3].T)
    pred[:, :3, 3] = scale * (pred[:, :3, 3] @ rotation.T) + translation.reshape(1, 3)
    pred[:, :3, :3] = rotation[None] @ pred[:, :3, :3]

    ate = np.linalg.norm(pred[:, :3, 3] - gt[:, :3, 3], axis=1)
    absolute_rotation = rotation_error_degrees(gt[:, :3, :3], pred[:, :3, :3])
    pred_w2c = np.linalg.inv(pred)
    gt_w2c = np.linalg.inv(gt)
    pair_rot, pair_trans, rpe_rot, rpe_trans = [], [], [], []
    for i in range(count - 1):
        for j in range(i + 1, count):
            pred_rel = pred_w2c[i] @ np.linalg.inv(pred_w2c[j])
            gt_rel = gt_w2c[i] @ np.linalg.inv(gt_w2c[j])
            rot_err = float(rotation_error_degrees(gt_rel[:3, :3][None], pred_rel[:3, :3][None])[0])
            pair_rot.append(rot_err)
            gt_t = gt_rel[:3, 3]
            pred_t = pred_rel[:3, 3]
            denom = np.linalg.norm(gt_t) * np.linalg.norm(pred_t)
            if denom <= 1e-15:
                trans_err = 180.0
            else:
                trans_err = float(np.degrees(np.arccos(np.clip(abs(np.dot(gt_t, pred_t)) / denom, 0, 1))))
            pair_trans.append(trans_err)
            if j == i + 1:
                rpe_rot.append(rot_err)
                rpe_trans.append(float(np.linalg.norm(gt_t - pred_t)))

    return {
        "ate_rmse_m": float(np.sqrt(np.mean(ate**2))),
        "are_deg": float(np.mean(absolute_rotation)),
        "rpe_translation_rmse_m": float(np.sqrt(np.mean(np.square(rpe_trans)))),
        "rpe_rotation_rmse_deg": float(np.sqrt(np.mean(np.square(rpe_rot)))),
        "rra_30_percent": float(100.0 * np.mean(np.asarray(pair_rot) < 30.0)),
        "rta_30_percent": float(100.0 * np.mean(np.asarray(pair_trans) < 30.0)),
    }


def depth_error_metrics(prediction: np.ndarray, ground_truth: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64)[valid]
    gt = np.asarray(ground_truth, dtype=np.float64)[valid]
    if not len(pred):
        raise ValueError("No valid depth pixels")
    diff = pred - gt
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "abs_rel": float(np.mean(np.abs(diff) / gt)),
        "sq_rel": float(np.mean(diff**2 / gt)),
        "rmse_m": float(np.sqrt(np.mean(diff**2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))),
        "mae_m": float(np.mean(np.abs(diff))),
        "delta_1_25_percent": float(100.0 * np.mean(ratio < 1.25)),
        "delta_1_25_sq_percent": float(100.0 * np.mean(ratio < 1.25**2)),
        "delta_1_25_cu_percent": float(100.0 * np.mean(ratio < 1.25**3)),
        "valid_depth_pixels": float(len(pred)),
    }


def depth_to_world_points(depth: np.ndarray, c2w: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project [F,H,W] depth maps into world point maps [F,H,W,3]."""
    frames, height, width = depth.shape
    ys, xs = np.mgrid[0:height, 0:width]
    points = np.empty((frames, height, width, 3), dtype=np.float32)
    for frame in range(frames):
        k = intrinsics[frame]
        z = depth[frame]
        cam = np.stack(
            ((xs - k[0, 2]) * z / k[0, 0], (ys - k[1, 2]) * z / k[1, 1], z),
            axis=-1,
        )
        points[frame] = cam @ c2w[frame, :3, :3].T + c2w[frame, :3, 3]
    return points


def scaled_intrinsics(
    intrinsics: np.ndarray,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
    frames: int,
) -> np.ndarray:
    """Scale a pixel-camera matrix after direct image/depth resizing."""
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    if intrinsics.ndim == 2:
        out = np.repeat(np.asarray(intrinsics, dtype=np.float64)[None], frames, axis=0)
    else:
        out = np.asarray(intrinsics, dtype=np.float64).copy()
    out[:, 0, :] *= target_w / source_w
    out[:, 1, :] *= target_h / source_h
    out[:, 2, 2] = 1.0
    return out


def evaluate_pi3_geometry(
    pred_point_map: np.ndarray,
    gt_point_map: np.ndarray,
    valid_mask: np.ndarray,
    *,
    max_points: int = 100_000,
    icp_threshold_m: float = 0.1,
) -> dict[str, float | int]:
    """Compute Pi3 ACC/Comp/NC after Sim(3)+ICP alignment."""
    pred_map = np.asarray(pred_point_map)
    gt_map = np.asarray(gt_point_map)
    valid_ids = np.flatnonzero(np.asarray(valid_mask).reshape(-1))
    if len(valid_ids) < 3:
        raise ValueError("geometry evaluation needs at least three valid points")
    sample_ids = valid_ids[np.linspace(0, len(valid_ids) - 1, min(max_points, len(valid_ids)), dtype=np.int64)]
    pred = pred_map.reshape(-1, 3)[sample_ids].astype(np.float64, copy=False)
    gt = gt_map.reshape(-1, 3)[sample_ids].astype(np.float64, copy=False)
    finite = np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
    pred = pred[finite]
    gt = gt[finite]
    if len(pred) < 3:
        raise ValueError("geometry evaluation needs at least three finite corresponding points")

    scale, rotation, translation = umeyama(pred.T, gt.T)
    pred = scale * (pred @ rotation.T) + translation.reshape(1, 3)
    pred_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred))
    gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt))
    icp = o3d.pipelines.registration.registration_icp(
        pred_pcd,
        gt_pcd,
        icp_threshold_m,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pred_pcd.transform(icp.transformation)
    pred_pcd.estimate_normals()
    gt_pcd.estimate_normals()
    pred = np.asarray(pred_pcd.points)
    gt = np.asarray(gt_pcd.points)
    pred_normals = np.asarray(pred_pcd.normals)
    gt_normals = np.asarray(gt_pcd.normals)

    gt_tree = cKDTree(gt)
    acc_dist, acc_ids = gt_tree.query(pred, workers=-1)
    pred_tree = cKDTree(pred)
    comp_dist, comp_ids = pred_tree.query(gt, workers=-1)
    nc1 = np.abs(np.sum(pred_normals * gt_normals[acc_ids], axis=-1))
    nc2 = np.abs(np.sum(gt_normals * pred_normals[comp_ids], axis=-1))
    acc_mean = float(np.mean(acc_dist))
    acc_median = float(np.median(acc_dist))
    comp_mean = float(np.mean(comp_dist))
    comp_median = float(np.median(comp_dist))
    precision = float(np.mean(acc_dist < icp_threshold_m))
    recall = float(np.mean(comp_dist < icp_threshold_m))
    f1 = 0.0 if precision + recall <= 1.0e-12 else 2.0 * precision * recall / (precision + recall)
    overall_mean = float((acc_mean + comp_mean) / 2.0)
    overall_median = float((acc_median + comp_median) / 2.0)
    return {
        "acc_mean_m": acc_mean,
        "acc_median_m": acc_median,
        "comp_mean_m": comp_mean,
        "comp_median_m": comp_median,
        "precision_at_threshold_percent": 100.0 * precision,
        "recall_at_threshold_percent": 100.0 * recall,
        "f1_at_threshold_percent": 100.0 * f1,
        "f1_10cm_percent": 100.0 * f1,
        "f1_threshold_m": float(icp_threshold_m),
        "nc_mean": float((np.mean(nc1) + np.mean(nc2)) / 2.0),
        "nc_median": float((np.median(nc1) + np.median(nc2)) / 2.0),
        "nc1_mean": float(np.mean(nc1)),
        "nc1_median": float(np.median(nc1)),
        "nc2_mean": float(np.mean(nc2)),
        "nc2_median": float(np.median(nc2)),
        "overall_mean_m": overall_mean,
        "overall_median_m": overall_median,
        "chamfer_mean_m": overall_mean,
        "chamfer_median_m": overall_median,
        "geometry_points": int(len(pred)),
        "sim3_scale": float(scale),
    }
