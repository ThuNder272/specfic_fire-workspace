"""
PnP solver utilities for pose estimation.

This module wraps OpenCV solvePnP/solvePnPRansac with a friendly interface.
Based on the implementation from AutoAiming project.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping, Optional

import cv2
import numpy as np


@dataclass
class PoseResult:
    """Result container for pose estimation."""
    success: bool
    rvec: Optional[np.ndarray] = None  # 旋转向量
    tvec: Optional[np.ndarray] = None  # 平移向量
    reprojection_error: Optional[float] = None  # 重投影误差
    inliers: Optional[np.ndarray] = None  # 内点索引


@dataclass
class CameraIntrinsics:
    """Wrapper around a camera matrix and distortion coefficients."""
    name: str
    matrix: np.ndarray  # 3x3 相机内参矩阵
    dist_coeffs: np.ndarray  # (k1, k2, p1, p2, k3) 畸变系数
    image_size: Optional[tuple[int, int]] = None  # (width, height)


class TargetGeometry:
    """目标几何类型枚举"""
    ARMOR_SMALL = "armor_small"
    ARMOR_BIG = "armor_big"


@dataclass
class ArmorTemplate:
    """装甲板3D模板容器"""
    target_type: str
    points: np.ndarray  # Nx3 3D点数组
    description: str = ""


# 默认目标模板（基于RoboMaster官方规格，单位：米）
_DEFAULT_TARGETS = {
    TargetGeometry.ARMOR_SMALL: ArmorTemplate(
        TargetGeometry.ARMOR_SMALL,
        points=np.array([
            [-0.0675, -0.0625, 0.0],  # 135mm x 125mm -> 半尺寸
            [0.0675, -0.0625, 0.0],
            [0.0675, 0.0625, 0.0],
            [-0.0675, 0.0625, 0.0],
        ], dtype=np.float32),
        description="小装甲板 (135mm x 125mm)",
    ),
    TargetGeometry.ARMOR_BIG: ArmorTemplate(
        TargetGeometry.ARMOR_BIG,
        points=np.array([
            [-0.1150, -0.0635, 0.0],  # 230mm x 127mm -> 半尺寸
            [0.1150, -0.0635, 0.0],
            [0.1150, 0.0635, 0.0],
            [-0.1150, 0.0635, 0.0],
        ], dtype=np.float32),
        description="大装甲板 (230mm x 127mm)",
    ),
}

# RM4PT 36类顺序，来源于 RM_4-points_yolov5-master/data/widerface.yaml
_RM4PT_CLASS_NAMES = (
    "BG",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "BO",
    "BBs",
    "BBb",
    "RG",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "RO",
    "RBs",
    "RBb",
    "NG",
    "N1",
    "N2",
    "N3",
    "N4",
    "N5",
    "NO",
    "NBs",
    "NBb",
    "PG",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "PO",
    "PBs",
    "PBb",
)
_RM4PT_KNOWN_CLASS_NAMES = frozenset(_RM4PT_CLASS_NAMES)
_RM4PT_BIG_ARMOR_CLASS_IDS = frozenset({1, 10})
_RM4PT_BIG_ARMOR_CLASS_NAMES = frozenset({"B1", "R1"})

# 默认相机内参（适配MER-131相机6mm镜头）
_DEFAULT_CAMERAS = {
    "default": CameraIntrinsics(
        name="default",
        matrix=np.array([
            [900.0, 0.0, 640.0],
            [0.0, 900.0, 360.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32),
        dist_coeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        image_size=(1280, 720),
    ),
    "mer_131_6mm": CameraIntrinsics(
        name="mer_131_6mm",
        matrix=np.array([
            # 1280x1024 标定参数在中心ROI(OffsetX=320, OffsetY=192)裁剪到 640x640 后：
            # fx/fy 保持不变，cx'=cx-320，cy'=cy-192
            # 经验补偿：焦距轻微放大 1.12x（距离偏近时可适当拉远）
            [1268.96780, 0.0, 339.1441010],
            [0.0, 1273.14367, 326.158114],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32),
        dist_coeffs=np.zeros(5, dtype=np.float32),
        image_size=(640, 640),
    ),
    "mer_139_210u3c": CameraIntrinsics(  # 适配当前项目的大恒相机
        name="mer_139_210u3c",
        matrix=np.array([
            [1268.96780, 0.0, 339.144101],  # 最新标定结果：fx, cx
            [0.0, 1273.14367, 326.158114],  # fy, cy
            [0.0, 0.0, 1.0],
        ], dtype=np.float32),
        dist_coeffs=np.array([
            -0.08101801,   # k1
            0.14113087,    # k2
            0.00065719,    # p1
            0.00381671,    # p2
            -0.02366227,   # k3
        ], dtype=np.float32),
        image_size=(1280, 1024),
    ),
}


def _compute_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    intrinsics: CameraIntrinsics,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    """计算重投影误差"""
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, intrinsics.matrix, intrinsics.dist_coeffs)
    projected = projected.squeeze()
    return float(np.mean(np.linalg.norm(projected - image_points, axis=1)))


def solve_pose(
    image_points: np.ndarray,
    target_type: str = TargetGeometry.ARMOR_SMALL,
    intrinsics: Optional[CameraIntrinsics] = None,
    use_ippe: bool = True,
    use_ransac: bool = True,
    reprojection_threshold: float = 3.0,
    max_iters: int = 100,
) -> PoseResult:
    """
    使用PnP算法求解目标姿态
    
    Args:
        image_points: 图像中的2D点坐标，形状为(N, 2)
        target_type: 目标类型，默认为标准装甲板
        intrinsics: 相机内参，如为None则使用默认参数
        use_ippe: 是否优先使用IPPE求解平面目标
        use_ransac: 是否使用RANSAC算法
        reprojection_threshold: 重投影误差阈值
        max_iters: 最大迭代次数
        
    Returns:
        PoseResult: 姿态估计结果
    """
    if intrinsics is None:
        intrinsics = _DEFAULT_CAMERAS["mer_139_210u3c"]  # 使用当前项目相机配置
    
    # 获取目标3D模板
    if target_type not in _DEFAULT_TARGETS:
        raise ValueError(f"不支持的目标类型: {target_type}")
    
    template = _DEFAULT_TARGETS[target_type]
    object_points = template.points
    
    # 检查点数量是否匹配
    if image_points.shape[0] != object_points.shape[0]:
        raise ValueError(f"图像点数量({image_points.shape[0]})与物体点数量({object_points.shape[0]})不匹配")
    
    # 确保输入数据类型正确
    image_points = image_points.astype(np.float32)
    object_points = object_points.astype(np.float32)
    
    try:
        if use_ippe and np.allclose(object_points[:, 2], 0.0):
            try:
                success, rvecs, tvecs = cv2.solvePnPGeneric(
                    object_points,
                    image_points,
                    intrinsics.matrix,
                    intrinsics.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE,
                )
                if success and rvecs and tvecs:
                    errors = [
                        _compute_reprojection_error(object_points, image_points, intrinsics, rvec, tvec)
                        for rvec, tvec in zip(rvecs, tvecs)
                    ]
                    if not errors:
                        raise RuntimeError("IPPE result empty")
                    best_idx = int(np.argmin(errors))
                    rvec = rvecs[best_idx]
                    tvec = tvecs[best_idx]
                    success_refine, rvec, tvec = cv2.solvePnP(
                        object_points,
                        image_points,
                        intrinsics.matrix,
                        intrinsics.dist_coeffs,
                        rvec,
                        tvec,
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                    if success_refine:
                        error = _compute_reprojection_error(object_points, image_points, intrinsics, rvec, tvec)
                        return PoseResult(
                            success=True,
                            rvec=rvec,
                            tvec=tvec,
                            reprojection_error=error,
                            inliers=None,
                        )
            except Exception:
                pass

        if use_ransac:
            # 使用RANSAC算法提高鲁棒性
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                intrinsics.matrix,
                intrinsics.dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP,
                reprojectionError=reprojection_threshold,
                iterationsCount=max_iters,
            )
        else:
            # 标准PnP求解
            success, rvec, tvec = cv2.solvePnP(
                object_points, 
                image_points, 
                intrinsics.matrix, 
                intrinsics.dist_coeffs, 
                flags=cv2.SOLVEPNP_EPNP
            )
            inliers = None
        
        if not success:
            return PoseResult(success=False)
        
        # 使用迭代法精化结果
        success_refine, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            intrinsics.matrix,
            intrinsics.dist_coeffs,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        
        if not success_refine:
            return PoseResult(success=False)
        
        # 计算重投影误差
        error = _compute_reprojection_error(object_points, image_points, intrinsics, rvec, tvec)
        
        return PoseResult(
            success=True, 
            rvec=rvec, 
            tvec=tvec, 
            reprojection_error=error, 
            inliers=inliers
        )
        
    except Exception as e:
        print(f"PnP求解失败: {e}")
        return PoseResult(success=False)


def get_camera_intrinsics(profile: str = "mer_139_210u3c") -> CameraIntrinsics:
    """获取相机内参"""
    if profile not in _DEFAULT_CAMERAS:
        raise KeyError(f"相机配置'{profile}'不存在")
    return _DEFAULT_CAMERAS[profile]


def get_target_template(target_type: str) -> ArmorTemplate:
    """获取目标3D模板"""
    if target_type not in _DEFAULT_TARGETS:
        raise KeyError(f"目标类型'{target_type}'不存在")
    return _DEFAULT_TARGETS[target_type]


def get_target_aspect_ratio(target_type: str) -> float:
    """获取目标宽高比（宽/高）。"""
    template = get_target_template(target_type)
    points = template.points
    width = float(np.max(points[:, 0]) - np.min(points[:, 0]))
    height = float(np.max(points[:, 1]) - np.min(points[:, 1]))
    if height <= 0.0:
        return 1.0
    return width / height


def _normalize_rm4pt_class_name(class_name) -> Optional[str]:
    if class_name is None:
        return None
    normalized = "".join(ch for ch in str(class_name).strip().upper() if ch.isalnum())
    return normalized or None


def get_rm4pt_class_name(class_id: int) -> Optional[str]:
    """根据标准RM4PT类别ID获取类别名。"""
    try:
        index = int(class_id)
    except Exception:
        return None
    if 0 <= index < len(_RM4PT_CLASS_NAMES):
        return _RM4PT_CLASS_NAMES[index]
    return None


def choose_target_type_by_class(
    class_id: Optional[int] = None,
    class_name: Optional[str] = None,
    default_type: str = TargetGeometry.ARMOR_SMALL,
) -> str:
    """根据RM4PT类别ID/名称选择装甲板类型。

    当前项目规则：
    - class 1(B1) 和 10(R1) 为大装甲板
    - 其余已知RM4PT类别统一按小装甲板处理
    - 无法识别类别时回退到 default_type
    """
    resolved_class_id = None
    if class_id is not None:
        try:
            resolved_class_id = int(class_id)
        except Exception:
            resolved_class_id = None

    normalized_name = _normalize_rm4pt_class_name(class_name)
    if normalized_name is None and resolved_class_id is not None:
        normalized_name = get_rm4pt_class_name(resolved_class_id)

    if resolved_class_id in _RM4PT_BIG_ARMOR_CLASS_IDS:
        return TargetGeometry.ARMOR_BIG
    if normalized_name in _RM4PT_BIG_ARMOR_CLASS_NAMES:
        return TargetGeometry.ARMOR_BIG
    if resolved_class_id is not None:
        return TargetGeometry.ARMOR_SMALL
    if normalized_name in _RM4PT_KNOWN_CLASS_NAMES:
        return TargetGeometry.ARMOR_SMALL
    return default_type


def choose_target_type_by_detection(
    detection: Optional[Mapping[str, object]],
    default_type: str = TargetGeometry.ARMOR_SMALL,
) -> str:
    """根据检测结果中的类别信息选择装甲板类型。"""
    if not isinstance(detection, Mapping):
        return default_type
    return choose_target_type_by_class(
        class_id=detection.get("class"),
        class_name=detection.get("class_name"),
        default_type=default_type,
    )


def choose_target_type_by_bbox(
    bbox: tuple[int, int, int, int],
    default_type: str = TargetGeometry.ARMOR_SMALL,
) -> str:
    """根据检测框宽高比选择目标类型。"""
    x1, y1, x2, y2 = bbox
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    if h <= 0.0:
        return default_type
    ratio = w / h
    best_type = default_type
    best_diff = float("inf")
    for target_type in _DEFAULT_TARGETS.keys():
        expected = get_target_aspect_ratio(target_type)
        diff = abs(ratio - expected)
        if diff < best_diff:
            best_diff = diff
            best_type = target_type
    return best_type


def convert_rvec_to_euler(rvec: np.ndarray) -> tuple:
    """将旋转向量转换为欧拉角（弧度）"""
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0
    
    return x, y, z


def compute_target_distance(tvec: np.ndarray) -> float:
    """计算目标距离（相机到目标的直线距离）"""
    return float(np.linalg.norm(tvec))


def scale_intrinsics_to_frame(
    intrinsics: CameraIntrinsics, frame_shape: tuple[int, int, int]
) -> CameraIntrinsics:
    """根据帧分辨率缩放相机内参矩阵。"""
    if intrinsics.image_size is None:
        return intrinsics
    frame_h, frame_w = frame_shape[:2]
    base_w, base_h = intrinsics.image_size
    if base_w <= 0 or base_h <= 0:
        return intrinsics
    sx = float(frame_w) / float(base_w)
    sy = float(frame_h) / float(base_h)
    if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
        return intrinsics
    matrix = intrinsics.matrix.copy()
    matrix[0, 0] *= sx
    matrix[0, 2] *= sx
    matrix[1, 1] *= sy
    matrix[1, 2] *= sy
    return CameraIntrinsics(
        name=f"{intrinsics.name}_{frame_w}x{frame_h}",
        matrix=matrix,
        dist_coeffs=intrinsics.dist_coeffs.copy(),
        image_size=(frame_w, frame_h),
    )


def compute_yaw_pitch(rvec: np.ndarray, tvec: np.ndarray, gun_offset_y: float) -> tuple:
    """计算目标的偏航角和俯仰角（用于瞄准）

    Args:
        rvec: 旋转向量
        tvec: 平移向量
        gun_offset_y: 枪口相对于相机光心的y轴偏移（米，正下方为正值）
    """
    
    target_pos = tvec.flatten()
    dx = float(target_pos[0])
    dy = float(target_pos[1]) - float(gun_offset_y)
    dz = float(target_pos[2])
    horiz = float(np.hypot(dx, dz))
    if horiz < 1e-9:
        horiz = 1e-9

    yaw = np.arctan2(dx, dz)
    pitch = np.arctan2(-dy, horiz)
    return float(yaw), float(pitch)

def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _detect_armor_corners(
    frame: np.ndarray, bbox: tuple[int, int, int, int]
) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    roi_area = float((x2 - x1) * (y2 - y1))
    if roi_area <= 1.0:
        return None
    if cv2.contourArea(contour) < roi_area * 0.1:
        return None

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(contour)
        pts = cv2.boxPoints(rect).astype(np.float32)

    pts = _order_quad_points(pts)
    pts[:, 0] += x1
    pts[:, 1] += y1
    return pts


def detect_armor_corners(
    frame: np.ndarray, bbox: tuple[int, int, int, int]
) -> Optional[np.ndarray]:
    """公开版角点检测接口（用于可视化）。"""
    return _detect_armor_corners(frame, bbox)


def _normalize_bbox_aspect(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int],
    target_type: str,
) -> tuple[int, int, int, int]:
    try:
        template = get_target_template(target_type)
    except Exception:
        return bbox
    points = template.points
    width = float(np.max(points[:, 0]) - np.min(points[:, 0]))
    height = float(np.max(points[:, 1]) - np.min(points[:, 1]))
    if width <= 0.0 or height <= 0.0:
        return bbox
    expected_ratio = width / height

    x1, y1, x2, y2 = bbox
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    current_ratio = w / h

    if current_ratio > expected_ratio:
        new_w = h * expected_ratio
        new_h = h
    else:
        new_w = w
        new_h = w / expected_ratio

    frame_h, frame_w = frame_shape[:2]
    new_w = min(new_w, max(1.0, float(frame_w - 1)))
    new_h = min(new_h, max(1.0, float(frame_h - 1)))

    nx1 = int(max(0, min(frame_w - 1, cx - new_w / 2.0)))
    ny1 = int(max(0, min(frame_h - 1, cy - new_h / 2.0)))
    nx2 = int(max(0, min(frame_w - 1, cx + new_w / 2.0)))
    ny2 = int(max(0, min(frame_h - 1, cy + new_h / 2.0)))
    if nx2 <= nx1:
        nx2 = min(frame_w - 1, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(frame_h - 1, ny1 + 1)
    return nx1, ny1, nx2, ny2


def _shrink_bbox(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int],
    shrink_ratio: float,
) -> tuple[int, int, int, int]:
    if shrink_ratio is None:
        return bbox
    ratio = float(shrink_ratio)
    if ratio >= 0.999:
        return bbox
    ratio = max(0.1, min(1.0, ratio))
    x1, y1, x2, y2 = bbox
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    new_w = max(1.0, w * ratio)
    new_h = max(1.0, h * ratio)
    frame_h, frame_w = frame_shape[:2]
    nx1 = int(max(0, min(frame_w - 1, cx - new_w / 2.0)))
    ny1 = int(max(0, min(frame_h - 1, cy - new_h / 2.0)))
    nx2 = int(max(0, min(frame_w - 1, cx + new_w / 2.0)))
    ny2 = int(max(0, min(frame_h - 1, cy + new_h / 2.0)))
    if nx2 <= nx1:
        nx2 = min(frame_w - 1, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(frame_h - 1, ny1 + 1)
    return nx1, ny1, nx2, ny2


def _solve_pose_from_points(
    image_points: np.ndarray,
    target_type: str,
    intrinsics: CameraIntrinsics,
    max_pnp_error: float,
    gun_offset_y: float,
    return_comp: bool = False,
    return_distance: bool = False,
    return_tvec: bool = False,
    return_rvec: bool = False,
    log_comp: bool = False,
    log_label: str = "",
) -> tuple[Optional[float], ...]:
    def _pack_result(
        yaw_deg: Optional[float],
        pitch_deg: Optional[float],
        reprojection_error: Optional[float],
        elapsed_ms: Optional[float],
        comp_yaw: Optional[float],
        comp_pitch: Optional[float],
        distance_m: Optional[float] = None,
        tvec: Optional[np.ndarray] = None,
        rvec: Optional[np.ndarray] = None,
    ) -> tuple[Optional[float], ...]:
        if return_distance and return_tvec and return_rvec:
            return (
                yaw_deg,
                pitch_deg,
                reprojection_error,
                elapsed_ms,
                comp_yaw,
                comp_pitch,
                distance_m,
                tvec,
                rvec,
            )
        if return_distance and return_tvec:
            return (
                yaw_deg,
                pitch_deg,
                reprojection_error,
                elapsed_ms,
                comp_yaw,
                comp_pitch,
                distance_m,
                tvec,
            )
        if return_distance:
            return (
                yaw_deg,
                pitch_deg,
                reprojection_error,
                elapsed_ms,
                comp_yaw,
                comp_pitch,
                distance_m,
            )
        if return_tvec and return_rvec:
            return (
                yaw_deg,
                pitch_deg,
                reprojection_error,
                elapsed_ms,
                comp_yaw,
                comp_pitch,
                tvec,
                rvec,
            )
        if return_tvec:
            return (
                yaw_deg,
                pitch_deg,
                reprojection_error,
                elapsed_ms,
                comp_yaw,
                comp_pitch,
                tvec,
            )
        return (
            yaw_deg,
            pitch_deg,
            reprojection_error,
            elapsed_ms,
            comp_yaw,
            comp_pitch,
        )

    try:
        start = time.time()
        pose = solve_pose(
            image_points,
            target_type=target_type,
            intrinsics=intrinsics,
        )
        elapsed_ms = (time.time() - start) * 1000.0
    except Exception as exc:
        print(f"⚠️ PnP解算异常: {exc}")
        return _pack_result(None, None, None, None, None, None)

    if not pose.success or pose.reprojection_error is None:
        return _pack_result(None, None, pose.reprojection_error, elapsed_ms, None, None)
    if pose.reprojection_error > max_pnp_error:
        return _pack_result(None, None, pose.reprojection_error, elapsed_ms, None, None)

    yaw, pitch = compute_yaw_pitch(pose.rvec, pose.tvec, gun_offset_y)
    comp_yaw = None
    comp_pitch = None
    distance_m = compute_target_distance(pose.tvec) if return_distance else None
    tvec = pose.tvec.copy() if return_tvec else None
    rvec = pose.rvec.copy() if (return_tvec and return_rvec) else None
    if return_comp or log_comp:
        raw_yaw, raw_pitch = compute_yaw_pitch(pose.rvec, pose.tvec, 0.0)
        comp_yaw = float(np.degrees(yaw - raw_yaw))
        comp_pitch = float(np.degrees(pitch - raw_pitch))
        if log_comp:
            tvec_flat = pose.tvec.flatten()
            horiz = float(np.hypot(tvec_flat[0], tvec_flat[2]))
            label = f"{log_label}" if log_label else "PnP"
            print(
                f"{label}补偿: dyaw={comp_yaw:.2f} dpitch={comp_pitch:.2f} "
                f"h={horiz:.3f}m offset_y={gun_offset_y:.4f}m"
            )
    return _pack_result(
        float(np.degrees(yaw)),
        float(np.degrees(pitch)),
        pose.reprojection_error,
        elapsed_ms,
        comp_yaw,
        comp_pitch,
        distance_m,
        tvec,
        rvec,
    )



def solve_angles_from_bbox(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    target_type: str,
    intrinsics: Optional[CameraIntrinsics],
    max_pnp_error: float,
    gun_offset_y: float,
    use_corners: bool = True,
    corners: Optional[np.ndarray] = None,
    bbox_shrink: float = 1.0,
    return_comp: bool = False,
    return_distance: bool = False,
    log_comp: bool = False,
    log_label: str = "",
    return_tvec: bool = False,
    return_rvec: bool = False,
) -> tuple[Optional[float], ...]:
    if intrinsics is None:
        intrinsics = get_camera_intrinsics("default")

    last_err = None
    last_ms = None
    last_comp_yaw = None
    last_comp_pitch = None
    last_distance = None
    last_tvec = None
    last_rvec = None

    def _pack_result(
        yaw: Optional[float],
        pitch: Optional[float],
        err: Optional[float],
        ms: Optional[float],
        comp_yaw: Optional[float],
        comp_pitch: Optional[float],
        distance_m: Optional[float],
        tvec: Optional[np.ndarray],
        rvec: Optional[np.ndarray],
    ):
        if yaw is None or pitch is None:
            tvec = None
            rvec = None
        if return_distance and return_tvec and return_rvec:
            return yaw, pitch, err, ms, comp_yaw, comp_pitch, distance_m, tvec, rvec
        if return_distance and return_tvec:
            return yaw, pitch, err, ms, comp_yaw, comp_pitch, distance_m, tvec
        if return_distance:
            return yaw, pitch, err, ms, comp_yaw, comp_pitch, distance_m
        if return_tvec and return_rvec:
            return yaw, pitch, err, ms, comp_yaw, comp_pitch, tvec, rvec
        if return_tvec:
            return yaw, pitch, err, ms, comp_yaw, comp_pitch, tvec
        return yaw, pitch, err, ms, comp_yaw, comp_pitch

    def _solve_points(points: np.ndarray):
        if return_distance:
            if return_tvec:
                result = _solve_pose_from_points(
                    points,
                    target_type,
                    intrinsics,
                    max_pnp_error,
                    gun_offset_y,
                    return_comp=return_comp,
                    return_distance=True,
                    return_tvec=True,
                    return_rvec=return_rvec,
                    log_comp=log_comp,
                    log_label=log_label,
                )
                if return_rvec:
                    yaw, pitch, err, ms, comp_yaw, comp_pitch, distance_m, tvec, rvec = result
                else:
                    yaw, pitch, err, ms, comp_yaw, comp_pitch, distance_m, tvec = result
                    rvec = None
            else:
                yaw, pitch, err, ms, comp_yaw, comp_pitch, distance_m = _solve_pose_from_points(
                    points,
                    target_type,
                    intrinsics,
                    max_pnp_error,
                    gun_offset_y,
                    return_comp=return_comp,
                    return_distance=True,
                    log_comp=log_comp,
                    log_label=log_label,
                )
                tvec = None
                rvec = None
        else:
            if return_tvec:
                result = _solve_pose_from_points(
                    points,
                    target_type,
                    intrinsics,
                    max_pnp_error,
                    gun_offset_y,
                    return_comp=return_comp,
                    return_tvec=True,
                    return_rvec=return_rvec,
                    log_comp=log_comp,
                    log_label=log_label,
                )
                if return_rvec:
                    yaw, pitch, err, ms, comp_yaw, comp_pitch, tvec, rvec = result
                else:
                    yaw, pitch, err, ms, comp_yaw, comp_pitch, tvec = result
                    rvec = None
            else:
                yaw, pitch, err, ms, comp_yaw, comp_pitch = _solve_pose_from_points(
                    points,
                    target_type,
                    intrinsics,
                    max_pnp_error,
                    gun_offset_y,
                    return_comp=return_comp,
                    log_comp=log_comp,
                    log_label=log_label,
                )
                tvec = None
                rvec = None
            distance_m = None
        return yaw, pitch, err, ms, comp_yaw, comp_pitch, distance_m, tvec, rvec

    def _try_points(points: Optional[np.ndarray]):
        nonlocal last_err, last_ms, last_comp_yaw, last_comp_pitch, last_distance, last_tvec, last_rvec
        if points is None:
            return None
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if pts.shape != (4, 2):
            return None
        pts = _order_quad_points(pts)
        yaw, pitch, err, ms, comp_yaw, comp_pitch, distance_m, tvec, rvec = _solve_points(pts)
        last_err = err
        last_ms = ms
        last_comp_yaw = comp_yaw
        last_comp_pitch = comp_pitch
        if distance_m is not None:
            last_distance = distance_m
        if tvec is not None:
            last_tvec = tvec
        if rvec is not None:
            last_rvec = rvec
        if yaw is None or pitch is None:
            return None
        return _pack_result(yaw, pitch, err, ms, comp_yaw, comp_pitch, distance_m, tvec, rvec)

    result = _try_points(corners)
    if result is not None:
        return result

    if use_corners:
        result = _try_points(_detect_armor_corners(frame, bbox))
        if result is not None:
            return result

    base_bbox = _shrink_bbox(bbox, frame.shape, bbox_shrink)
    x1, y1, x2, y2 = base_bbox
    bbox_points = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )
    yaw, pitch, err, ms, comp_yaw, comp_pitch, distance_m, tvec, rvec = _solve_points(bbox_points)
    last_err = err
    last_ms = ms
    last_comp_yaw = comp_yaw
    last_comp_pitch = comp_pitch
    if distance_m is not None:
        last_distance = distance_m
    if tvec is not None:
        last_tvec = tvec
    if rvec is not None:
        last_rvec = rvec
    if yaw is not None and pitch is not None:
        return _pack_result(yaw, pitch, err, ms, comp_yaw, comp_pitch, distance_m, tvec, rvec)

    normalized_bbox = _normalize_bbox_aspect(base_bbox, frame.shape, target_type)
    if normalized_bbox == base_bbox:
        return _pack_result(
            yaw,
            pitch,
            last_err,
            last_ms,
            last_comp_yaw,
            last_comp_pitch,
            last_distance,
            last_tvec,
            last_rvec,
        )

    nx1, ny1, nx2, ny2 = normalized_bbox
    norm_points = np.array(
        [[nx1, ny1], [nx2, ny1], [nx2, ny2], [nx1, ny2]],
        dtype=np.float32,
    )
    norm_yaw, norm_pitch, norm_err, norm_ms, comp_yaw, comp_pitch, distance_m, tvec, rvec = _solve_points(norm_points)
    if norm_yaw is None or norm_pitch is None:
        return _pack_result(
            yaw,
            pitch,
            norm_err if norm_err is not None else last_err,
            norm_ms,
            comp_yaw if comp_yaw is not None else last_comp_yaw,
            comp_pitch if comp_pitch is not None else last_comp_pitch,
            distance_m if distance_m is not None else last_distance,
            tvec if tvec is not None else last_tvec,
            rvec if rvec is not None else last_rvec,
        )
    return _pack_result(
        norm_yaw,
        norm_pitch,
        norm_err,
        norm_ms,
        comp_yaw,
        comp_pitch,
        distance_m,
        tvec,
        rvec,
    )



__all__ = [
    "PoseResult",
    "CameraIntrinsics", 
    "TargetGeometry",
    "ArmorTemplate",
    "solve_pose",
    "get_camera_intrinsics",
    "get_target_template",
    "get_target_aspect_ratio",
    "get_rm4pt_class_name",
    "choose_target_type_by_class",
    "choose_target_type_by_detection",
    "choose_target_type_by_bbox",
    "convert_rvec_to_euler",
    "compute_target_distance",
    "scale_intrinsics_to_frame",
    "compute_yaw_pitch",
    "detect_armor_corners",
    "solve_angles_from_bbox",
]
