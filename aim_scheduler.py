#!/usr/bin/env python
# coding=utf-8

"""
Aim scheduler entrypoint: parse args and run the aim pipeline.
"""

import argparse

from camera_adaptation.aim_pipeline import (
    AimPipeline,
    DEFAULT_COORD_MODEL,
    DEFAULT_DAHENG_CONFIG,
    DEFAULT_YOLO_MODEL,
    SERIAL_BAUD_DEFAULT,
    SERIAL_PORT_DEFAULT,
    TargetGeometry,
)


def _parse_csv_ints(text):
    if text is None:
        return None
    values = []
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item))
    return tuple(values) if values else None


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Aim scheduler: detect armor, compute yaw/pitch with PnP, send UART frames.",
    )
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL, help="YOLO模型路径")
    parser.add_argument("--coord-model", default=DEFAULT_COORD_MODEL, help="坐标预测模型路径")
    parser.add_argument("--port", default=SERIAL_PORT_DEFAULT, help="串口设备路径")
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD_DEFAULT, help="波特率")
    parser.add_argument("--rate", type=float, default=210.0, help="发送频率 (Hz)")
    parser.add_argument("--confidence", type=float, default=0.1, help="检测置信度阈值")
    parser.add_argument(
        "--fire-confidence-threshold",
        type=float,
        default=0.84,
        help="开火置信度阈值(>=该值优先开火)",
    )
    parser.add_argument(
        "--fire-force-interval-ms",
        type=float,
        default=1000.0,
        help="开火保底间隔(ms): 低置信度时，超过该时间且有预测结果则强制开火一次",
    )
    parser.add_argument(
        "--yolo-imgsz",
        default=None,
        help="YOLO输入尺寸，格式: 512,640 或 640",
    )
    parser.add_argument(
        "--yolo-max-det",
        type=int,
        default=1,
        help="YOLO最大输出框数(最终只取最高置信度)",
    )
    parser.add_argument(
        "--yolo-log-speed",
        action="store_true",
        default=False,
        help="打印YOLO分段耗时(会降低FPS)",
    )
    parser.add_argument(
        "--yolo-verbose",
        action="store_true",
        default=False,
        help="启用YOLO verbose输出",
    )
    parser.add_argument(
        "--detector-backend",
        default="rm4pt",
        choices=["auto", "ultralytics", "rm4pt"],
        help="检测后端选择: auto/ultralytics/rm4pt",
    )
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV摄像头编号")
    parser.add_argument("--use-opencv", action="store_true", help="使用OpenCV摄像头")
    parser.add_argument("--use-daheng", action="store_true", help="强制使用大恒摄像头（默认）")
    parser.add_argument("--daheng-config", default=DEFAULT_DAHENG_CONFIG, help="大恒配置文件路径(可选)")
    parser.add_argument("--daheng-sn", default=None, help="大恒摄像头序列号")
    parser.add_argument("--swap-rb", action="store_true", help="交换颜色通道")
    parser.add_argument("--pnp-profile", default="mer_139_210u3c", help="相机内参配置名")
    parser.add_argument(
        "--target-type",
        default=TargetGeometry.ARMOR_SMALL,
        help="装甲板类型: armor_small/armor_big",
    )
    parser.add_argument(
        "--target-color",
        choices=["red", "blue"],
        default=None,
        help="仅接受 rm4pt 输出中的指定颜色目标，避免红蓝混打",
    )
    parser.add_argument(
        "--target-class-ids",
        default=None,
        help="按 rm4pt class id 白名单过滤，逗号分隔，例如 2,11；优先于 target-color",
    )
    parser.add_argument(
        "--exclude-class-ids",
        default=None,
        help="按 rm4pt class id 黑名单过滤，逗号分隔，例如 2,11；会在保留结果里排除这些类别",
    )
    parser.add_argument("--max-pnp-error", type=float, default=5.0, help="PnP误差阈值")
    parser.add_argument("--angle-scale", type=float, default=100.0, help="角度缩放系数")
    parser.add_argument("--gun-offset-y", type=float, required=True, help="枪口相对于相机光心的y轴偏移(毫米)")
    parser.add_argument("--bullet-speed", type=float, default=27.0, help="弹速(m/s)")
    parser.add_argument("--system-latency-ms", type=float, default=125.0, help="系统时延(ms)")
    parser.add_argument(
        "--ec-feedback",
        action="store_true",
        default=True,
        help="启用电控回传(yaw/pitch)用于角域预测与 yaw_v*t0 补偿（默认开启）",
    )
    parser.add_argument(
        "--no-ec-feedback",
        action="store_false",
        dest="ec_feedback",
        help="禁用电控回传角域补偿",
    )
    parser.add_argument(
        "--latency-viz",
        action="store_true",
        default=True,
        help="显示中间过程延迟可视化(蓝=视觉目标值 绿=电机真实值)曲线(默认开启)",
    )
    parser.add_argument(
        "--no-latency-viz",
        action="store_false",
        dest="latency_viz",
        help="关闭中间过程延迟可视化曲线",
    )
    parser.add_argument(
        "--latency-viz-window-s",
        type=float,
        default=5.0,
        help="延迟可视化曲线的滚动时间窗口(秒)",
    )
    parser.add_argument(
        "--latency-viz-csv",
        default=None,
        help="将目标值/真实值时序写入CSV(供VOFA+/离线绘图)，例如 latency_viz.csv",
    )
    parser.add_argument("--ec-invert-yaw", action="store_true", default=False, help="反转电控回传yaw方向")
    parser.add_argument("--ec-invert-pitch", action="store_true", default=False, help="反转电控回传pitch方向")
    parser.add_argument("--ec-t0-ms", type=float, default=20.0, help="yaw_v*t0 补偿时间常数(ms)")
    parser.add_argument("--ec-additional-predict-ms", type=float, default=0, help="角域额外预测时间(ms)")
    parser.add_argument(
        "--spin-aware",
        action="store_true",
        default=True,
        help="启用单框小陀螺感知，抑制自旋切向速度污染（默认开启）",
    )
    parser.add_argument(
        "--no-spin-aware",
        action="store_false",
        dest="spin_aware",
        help="关闭小陀螺感知，回退到原始补偿逻辑",
    )
    parser.add_argument(
        "--spin-enter-threshold",
        type=float,
        default=0.66,
        help="进入小陀螺状态的置信度阈值(0~1)",
    )
    parser.add_argument(
        "--spin-exit-threshold",
        type=float,
        default=0.45,
        help="退出小陀螺状态的置信度阈值(0~1)",
    )
    parser.add_argument(
        "--spin-yaw-reverse-bias-deg",
        type=float,
        default=-5.0,
        help="小陀螺方向锁定后，沿假切向yaw速度反方向增加的固定偏置角度(deg, 设为0关闭)",
    )
    parser.add_argument(
        "--spin-yaw-dir-lock-min-conf",
        type=float,
        default=0.66,
        help="首次锁定小陀螺yaw方向所需的最小spin置信度(0~1)",
    )
    parser.add_argument(
        "--spin-yaw-dir-min-rate-dps",
        type=float,
        default=10.0,
        help="首次锁定小陀螺yaw方向所需的最小假切向yaw速度(deg/s)",
    )
    parser.add_argument(
        "--spin-yaw-dir-lock-threshold",
        type=float,
        default=4.0,
        help="首次锁定小陀螺yaw方向所需的累计分数阈值",
    )
    parser.add_argument(
        "--spin-yaw-dir-switch-min-conf",
        type=float,
        default=0.75,
        help="从当前yaw补偿方向切换到反方向所需的最小spin置信度(0~1)",
    )
    parser.add_argument(
        "--spin-yaw-dir-switch-min-rate-dps",
        type=float,
        default=15.0,
        help="切换yaw补偿方向所需的最小假切向yaw速度(deg/s)",
    )
    parser.add_argument(
        "--spin-yaw-dir-switch-threshold",
        type=float,
        default=4.0,
        help="从当前yaw补偿方向切换到反方向所需的累计分数阈值",
    )
    parser.add_argument(
        "--ec-keep-image-comp",
        action="store_true",
        default=False,
        help="即使有电控回传也保留像素速度时间补偿（不推荐）",
    )
    parser.add_argument("--max-comp-distance", type=float, default=6.0, help="时间补偿最大距离(m)")
    parser.add_argument("--model-bullet-speed", type=float, default=28.0, help="坐标模型训练弹速(m/s)")
    parser.add_argument("--model-latency-ms", type=float, default=0.0, help="坐标模型训练系统时延(ms)")
    parser.add_argument("--ballistic-enable", action="store_true", default=False, help="启用弹道补偿")
    parser.add_argument("--drag-k", type=float, default=0.02, help="二次阻力系数k(1/m)")
    parser.add_argument("--pitch-min", type=float, default=-10.0, help="弹道搜索最小pitch角度(deg)")
    parser.add_argument("--pitch-max", type=float, default=30.0, help="弹道搜索最大pitch角度(deg)")
    parser.add_argument("--ballistic-dt-ms", type=float, default=1.0, help="弹道积分步长(ms)")
    parser.add_argument("--use-ballistic-time", action="store_true", default=False, help="用弹道时间做预测补偿")
    parser.add_argument("--input-seq", type=int, default=5, help="预测输入序列长度")
    parser.add_argument("--sequence-length", type=int, default=20, help="预测缓冲区长度")
    parser.add_argument("--context-padding", type=int, default=10, help="ROI上下文扩展像素")
    parser.add_argument("--max-yaw-rate", type=float, default=120.0, help="yaw角速度上限(deg/s)")
    parser.add_argument("--max-pitch-rate", type=float, default=120.0, help="pitch角速度上限(deg/s)")
    parser.add_argument(
        "--rate-fast-alpha",
        type=float,
        default=0.25,
        help="目标角速度快速EMA系数(0~1)，越大对速度变化响应越快，用于跟踪加速/转向，建议0.4~0.6",
    )
    parser.add_argument(
        "--rate-slow-alpha",
        type=float,
        default=0.08,
        help="目标角速度慢速EMA系数(0~1)，用于小陀螺感知的平滑基准，建议0.05~0.12",
    )
    parser.add_argument(
        "--lost-threshold",
        type=int,
        default=10,
        help="连续丢失多少次后才发送空包进入扫描模式",
    )
    parser.add_argument("--show-window", action="store_true", default=True, help="显示检测/预测画面")
    parser.add_argument("--no-show-window", action="store_false", dest="show_window", help="关闭显示窗口")
    parser.add_argument("--window-name", default="Aim Scheduler", help="显示窗口名称")
    parser.add_argument(
        "--display-max-fps",
        type=float,
        default=30.0,
        help="显示线程最大刷新率(Hz)，仅在开启窗口时生效",
    )
    parser.add_argument("--invert-yaw", action="store_true", default=True, help="反转yaw角度方向")
    parser.add_argument("--no-invert-yaw", action="store_false", dest="invert_yaw", help="不反转yaw方向")
    parser.add_argument("--invert-pitch", action="store_true", help="反转pitch角度方向")
    parser.add_argument("--show-tx", action="store_true", default=True, help="显示每帧发送数据")
    parser.add_argument("--no-show-tx", action="store_false", dest="show_tx", help="关闭发送数据打印")
    parser.add_argument(
        "--profile-pred",
        action="store_true",
        default=False,
        help="预测耗时细分统计(会降低FPS)",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        default=False,
        help="录制比赛原始画面，用于后续离线训练LSTM",
    )
    parser.add_argument(
        "--record-path",
        default=None,
        help="录制输出路径；不填时自动保存到 recordings/ 目录",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=30.0,
        help="录制视频FPS",
    )
    parser.add_argument(
        "--record-fourcc",
        default="XVID",
        help="录制编码 fourcc，例如 XVID/mp4v/MJPG",
    )
    parser.add_argument(
        "--perf-log",
        action="store_true",
        default=False,
        help="按时间窗口输出整条链路的耗时聚合统计",
    )
    parser.add_argument(
        "--perf-log-interval-ms",
        type=float,
        default=1000.0,
        help="性能统计输出周期(ms)",
    )
    parser.add_argument(
        "--perf-log-percentiles",
        default="50,90",
        help="性能统计百分位，格式: 50,90 或 50,90,99",
    )
    parser.add_argument(
        "--perf-profile-hint",
        action="store_true",
        default=True,
        help="启动时检查 Jetson 性能模式并给出提示(默认开启)",
    )
    parser.add_argument(
        "--no-perf-profile-hint",
        action="store_false",
        dest="perf_profile_hint",
        help="关闭 Jetson 性能模式提示",
    )
    parser.add_argument(
        "--cv-threads",
        type=int,
        default=0,
        help="覆盖 OpenCV 线程数，0 表示使用系统默认值",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="覆盖 PyTorch CPU 线程数，0 表示使用系统默认值",
    )
    parser.add_argument(
        "--pred-async",
        action="store_true",
        default=False,
        help="启用预测异步流水（检测与预测并行）",
    )
    parser.add_argument(
        "--pred-max-lag",
        type=int,
        default=300,
        help="预测结果最大允许落后帧数（异步模式）",
    )
    parser.add_argument(
        "--lag-comp-enable",
        action="store_true",
        default=True,
        help="启用基于异步落后帧数的动态时间补偿",
    )
    parser.add_argument(
        "--no-lag-comp",
        action="store_false",
        dest="lag_comp_enable",
        help="禁用基于异步落后帧数的动态时间补偿",
    )
    parser.add_argument(
        "--lag-comp-max-ms",
        type=float,
        default=120.0,
        help="动态滞后补偿时间上限(ms)",
    )
    parser.add_argument(
        "--no-pred",
        action="store_false",
        dest="enable_prediction",
        default=True,
        help="关闭预测模块（仅使用检测）",
    )
    parser.add_argument(
        "--scale-intrinsics",
        action="store_true",
        default=False,
        help="根据当前分辨率缩放相机内参",
    )
    parser.add_argument(
        "--no-scale-intrinsics",
        action="store_false",
        dest="scale_intrinsics",
        help="不缩放相机内参",
    )
    parser.add_argument(
        "--auto-target-type",
        action="store_true",
        default=True,
        help="根据检测类别(class)自动选择装甲板类型：1/10为大装甲，其余为小装甲",
    )
    parser.add_argument(
        "--no-auto-target-type",
        action="store_false",
        dest="auto_target_type",
        help="不自动选择装甲板类型",
    )
    parser.add_argument(
        "--use-corners",
        action="store_true",
        default=False,
        help="启用角点检测用于PnP",
    )
    parser.add_argument(
        "--no-use-corners",
        action="store_false",
        dest="use_corners",
        help="禁用角点检测（默认）",
    )
    parser.add_argument(
        "--bbox-shrink",
        type=float,
        default=0.85,
        help="PnP使用的bbox缩放比例(0.1~1.0)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    use_daheng = not args.use_opencv
    if args.use_daheng:
        use_daheng = True
    swap_rb = args.swap_rb

    yolo_imgsz = None
    if args.yolo_imgsz:
        text = str(args.yolo_imgsz).strip()
        if "," in text:
            h, w = text.split(",", 1)
            yolo_imgsz = (int(h), int(w))
        else:
            size = int(text)
            yolo_imgsz = (size, size)

    perf_percentiles = []
    for part in str(args.perf_log_percentiles).split(","):
        text = part.strip()
        if not text:
            continue
        perf_percentiles.append(float(text))
    if not perf_percentiles:
        perf_percentiles = [50.0, 90.0]

    pipeline = AimPipeline(
        yolo_model_path=args.yolo_model,
        coord_model_path=args.coord_model,
        uart_port=args.port,
        uart_baud=args.baud,
        send_rate=args.rate,
        confidence_threshold=args.confidence,
        camera_id=args.camera_id,
        use_daheng=use_daheng,
        daheng_config=args.daheng_config,
        daheng_sn=args.daheng_sn,
        swap_rb=swap_rb,
        pnp_profile=args.pnp_profile,
        target_type=args.target_type,
        target_color=args.target_color,
        max_pnp_error=args.max_pnp_error,
        target_class_ids=_parse_csv_ints(args.target_class_ids),
        exclude_class_ids=_parse_csv_ints(args.exclude_class_ids),
        angle_scale=args.angle_scale,
        input_sequence_length=args.input_seq,
        sequence_length=args.sequence_length,
        context_padding=args.context_padding,
        max_yaw_rate=args.max_yaw_rate,
        max_pitch_rate=args.max_pitch_rate,
        lost_threshold=args.lost_threshold,
        show_window=args.show_window,
        window_name=args.window_name,
        display_max_fps=args.display_max_fps,
        invert_yaw=args.invert_yaw,
        invert_pitch=args.invert_pitch,
        show_tx=args.show_tx,
        gun_offset_y=args.gun_offset_y,
        scale_intrinsics=args.scale_intrinsics,
        auto_target_type=args.auto_target_type,
        use_corners=args.use_corners,
        bbox_shrink=args.bbox_shrink,
        yolo_max_det=args.yolo_max_det,
        yolo_log_speed=args.yolo_log_speed,
        yolo_verbose=args.yolo_verbose,
        yolo_imgsz=yolo_imgsz,
        detector_backend=args.detector_backend,
        perf_log=args.perf_log,
        perf_log_interval_s=args.perf_log_interval_ms / 1000.0,
        perf_log_percentiles=tuple(perf_percentiles),
        perf_profile_hint=args.perf_profile_hint,
        cv_threads=args.cv_threads,
        torch_threads=args.torch_threads,
        profile_pred=args.profile_pred,
        enable_prediction=args.enable_prediction,
        pred_async=args.pred_async,
        pred_max_lag=args.pred_max_lag,
        lag_comp_enable=args.lag_comp_enable,
        lag_comp_max_s=args.lag_comp_max_ms / 1000.0,
        bullet_speed_mps=args.bullet_speed,
        system_latency_s=args.system_latency_ms / 1000.0,
        enable_ec_feedback=args.ec_feedback,
        ec_feedback_invert_yaw=args.ec_invert_yaw,
        ec_feedback_invert_pitch=args.ec_invert_pitch,
        ec_t0_s=args.ec_t0_ms / 1000.0,
        ec_additional_predict_time_s=args.ec_additional_predict_ms / 1000.0,
        spin_aware=args.spin_aware,
        spin_enter_threshold=args.spin_enter_threshold,
        spin_exit_threshold=args.spin_exit_threshold,
        spin_yaw_reverse_bias_deg=args.spin_yaw_reverse_bias_deg,
        spin_yaw_dir_lock_min_conf=args.spin_yaw_dir_lock_min_conf,
        spin_yaw_dir_min_rate_dps=args.spin_yaw_dir_min_rate_dps,
        spin_yaw_dir_lock_threshold=args.spin_yaw_dir_lock_threshold,
        spin_yaw_dir_switch_min_conf=args.spin_yaw_dir_switch_min_conf,
        spin_yaw_dir_switch_min_rate_dps=args.spin_yaw_dir_switch_min_rate_dps,
        spin_yaw_dir_switch_threshold=args.spin_yaw_dir_switch_threshold,
        disable_image_time_comp_with_feedback=(not args.ec_keep_image_comp),
        max_comp_distance_m=args.max_comp_distance,
        model_bullet_speed_mps=args.model_bullet_speed,
        model_latency_s=args.model_latency_ms / 1000.0,
        ballistic_enable=args.ballistic_enable,
        ballistic_drag_k=args.drag_k,
        ballistic_pitch_min_deg=args.pitch_min,
        ballistic_pitch_max_deg=args.pitch_max,
        ballistic_dt_ms=args.ballistic_dt_ms,
        use_ballistic_time=args.use_ballistic_time,
        fire_confidence_threshold=args.fire_confidence_threshold,
        fire_force_interval_s=args.fire_force_interval_ms / 1000.0,
        record_video=args.record_video,
        record_path=args.record_path,
        record_fps=args.record_fps,
        record_fourcc=args.record_fourcc,
        rate_fast_alpha=args.rate_fast_alpha,
        rate_slow_alpha=args.rate_slow_alpha,
        latency_viz=args.latency_viz,
        latency_viz_window_s=args.latency_viz_window_s,
        latency_viz_csv=args.latency_viz_csv,
    )
    return pipeline.run()


if __name__ == "__main__":
    raise SystemExit(main())
