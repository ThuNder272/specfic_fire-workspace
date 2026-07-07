#!/usr/bin/env python
# coding=utf-8

"""Camera API wrapper for Daheng/GigE cameras (gxipy) with OpenCV fallback.

提供类化接口 CameraAPI，可被其他程序导入并调用：
- CameraAPI.start() 启动后台采集线程
- CameraAPI.read() 获取最近一帧 (BGR numpy array)
- CameraAPI.get_jpeg() 获取 JPEG bytes
- CameraAPI.start_record(path) / stop_record() 控制视频写入
- CameraAPI.stop() 停止采集并释放资源

如果系统没有 gxipy，会自动回退到 OpenCV 的 cv2.VideoCapture(0)。
示例用法见 __main__。
"""

import threading
import time
import os
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import gxipy as gx
    _HAS_GXI = True
except Exception:
    gx = None
    _HAS_GXI = False


class CameraAPI:
    """摄像头抽象，支持 Daheng (gxipy) 和 OpenCV 回退。

    设计要点：
    - 后台线程持续抓取最新帧，read() 返回最新一帧（非阻塞）。
    - 支持可选的视频录制（OpenCV VideoWriter）。
    - 提供 get_jpeg() 方便网络/API 使用。
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        device_sn: Optional[str] = None,
        allow_opencv_fallback: bool = True,
    ):
        self.config_path = config_path
        self.device_sn = device_sn
        self.allow_opencv_fallback = allow_opencv_fallback

        self._use_gx = False
        self._dev = None
        self._device_manager = None
        self._stream_on = False

        # background thread
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # frame storage
        self._frame = None
        self._frame_lock = threading.Lock()

        # recording
        self._writer = None
        self._recording = False

    def open(self) -> bool:
        """打开设备，返回是否成功。"""
        if _HAS_GXI:
            try:
                self._device_manager = gx.DeviceManager()
                dev_num, dev_info_list = self._device_manager.update_device_list()
                if dev_num == 0:
                    print("未发现 gxipy 设备，回退到 OpenCV")
                    self._device_manager = None
                    if self.allow_opencv_fallback:
                        return self._open_opencv()
                    return False

                # 如果提供了序列号，尝试按序列号打开
                if self.device_sn:
                    self._dev = self._device_manager.open_device_by_sn(self.device_sn)
                else:
                    str_sn = dev_info_list[0].get("sn")
                    self._dev = self._device_manager.open_device_by_sn(str_sn)

                # 导入配置（若提供）
                config_loaded = False
                if self.config_path:
                    if os.path.exists(self.config_path):
                        try:
                            self._dev.import_config_file(self.config_path)
                            config_loaded = True
                        except Exception as e:
                            print(f"导入相机配置失败: {e}")
                    else:
                        print(f"相机配置文件不存在: {self.config_path}")
                if self.config_path:
                    print(
                        f"相机配置文件: {self.config_path} | "
                        f"导入{'成功' if config_loaded else '失败/未导入'}"
                    )
                else:
                    print("相机配置文件: 未指定")

                self._dev.stream_on()
                self._use_gx = True
                self._stream_on = True
                return True
            except Exception as e:
                self._device_manager = None
                if self.allow_opencv_fallback:
                    print(f"gxipy 打开失败，回退到 OpenCV：{e}")
                    return self._open_opencv()
                print(f"gxipy 打开失败：{e}")
                return False
        else:
            if self.allow_opencv_fallback:
                return self._open_opencv()
            print("gxipy 不可用，且禁用 OpenCV 回退")
            return False

    def _open_opencv(self) -> bool:
        try:
            self._dev = cv2.VideoCapture(0)
            opened = bool(self._dev.isOpened())
            if not opened:
                print("OpenCV 无法打开默认相机")
            return opened
        except Exception as e:
            print(f"OpenCV 打开相机失败: {e}")
            return False

    def start(self):
        """启动后台采集线程（open() 之后调用）。"""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """后台循环，不断抓取最新帧。"""
        while not self._stop_event.is_set():
            frame = None
            try:
                if self._use_gx and self._dev:
                    raw_image = self._dev.data_stream[0].get_image()
                    if raw_image is None:
                        print("⚠️ 大恒相机 get_image() 返回 None，摄像头可能未连接或无数据")
                        time.sleep(0.05)
                        continue
                    rgb_image = raw_image.convert("RGB")
                    numpy_image = rgb_image.get_numpy_array()
                    frame = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
                elif isinstance(self._dev, cv2.VideoCapture):
                    ret, f = self._dev.read()
                    if ret:
                        frame = f
                else:
                    # unknown device
                    time.sleep(0.01)

            except Exception as e:
                # 读取失败时打印并继续
                print(f"读取帧失败: {e}")

            if frame is not None:
                with self._frame_lock:
                    self._frame = frame.copy()
                # 如果正在录制，则写入
                if self._recording and self._writer is not None:
                    try:
                        self._writer.write(frame)
                    except Exception as e:
                        print(f"写入视频失败: {e}")

            # 控制抓取频率略微休眠
            time.sleep(0.005)

    def read(self) -> Optional[np.ndarray]:
        """返回最近一帧的 BGR numpy array（或 None）。"""
        with self._frame_lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def get_jpeg(self, quality: int = 80) -> Optional[bytes]:
        """返回最近一帧的 JPEG bytes（用于网络传输）。"""
        frame = self.read()
        if frame is None:
            return None
        ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ret:
            return None
        return buf.tobytes()

    def start_record(self, path: str, fourcc_str: str = 'XVID', fps: float = 30.0, size: Optional[Tuple[int, int]] = None):
        """开始录制到文件。若 size 为空，将尝试从当前帧获取尺寸。"""
        if self._recording:
            return
        if size is None:
            f = self.read()
            if f is None:
                raise RuntimeError("无法确定视频尺寸：尚未收到帧")
            size = (f.shape[1], f.shape[0])

        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        self._writer = cv2.VideoWriter(path, fourcc, fps, size)
        if not self._writer.isOpened():
            self._writer = None
            raise RuntimeError(f"无法打开视频写入: {path}")
        self._recording = True

    def stop_record(self):
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
        self._writer = None
        self._recording = False

    def stop(self):
        """停止后台线程并释放设备。"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

        # 停止设备
        if self._use_gx and self._dev:
            try:
                self._dev.stream_off()
            except Exception:
                pass
            try:
                self._dev.close_device()
            except Exception:
                pass
            self._device_manager = None
        elif isinstance(self._dev, cv2.VideoCapture):
            try:
                self._dev.release()
            except Exception:
                pass

        self.stop_record()


def _demo_loop(camera: CameraAPI):
    """简单 demo：显示窗口并在按 ESC 时退出。"""
    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = camera.read()
            if frame is not None:
                cv2.imshow('video', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            time.sleep(0.01)
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # 命令行示例：保留原脚本的行为
    cam = CameraAPI(config_path=None)
    ok = cam.open()
    if not ok:
        print('无法打开相机，退出')
        raise SystemExit(1)
    cam.start()

    # 自动开始录制到当前目录（可根据需要修改路径）
    ts = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    out_path = os.path.join(os.getcwd(), f'video_{ts}.avi')
    # 等待第一帧以获得尺寸
    for _ in range(100):
        if cam.read() is not None:
            break
        time.sleep(0.01)

    try:
        cam.start_record(out_path, fps=30.0)
    except Exception as e:
        print(f'开始录制失败: {e}')

    try:
        _demo_loop(cam)
    finally:
        cam.stop()
        print('已停止相机')
