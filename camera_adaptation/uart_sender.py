#!/usr/bin/env python
# coding=utf-8

"""
UART sender for the k5 12-byte protocol.
Matches the behavior of send_fixed.py while being reusable as a module.
"""

import argparse
import time
import numpy as np
import serial
from dataclasses import dataclass


# ------------------------- Serial defaults (aligned with k5.py) -------------------------
SERIAL_PORT_DEFAULT = "/dev/ttyUSB0"
SERIAL_BAUD_DEFAULT = 115200
SERIAL_TIMEOUT_DEFAULT = 0  # non-blocking


# ------------------------- k5 12-byte frame constants -------------------------
HEADER_BYTE = 0x43
TAIL_BYTE = 0xFF
ANGLE_SCALE_DEFAULT = 100


def _to_int16_signed_le(value):
    """Split signed int16 into little-endian low/high bytes."""
    v = int(value) & 0xFFFF
    lo = v & 0xFF
    hi = (v >> 8) & 0xFF
    return lo, hi


def _to_uint16_le(value):
    """Split unsigned int16 into little-endian low/high bytes."""
    v = max(0, int(value)) & 0xFFFF
    lo = v & 0xFF
    hi = (v >> 8) & 0xFF
    return lo, hi


def build_armor_packet(
    yaw_deg,
    pitch_deg,
    armor_x,
    armor_y,
    armor_cmd=0x01,
    fire_cmd=0x00,
    angle_scale=ANGLE_SCALE_DEFAULT,
    tx_timestamp_ms=None,
):
    """构造16字节协议（含视觉时间戳）：
    DATA[0]  =0x43
    DATA[1]  =装甲板检验指令（默认0x00）
    DATA[2]  =开火指令（默认0x00）
    DATA[3..4]=Yaw(centideg) 小端 低8/高8（有符号）
    DATA[5..6]=Pitch(centideg) 小端 低8/高8（有符号）
    DATA[7..8]=x 像素坐标 小端（无符号）
    DATA[9..10]=y 像素坐标 小端（无符号）
    DATA[11..14]=视觉发送时间戳 uint32 毫秒 小端
                 电控原样回传到 IMU 反馈包，用于计算单程延迟
    DATA[15]=帧尾（0xFF）
    """
    import time as _time
    yaw_scaled = int(np.round(yaw_deg * angle_scale))
    pitch_scaled = int(np.round(pitch_deg * angle_scale))
    yaw_lo, yaw_hi = _to_int16_signed_le(yaw_scaled)
    pitch_lo, pitch_hi = _to_int16_signed_le(pitch_scaled)
    x_lo, x_hi = _to_uint16_le(armor_x)
    y_lo, y_hi = _to_uint16_le(armor_y)

    if tx_timestamp_ms is None:
        ts = int(_time.time() * 1000) & 0xFFFFFFFF
    else:
        ts = int(tx_timestamp_ms) & 0xFFFFFFFF
    ts0 = ts & 0xFF
    ts1 = (ts >> 8) & 0xFF
    ts2 = (ts >> 16) & 0xFF
    ts3 = (ts >> 24) & 0xFF

    packet = [
        HEADER_BYTE,
        armor_cmd & 0xFF,
        fire_cmd & 0xFF,
        yaw_lo, yaw_hi,
        pitch_lo, pitch_hi,
        x_lo, x_hi,
        y_lo, y_hi,
        ts0, ts1, ts2, ts3,
        TAIL_BYTE,
    ]
    return bytes(packet)


@dataclass
class UartSenderConfig:
    port: str = SERIAL_PORT_DEFAULT
    baud: int = SERIAL_BAUD_DEFAULT
    rate: float = 50.0
    yaw: float = 0.0
    pitch: float = 0.0
    x: int = 0
    y: int = 0
    armor_cmd: int = 0x01
    fire_cmd: int = 0x00
    angle_scale: float = ANGLE_SCALE_DEFAULT
    serial_timeout: float = SERIAL_TIMEOUT_DEFAULT


class UartSender:
    """UART sender with k5 protocol frames."""

    def __init__(self, config: UartSenderConfig):
        self.config = config
        self.serial = None
        self._tx_interval = self._rate_to_interval(self.config.rate)

    @staticmethod
    def _rate_to_interval(rate):
        return 1.0 / max(1e-9, rate)

    def _open_serial(self):
        try:
            self.serial = serial.Serial(
                self.config.port,
                self.config.baud,
                timeout=self.config.serial_timeout,
            )
        except Exception as exc:
            print(f"串口打开失败: {exc}")
            return False

        if not self.serial.is_open:
            print("串口未打开")
            return False
        return True

    def _print_help(self):
        print(f"串口打开成功: {self.config.port} @ {self.config.baud}")
        print(
            f"固定发送: yaw={self.config.yaw} deg, pitch={self.config.pitch} deg, "
            f"rate={self.config.rate} Hz"
        )

    def run(self):
        try:
            if not self._open_serial():
                return 1
            self._print_help()

            next_tx_time = time.time()
            tx_count = 0
            tx_bytes = 0
            win_start = time.time()

            while True:
                now = time.time()

                if now < next_tx_time:
                    time.sleep(max(0.0, next_tx_time - now))
                    continue

                packet = build_armor_packet(
                    yaw_deg=self.config.yaw,
                    pitch_deg=self.config.pitch,
                    armor_x=self.config.x,
                    armor_y=self.config.y,
                    armor_cmd=self.config.armor_cmd,
                    fire_cmd=self.config.fire_cmd,
                    angle_scale=self.config.angle_scale,
                )
                hex_str = ' '.join(f"{b:02X}" for b in packet)
                print(f"TX: yaw={self.config.yaw:.2f} pitch={self.config.pitch:.2f} | {hex_str}")

                try:
                    sent = self.serial.write(packet)
                    tx_bytes += int(sent)
                except Exception as exc:
                    print(f"串口发送失败: {exc}")

                tx_count += 1
                next_tx_time = now + self._tx_interval

                if (now - win_start) >= 1.0:
                    print(f"TX 速率: {tx_count} pkt/s, {tx_bytes} B/s")
                    tx_count = 0
                    tx_bytes = 0
                    win_start = now

        except KeyboardInterrupt:
            pass
        finally:
            try:
                if self.serial:
                    self.serial.close()
                    print("串口已关闭")
            except Exception:
                pass
        return 0


def _parse_args():
    parser = argparse.ArgumentParser(
        description="通过 UART 发送固定 yaw/pitch（k5 12 字节帧）"
    )
    parser.add_argument("--port", default=SERIAL_PORT_DEFAULT, help="串口设备路径")
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD_DEFAULT, help="波特率")
    parser.add_argument("--rate", type=float, default=50.0, help="发送频率 (Hz)")
    parser.add_argument("--yaw", type=float, default=0.0, help="固定 yaw 角度 (deg)")
    parser.add_argument("--pitch", type=float, default=0.0, help="固定 pitch 角度 (deg)")
    parser.add_argument("--x", type=int, default=0, help="像素 x (uint16)")
    parser.add_argument("--y", type=int, default=0, help="像素 y (uint16)")
    parser.add_argument(
        "--armor-cmd",
        type=lambda x: int(x, 0),
        default=0x01,
        help="装甲板指令字节（支持 0x 前缀）",
    )
    parser.add_argument(
        "--fire-cmd",
        type=lambda x: int(x, 0),
        default=0x00,
        help="开火指令字节（支持 0x 前缀）",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    args.yaw = 0.0
    args.pitch = 0.0

    config = UartSenderConfig(
        port=args.port,
        baud=args.baud,
        rate=args.rate,
        yaw=args.yaw,
        pitch=args.pitch,
        x=args.x,
        y=args.y,
        armor_cmd=args.armor_cmd,
        fire_cmd=args.fire_cmd,
    )
    sender = UartSender(config)
    return sender.run()


if __name__ == "__main__":
    raise SystemExit(main())
