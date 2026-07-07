#!/usr/bin/env python3
"""
快速开始指南 - 装甲板检测 + 预测 + PnP角度 + 串口发送
"""

def print_header(title):
    """打印标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_step(step_num, title, description):
    """打印步骤"""
    print(f"\n{step_num}. {title}")
    print(f"   {description}")


def print_command(command, description):
    """打印命令"""
    print(f"   💻 命令: {command}")
    print(f"   📝 说明: {description}")


def main():
    """主函数"""
    print_header("🚀 装甲板锁定 + 串口发送快速开始")

    print("\n📋 项目概述:")
    print("   目标：检测装甲板、预测目标位置、PnP解算角度，并通过UART发送两帧（检测+预测）")

    print_header("🎯 完整使用流程")

    # 步骤1: 环境准备
    print_step(1, "环境准备", "确保Python环境和依赖包已安装")
    print_command("conda activate py310", "激活Python环境")
    print_command("pip install -r requirements.txt", "安装依赖包")

    # 步骤2: 数据/模型准备
    print_step(2, "模型准备", "准备YOLO与坐标预测权重")
    print("   📁 需要准备的文件:")
    print("      - best.pt (YOLO权重，项目根目录)")
    print("      - coordinate_prediction_models/Coordinate-Prediction-Model_best.pth (预测权重)")

    # 步骤3: 主流程运行
    print_step(3, "运行主流程", "实时锁定并串口发送（检测+预测两帧，默认大恒相机）")
    print_command(
        "python aim_scheduler.py --port /dev/ttyUSB0 --baud 115200 --rate 50 --show-window",
        "启动实时锁定与串口发送",
    )
    print("   💡 默认会加载 /workspace/RobotMaster/paper/MER-139-210U3C(KE0210010001).txt")
    print("   💡 如需使用电脑摄像头，加参数: --use-opencv")
    print("   💡 如需指定其他相机配置文件，加参数: --daheng-config <path>")
    print("   💡 若大恒相机不可用，当前不会自动回退到电脑摄像头")
    print("   💡 默认打印串口发送数据；如需关闭，加参数: --no-show-tx")
    print("   💡 默认开启显示窗口与 yaw 反向；如需关闭，加参数: --no-show-window / --no-invert-yaw")
    print("   💡 若转动方向相反，可加参数: --invert-yaw / --invert-pitch")
    print("   💡 可用 --max-yaw-rate/--max-pitch-rate 限制角速度")

    # 步骤4: 串口调试
    print_step(4, "串口调试", "固定数据发送，用于验证串口链路")
    print_command(
        "python camera_adaptation/uart_sender.py --port /dev/ttyUSB0 --baud 115200 --rate 50",
        "固定发送k5协议帧",
    )

    # 步骤5: 可选归档脚本
    print_step(5, "归档脚本（可选）", "演示/对比脚本已归档")
    print_command(
        "python archive/demos/coordinate_prediction_demo.py",
        "坐标预测演示（归档）",
    )
    print_command(
        "python archive/analysis/coordinate_prediction_comparison.py",
        "多模型对比（归档）",
    )

    print_header("⚙️ 关键参数")

    print("\n📝 常用参数说明:")
    print("   --port / --baud     串口端口与波特率")
    print("   --rate              发送频率(Hz)，每周期发送两帧")
    print("   --pnp-profile       相机内参配置名(默认 mer_139_210u3c)")
    print("   --use-daheng        使用大恒相机")
    print("   --daheng-config     大恒相机配置文件路径")
    print("   --input-seq         预测输入序列长度")
    print("   --context-padding   ROI上下文扩展像素")

    print_header("🚨 注意事项")
    print("\n⚠️  重要提醒:")
    print("   1. 串口设备需要权限（如 /dev/ttyUSB0）")
    print("   2. PnP角度依赖相机内参，建议校准后更新")
    print("   3. 预测未就绪时，预测帧会回退为检测帧")

    print_header("📚 更多信息")
    print("\n📖 文档:")
    print("   - README.md: 项目说明与入口")
    print("   - USAGE.md: 运行指令与参数")

    print("\n🎉 完成！")


if __name__ == "__main__":
    main()
