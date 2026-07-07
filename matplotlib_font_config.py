# matplotlib字体配置文件
# 将此文件放在项目根目录，或在代码中导入

import matplotlib.pyplot as plt

def configure_matplotlib_fonts():
    """配置matplotlib字体设置，确保标题正常显示"""
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    
    print("✓ matplotlib字体设置已配置")
    print(f"  字体族: {plt.rcParams['font.family']}")
    print(f"  基础字体大小: {plt.rcParams['font.size']}")
    print(f"  标题字体大小: {plt.rcParams['axes.titlesize']}")
    print(f"  图表标题字体大小: {plt.rcParams['figure.titlesize']}")

# 自动配置字体
if __name__ == "__main__":
    configure_matplotlib_fonts()
