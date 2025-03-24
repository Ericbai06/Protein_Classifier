"""
为matplotlib图表提供中文支持的工具模块
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import numpy as np

def set_chinese_font():
    """
    设置matplotlib的中文字体支持
    
    该函数会检测系统上可用的中文字体，并相应地配置matplotlib
    """
    # 检测操作系统
    system = sys.platform
    
    # 尝试设置字体的方法
    font_set = False
    
    # 特定系统的字体设置
    if system.startswith('darwin'):  # macOS
        try:
            # 直接设置苹果系统常用中文字体
            plt.rcParams['font.family'] = ['Arial Unicode MS', 'STHeiti', 'PingFang SC', 'Heiti SC']
            plt.rcParams['axes.unicode_minus'] = False
            print("使用macOS默认中文字体")
            font_set = True
        except:
            pass
            
    elif system.startswith('win'):  # Windows
        try:
            # 直接设置Windows常用中文字体
            plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun']
            plt.rcParams['axes.unicode_minus'] = False
            print("使用Windows默认中文字体")
            font_set = True
        except:
            pass
            
    elif system.startswith('linux'):  # Linux
        try:
            # 直接设置Linux常用中文字体
            plt.rcParams['font.family'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback']
            plt.rcParams['axes.unicode_minus'] = False
            print("使用Linux默认中文字体")
            font_set = True
        except:
            pass
    
    # 如果以上特定系统设置失败，尝试更通用的方法
    if not font_set:
        try:
            # 列出系统上所有可用字体
            font_names = [f.name for f in fm.fontManager.ttflist]
            
            # 查找常见的中文字体
            chinese_fonts = []
            for font in ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'STHeiti', 'WenQuanYi Micro Hei', 
                         'Hiragino Sans GB', 'Heiti TC', 'Heiti SC', 'SimSun', 'Noto Sans CJK SC']:
                if font in font_names:
                    chinese_fonts.append(font)
                    
            if chinese_fonts:
                plt.rcParams['font.sans-serif'] = chinese_fonts
                plt.rcParams['axes.unicode_minus'] = False
                print(f"使用检测到的中文字体: {chinese_fonts[0]}")
                font_set = True
        except:
            pass
    
    # 最后的备选方案：硬编码尝试常见的字体文件路径
    if not font_set:
        # 系统字体文件路径
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',  # macOS
            '/System/Library/Fonts/STHeiti Light.ttc',  # macOS 备选
            '/System/Library/Fonts/STHeiti Medium.ttc',  # macOS 备选
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # Linux
            'C:/Windows/Fonts/simhei.ttf',  # Windows
            'C:/Windows/Fonts/msyh.ttc',  # Windows
        ]
        
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    # 注册字体
                    font_prop = fm.FontProperties(fname=font_path)
                    # 设置字体
                    plt.rcParams['font.family'] = font_prop.get_name()
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"使用字体文件: {font_path}")
                    font_set = True
                    break
            except:
                continue
    
    # 最后的提示
    if not font_set:
        print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")
        print("可用的字体有:", fm.findSystemFonts())
        
    # 显示一些关于当前字体设置的信息
    print(f"当前字体设置: {plt.rcParams['font.family']}")
    
    # 尝试创建一个测试图形，以确保设置有效
    try:
        plt.figure(figsize=(1, 1))
        plt.text(0.5, 0.5, '测试中文', ha='center', va='center')
        plt.savefig('font_test.png')
        plt.close()
        print("已保存字体测试图像到 font_test.png")
    except Exception as e:
        print(f"创建测试图像时出错: {e}")

def configure_plot_style():
    """配置matplotlib使用中文字体并设置美观的绘图风格"""
    # 1. 检测系统
    system = sys.platform
    
    # 2. 尝试设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 修复坐标轴负号显示问题
    
    # 获取系统中可用的字体
    font_names = [f.name for f in fm.fontManager.ttflist]
    
    # 常见中文字体优先级列表
    chinese_fonts = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Microsoft YaHei', 
                    'PingFang SC', 'Hiragino Sans GB', 'WenQuanYi Micro Hei', 
                    'Heiti SC', 'Heiti TC', 'SimSun', 'Noto Sans CJK SC']
    
    # 检查哪些中文字体可用
    available_chinese_fonts = []
    for font in chinese_fonts:
        if font in font_names:
            available_chinese_fonts.append(font)
    
    if available_chinese_fonts:
        # 设置字体优先级列表
        plt.rcParams['font.sans-serif'] = available_chinese_fonts + plt.rcParams['font.sans-serif']
    else:
        # 基于操作系统设置默认字体
        if system.startswith('darwin'):  # macOS
            # 尝试注册macOS中的中文字体
            macos_font_paths = [
                '/System/Library/Fonts/PingFang.ttc',
                '/System/Library/Fonts/STHeiti Light.ttc',
                '/System/Library/Fonts/STHeiti Medium.ttc',
                '/Library/Fonts/Arial Unicode.ttf'
            ]
            
            for font_path in macos_font_paths:
                if os.path.exists(font_path):
                    try:
                        fm.fontManager.addfont(font_path)
                        print(f"已加载字体: {font_path}")
                    except:
                        pass
            
            # 更新字体列表
            font_names = [f.name for f in fm.fontManager.ttflist]
            
            # 为macOS设置常见的中文字体
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti', 'Heiti SC'] + plt.rcParams['font.sans-serif']
            
        elif system.startswith('win'):  # Windows
            # Windows中的常见中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun'] + plt.rcParams['font.sans-serif']
            
        elif system.startswith('linux'):  # Linux
            # Linux中的常见中文字体
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans CN'] + plt.rcParams['font.sans-serif']
    
    # 3. 设置美观的绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')  # 使用较新的seaborn风格
    
    # 4. 全局绘图设置
    plt.rcParams['figure.figsize'] = (10, 6)  # 默认图形大小
    plt.rcParams['figure.dpi'] = 100  # 图形分辨率
    
    # 设置轴标签和标题的字体大小
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    
    # 设置刻度标签的字体大小
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # 设置图例的字体大小和位置
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['legend.loc'] = 'best'
    
    # 设置保存图片时的紧凑布局
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    
    # 创建测试图像验证字体设置是否生效
    _test_font_settings()

def _test_font_settings():
    """创建测试图像验证字体设置是否生效（内部函数）"""
    # 创建一个简单的测试图
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 添加中文文本以测试字体
    ax.set_title('中文字体测试')
    ax.set_xlabel('横坐标 (X)')
    ax.set_ylabel('纵坐标 (Y)')
    
    # 添加一些数据
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y, label='正弦曲线')
    
    # 添加中文图例
    ax.legend()
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 保存测试图像
    plt.tight_layout()
    plt.savefig('font_test_plot_utils.png')
    
    # 关闭图形，不显示
    plt.close(fig) 