"""
Matplotlib中文字体辅助脚本

这个脚本可以帮助:
1. 检测系统中的可用字体
2. 尝试安装中文字体
3. 显示matplotlib配置信息
4. 创建测试图表检查中文显示
"""

import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import subprocess
import numpy as np

def list_system_fonts():
    """列出系统中的所有字体"""
    fonts = fm.findSystemFonts()
    print(f"系统中检测到 {len(fonts)} 个字体文件")
    
    # 获取字体名称
    font_names = sorted([f.name for f in fm.fontManager.ttflist])
    print("\n系统中的字体名称:")
    for i, name in enumerate(font_names):
        if i > 0 and i % 3 == 0:
            print()
        print(f"{name:<30}", end="")
    print("\n")
    
    # 查找常见中文字体
    chinese_fonts = ['SimHei', 'SimSun', 'Microsoft YaHei', 'PingFang SC', 'STHeiti', 
                     'FangSong', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Hiragino Sans GB']
    print("检测常见中文字体:")
    for font in chinese_fonts:
        if font in font_names:
            print(f"✓ {font} (已安装)")
        else:
            print(f"✗ {font} (未安装)")
    return font_names

def show_matplotlib_config():
    """显示matplotlib的配置信息"""
    print("\nMatplotlib配置信息:")
    print(f"配置文件位置: {matplotlib.get_configdir()}")
    print(f"缓存目录: {matplotlib.get_cachedir()}")
    print(f"后端: {matplotlib.get_backend()}")
    print(f"字体目录: {os.path.join(matplotlib.get_data_path(), 'fonts/ttf')}")
    
    # 当前字体设置
    print("\n当前字体设置:")
    print(f"font.family: {plt.rcParams['font.family']}")
    if 'font.sans-serif' in plt.rcParams:
        print(f"font.sans-serif: {plt.rcParams['font.sans-serif']}")
    print(f"axes.unicode_minus: {plt.rcParams['axes.unicode_minus']}")

def test_chinese_display():
    """创建一个测试图表，检查中文显示"""
    print("\n创建测试图表...")
    plt.figure(figsize=(10, 6))
    
    # 尝试多种字体设置
    font_settings = [
        {'name': '默认设置', 'font': None},
        {'name': 'Arial Unicode MS', 'font': {'family': 'Arial Unicode MS'}},
        {'name': 'SimHei', 'font': {'family': 'SimHei'}},
        {'name': 'STHeiti', 'font': {'family': 'STHeiti'}},
        {'name': 'Microsoft YaHei', 'font': {'family': 'Microsoft YaHei'}}
    ]
    
    # 添加文本以测试不同字体
    for i, setting in enumerate(font_settings):
        if setting['font']:
            text = f"中文测试 {setting['name']}"
            plt.text(0.5, 0.9 - i*0.15, text, fontdict=setting['font'], 
                     ha='center', va='center', fontsize=16)
        else:
            text = f"中文测试 {setting['name']}"
            plt.text(0.5, 0.9 - i*0.15, text, ha='center', va='center', fontsize=16)
    
    plt.title('中文字体测试')
    plt.xlabel('横坐标标签')
    plt.ylabel('纵坐标标签')
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # 保存图片
    plt.savefig('chinese_font_test.png', dpi=300)
    print(f"测试图表已保存为 chinese_font_test.png")
    
    # 显示图表
    plt.show()

def install_chinese_fonts():
    """尝试安装中文字体"""
    system = sys.platform
    
    print(f"\n检测到操作系统: {system}")
    
    if system.startswith('darwin'):  # macOS
        print("macOS系统通常已包含中文字体如PingFang SC或STHeiti")
        print("检查系统字体...")
        
        # 检查常见的macOS中文字体
        mac_fonts = [
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
            '/Library/Fonts/Arial Unicode.ttf'
        ]
        
        for font in mac_fonts:
            if os.path.exists(font):
                print(f"找到字体文件: {font}")
            else:
                print(f"未找到字体文件: {font}")
                
    elif system.startswith('win'):  # Windows
        print("Windows系统通常已包含中文字体如SimSun、SimHei和Microsoft YaHei")
        print("检查系统字体...")
        
        # 检查常见的Windows中文字体
        win_fonts = [
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/simsun.ttc',
            'C:/Windows/Fonts/msyh.ttc'
        ]
        
        for font in win_fonts:
            if os.path.exists(font):
                print(f"找到字体文件: {font}")
            else:
                print(f"未找到字体文件: {font}")
                
    elif system.startswith('linux'):  # Linux
        print("Linux系统可能需要安装中文字体")
        print("尝试安装WenQuanYi Micro Hei字体...")
        
        try:
            # Ubuntu/Debian系统
            subprocess.run(['sudo', 'apt-get', 'update'])
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'fonts-wqy-microhei'])
            print("字体安装完成，请重启Python进程使更改生效")
        except:
            try:
                # CentOS/Fedora系统
                subprocess.run(['sudo', 'yum', 'install', '-y', 'wqy-microhei-fonts'])
                print("字体安装完成，请重启Python进程使更改生效")
            except:
                print("自动安装失败，请手动安装中文字体")
    
    # 刷新Matplotlib字体缓存
    print("\n刷新Matplotlib字体缓存...")
    fm._rebuild()
    print("字体缓存已更新")

def configure_chinese_font():
    """配置matplotlib使用中文字体"""
    print("\n配置Matplotlib使用中文字体...")
    
    # 获取系统中的字体名称
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
        print(f"找到可用的中文字体: {', '.join(available_chinese_fonts)}")
        plt.rcParams['font.sans-serif'] = available_chinese_fonts + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        print(f"已配置字体优先级: {plt.rcParams['font.sans-serif'][:5]}")
    else:
        print("未找到可用的中文字体，使用默认设置")
    
    # 创建配置文件
    config_dir = matplotlib.get_configdir()
    config_file = os.path.join(config_dir, 'matplotlibrc')
    
    try:
        with open(config_file, 'w') as f:
            f.write("# Matplotlib configuration for Chinese font support\n")
            if available_chinese_fonts:
                f.write(f"font.sans-serif: {', '.join(available_chinese_fonts)}, DejaVu Sans, Bitstream Vera Sans, Computer Modern Sans Serif, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, sans-serif\n")
            f.write("axes.unicode_minus: False\n")
        print(f"配置已保存到 {config_file}")
    except:
        print(f"无法写入配置文件 {config_file}")

if __name__ == "__main__":
    print("=" * 60)
    print("Matplotlib中文字体助手")
    print("=" * 60)
    
    # 显示matplotlib版本
    print(f"Matplotlib版本: {matplotlib.__version__}")
    print(f"Python版本: {sys.version}")
    print(f"操作系统: {sys.platform}")
    
    # 显示菜单
    while True:
        print("\n请选择操作:")
        print("1. 列出系统中的字体")
        print("2. 显示Matplotlib配置信息")
        print("3. 测试中文显示")
        print("4. 尝试安装中文字体")
        print("5. 配置Matplotlib使用中文字体")
        print("6. 执行所有操作")
        print("0. 退出")
        
        choice = input("\n输入选项 [0-6]: ")
        
        if choice == '1':
            list_system_fonts()
        elif choice == '2':
            show_matplotlib_config()
        elif choice == '3':
            test_chinese_display()
        elif choice == '4':
            install_chinese_fonts()
        elif choice == '5':
            configure_chinese_font()
        elif choice == '6':
            list_system_fonts()
            show_matplotlib_config()
            configure_chinese_font()
            test_chinese_display()
            install_chinese_fonts()
        elif choice == '0':
            print("退出程序...")
            break
        else:
            print("无效选项，请重新输入") 