#!/usr/bin/env python3
"""
修复后的main.py - 统一入口点
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config import parse_command_line_args, create_config_from_args, SchedulerConfig


def print_banner():
    """打印程序启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                        NPU Scheduler                         ║
║              多资源神经网络任务调度器                          ║
║                                                              ║
║  功能特性:                                                    ║
║  • 多优先级任务调度                                           ║
║  • NPU/DSP 多资源管理                                        ║
║  • 网络分段优化                                               ║
║  • 智能资源绑定                                               ║
║  • 实时性能监控                                               ║
║  • Chrome Tracing 支持                                       ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_config_summary(config: SchedulerConfig):
    """打印配置摘要"""
    print("🔧 当前配置:")
    print(f"  • 分段功能: {'✅ 启用' if config.enable_segmentation else '❌ 禁用'}")
    print(f"  • 补丁修复: {'✅ 启用' if config.apply_patches else '❌ 禁用'}")
    print(f"  • 调度验证: {'✅ 启用' if config.enable_validation else '❌ 禁用'}")
    print(f"  • 可视化风格: {config.visualization_style}")
    print(f"  • Chrome追踪: {'✅ 启用' if config.export_chrome_tracing else '❌ 禁用'}")
    print(f"  • 时间窗口: {config.default_time_window}ms")
    print(f"  • 详细输出: {'✅ 启用' if config.verbose_logging else '❌ 禁用'}")
    print()


def run_basic_demo(config: SchedulerConfig):
    """运行基础演示"""
    print("🚀 运行基础演示...")
    try:
        from examples.basic_demo import run_basic_demo as run_demo
        return run_demo(config)
    except Exception as e:
        print(f"❌ 基础演示失败: {e}")
        return None


def run_optimization_demo(config: SchedulerConfig):
    """运行优化演示"""
    print("🎯 运行优化演示...")
    try:
        from examples.optimization_demo import run_optimization_demo as run_demo
        return run_demo(config)
    except Exception as e:
        print(f"❌ 优化演示失败: {e}")
        return None


def run_tutorial_demo(config: SchedulerConfig):
    """运行教程演示"""
    print("🎓 运行教程演示...")
    try:
        from examples.tutorial_demo import run_tutorial_demo as run_demo
        return run_demo(interactive=False)  # 非交互模式
    except Exception as e:
        print(f"❌ 教程演示失败: {e}")
        return None


def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_command_line_args()
        
        # 创建配置
        config = create_config_from_args(args)
        
        # 静默模式检查
        if not args.quiet:
            print_banner()
            print_config_summary(config)
        
        # 记录开始时间
        start_time = time.time()
        
        # 根据模式运行相应演示
        result = None
        if args.mode == 'basic':
            result = run_basic_demo(config)
        elif args.mode == 'optimization':
            result = run_optimization_demo(config)
        elif args.mode == 'tutorial':
            result = run_tutorial_demo(config)
        else:
            print(f"❌ 未知的运行模式: {args.mode}")
            print("可用模式: basic, optimization, tutorial")
            return 1
        
        # 计算运行时间
        elapsed_time = time.time() - start_time
        
        if not args.quiet:
            if result is not None:
                print(f"\n✅ 演示完成! 总耗时: {elapsed_time:.2f}秒")
                
                if config.export_chrome_tracing:
                    print("📊 Chrome追踪文件已生成，使用以下步骤查看:")
                    print("   1. 打开Chrome浏览器")
                    print("   2. 访问 chrome://tracing")
                    print("   3. 点击Load按钮加载生成的JSON文件")
                    print("   4. 使用WASD键导航时间轴")
            else:
                print(f"\n⚠️ 演示未成功完成，耗时: {elapsed_time:.2f}秒")
        
        return 0 if result is not None else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断程序执行")
        return 130
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        if '--verbose' in sys.argv:
            import traceback
            traceback.print_exc()
        return 1


def show_help():
    """显示帮助信息"""
    help_text = """
NPU Scheduler 使用指南

基本用法:
  python main.py [选项]

运行模式:
  --mode basic         基础调度演示（默认）
  --mode optimization  优化算法演示
  --mode tutorial      新手教程演示

配置选项:
  --config production   生产环境配置
  --config development  开发环境配置（默认）
  --config testing     测试环境配置

功能开关:
  --segmentation       启用网络分段功能
  --no-patches         禁用调度器补丁
  --no-validation      禁用调度验证
  --export-trace       导出Chrome追踪文件

输出控制:
  --verbose, -v        详细输出模式
  --quiet, -q          静默输出模式

性能参数:
  --time-window 500    设置调度时间窗口（毫秒）
  --iterations 10      设置优化迭代次数

示例:
  # 基础演示，启用分段功能
  python main.py --mode basic --segmentation
  
  # 优化演示，生产配置，导出追踪
  python main.py --mode optimization --config production --export-trace
  
  # 教程演示，详细输出
  python main.py --mode tutorial --verbose
  
  # 快速测试，静默模式
  python main.py --config testing --quiet

更多信息请查看README.md文档。
"""
    print(help_text)


if __name__ == "__main__":
    # 检查是否请求帮助
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_help()
        sys.exit(0)
    
    # 运行主程序
    exit_code = main()
    sys.exit(exit_code)


# =================== config.py 更新 - 添加tutorial模式 ===================

#!/usr/bin/env python3
"""
config.py 更新 - 在parse_command_line_args中添加tutorial模式
"""

# 在原有的config.py中，需要更新这一行：
# parser.add_argument(
#     '--mode', 
#     choices=['basic', 'optimization', 'fixed'], 
#     default='basic',
#     help='运行模式选择'
# )

# 改为：
# parser.add_argument(
#     '--mode', 
#     choices=['basic', 'optimization', 'tutorial'], 
#     default='basic',
#     help='运行模式选择'
# )


# =================== examples/__init__.py (新增) ===================

#!/usr/bin/env python3
"""
Examples module for NPU Scheduler
演示模块入口
"""

# 导出主要的演示函数
from .basic_demo import run_basic_demo
from .optimization_demo import run_optimization_demo  
from .tutorial_demo import run_tutorial_demo

__version__ = "2.0.0"

__all__ = [
    'run_basic_demo',
    'run_optimization_demo',
    'run_tutorial_demo',
    '__version__'
]

# 演示注册表（用于动态调用）
DEMO_REGISTRY = {
    'basic': run_basic_demo,
    'optimization': run_optimization_demo,
    'tutorial': run_tutorial_demo
}


def get_demo_function(demo_name: str):
    """根据名称获取演示函数"""
    if demo_name not in DEMO_REGISTRY:
        raise ValueError(f"未知演示: {demo_name}。可用演示: {list(DEMO_REGISTRY.keys())}")
    
    return DEMO_REGISTRY[demo_name]


def list_available_demos():
    """列出所有可用演示"""
    print("📋 可用演示列表:")
    demos_info = {
        'basic': '基础调度演示 - 展示核心调度功能',
        'optimization': '优化算法演示 - 展示调度优化算法',
        'tutorial': '新手教程演示 - 逐步学习指导'
    }
    
    for name, description in demos_info.items():
        print(f"   • {name}: {description}")


if __name__ == "__main__":
    list_available_demos()


# =================== 顶层目录需要的额外文件 ===================

# .gitignore
"""
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Test outputs
.pytest_cache/
.coverage
htmlcov/
.tox/

# NPU Scheduler specific
*.json  # Chrome tracing files
test_output/
logs/
"""

# MANIFEST.in
"""
include README.md
include requirements.txt
include config.py
include main.py
recursive-include core *.py
recursive-include examples *.py
recursive-include tests *.py
recursive-include utils *.py
recursive-include visualization *.py
recursive-include optimization *.py
"""

# pyproject.toml (现代Python项目配置)
"""
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "npu-scheduler"
version = "2.0.0"
description = "Multi-resource Neural Network Task Scheduler"
authors = [{name = "NPU Scheduler Team"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
dependencies = [
    "numpy>=1.21.0",
    "matplotlib>=3.5.0",
]

[project.optional-dependencies]
optimization = ["scipy>=1.7.0"]
dev = ["pytest>=6.2.0", "black>=21.0.0", "flake8>=3.9.0"]
all = ["scipy>=1.7.0", "pytest>=6.2.0", "black>=21.0.0", "flake8>=3.9.0"]

[project.scripts]
npu-scheduler = "main:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

[tool.black]
line-length = 88
target-version = ['py38']

[tool.flake8]
max-line-length = 88
extend-ignore = ["E203", "W503"]
"""

# LICENSE (MIT License)
"""
MIT License

Copyright (c) 2024 NPU Scheduler Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""