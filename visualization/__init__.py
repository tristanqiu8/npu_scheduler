"""
Visualization module for NPU Scheduler
可视化模块，提供优雅的调度结果展示
"""

# 主要可视化类（使用elegant_viz作为默认）
from .elegant_viz import ElegantSchedulerVisualizer as SchedulerVisualizer

# Chrome追踪功能
from .chrome_tracer import ChromeTracer

# 版本信息
__version__ = "2.0.0"

# 导出接口
__all__ = [
    'SchedulerVisualizer',
    'ChromeTracer',
    '__version__'
]

# 便捷函数
def create_visualizer(scheduler, style='elegant'):
    """创建可视化器的便捷函数"""
    if style == 'elegant':
        return SchedulerVisualizer(scheduler)
    else:
        raise ValueError(f"不支持的可视化风格: {style}")


def quick_plot(scheduler, time_window=None, export_trace=False):
    """快速绘制调度结果"""
    visualizer = SchedulerVisualizer(scheduler)
    
    # 绘制甘特图
    visualizer.plot_elegant_gantt(bar_height=0.35, spacing=0.8)
    
    # 导出Chrome追踪（如果需要）
    if export_trace:
        tracer = ChromeTracer(scheduler)
        tracer.export("quick_trace.json")
        print("📊 Chrome追踪文件已保存为: quick_trace.json")


# 模块初始化
def _check_visualization_dependencies():
    """检查可视化依赖"""
    missing_deps = []
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append('matplotlib')
    
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    
    if missing_deps:
        print(f"⚠️  可视化模块缺少依赖: {', '.join(missing_deps)}")
        print("   请运行: pip install matplotlib numpy")
        return False
    
    return True


# 执行依赖检查
_viz_deps_ok = _check_visualization_dependencies()

# 设置matplotlib后端（如果可用）
if _viz_deps_ok:
    import matplotlib
    # 自动选择合适的后端
    try:
        matplotlib.use('TkAgg')  # 首选GUI后端
    except:
        try:
            matplotlib.use('Agg')  # 备选非GUI后端
        except:
            pass  # 使用默认后端
