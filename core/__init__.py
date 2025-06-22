"""
Core module for NPU Scheduler
核心调度器模块，包含所有基础组件
"""

# 枚举定义
from .enums import (
    ResourceType,
    TaskPriority, 
    RuntimeType,
    SegmentationStrategy
)

# 数据模型
from .models import (
    ResourceSegment,
    ResourceUnit,
    TaskScheduleInfo,
    ResourceBinding
)

# 任务类
from .task import NNTask

# 调度器
from .scheduler import MultiResourceScheduler

# 调度器工厂
from .scheduler_factory import SchedulerFactory

# 版本信息
__version__ = "2.0.0"
__author__ = "NPU Scheduler Team"

# 导出所有公共接口
__all__ = [
    # 枚举
    'ResourceType',
    'TaskPriority', 
    'RuntimeType',
    'SegmentationStrategy',
    
    # 数据模型
    'ResourceSegment',
    'ResourceUnit', 
    'TaskScheduleInfo',
    'ResourceBinding',
    
    # 核心类
    'NNTask',
    'MultiResourceScheduler',
    'SchedulerFactory',
    
    # 元信息
    '__version__',
    '__author__'
]

# 模块级别的便捷函数
def create_basic_scheduler(enable_segmentation=False, apply_patches=True):
    """创建基础调度器的便捷函数"""
    from config import SchedulerConfig
    
    config = SchedulerConfig(
        enable_segmentation=enable_segmentation,
        apply_patches=apply_patches
    )
    return SchedulerFactory.create_scheduler(config)


def create_task(task_id, name, priority=TaskPriority.NORMAL, runtime_type=RuntimeType.ACPU_RUNTIME):
    """创建任务的便捷函数"""
    return NNTask(task_id, name, priority=priority, runtime_type=runtime_type)


# 模块初始化检查
def _check_dependencies():
    """检查必要的依赖项"""
    try:
        import matplotlib
        import numpy
        return True
    except ImportError as e:
        print(f"警告: 缺少必要依赖 {e}")
        return False


# 执行依赖检查
_dependencies_ok = _check_dependencies()

if not _dependencies_ok:
    print("提示: 请运行 'pip install -r requirements.txt' 安装依赖")
