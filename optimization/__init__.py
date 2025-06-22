"""
Optimization module for NPU Scheduler
优化模块，提供任务调度优化算法
"""

# 主要优化器
from .optimizer import (
    TaskSchedulerOptimizer,
    SchedulingSearchSpace,
    SchedulingObjective,
    OptimizationSolution
)

# 版本信息
__version__ = "2.0.0"

# 导出接口
__all__ = [
    'TaskSchedulerOptimizer',
    'SchedulingSearchSpace', 
    'SchedulingObjective',
    'OptimizationSolution',
    '__version__'
]

# 便捷函数
def create_optimizer(scheduler, objective_weights=None):
    """创建优化器的便捷函数"""
    optimizer = TaskSchedulerOptimizer(scheduler)
    
    if objective_weights:
        optimizer.set_objective_weights(**objective_weights)
    
    return optimizer


def quick_optimize(scheduler, tasks, time_window=500.0, iterations=10):
    """快速优化便捷函数"""
    optimizer = create_optimizer(scheduler)
    
    # 为所有任务设置基本搜索空间
    from core.enums import TaskPriority, RuntimeType, ResourceType
    
    for task in tasks:
        # 创建基本搜索空间
        search_space = SchedulingSearchSpace(
            task_id=task.task_id,
            allowed_priorities=list(TaskPriority),
            allowed_runtime_types=list(RuntimeType),
            segmentation_options={},  # 将根据任务自动填充
            available_cores={}  # 将根据调度器自动填充
        )
        
        # 自动填充分段选项
        if hasattr(task, 'segments'):
            for segment in task.segments:
                if hasattr(segment, 'cut_points') and segment.cut_points:
                    search_space.segmentation_options[segment.segment_id] = [0, 1, 2]
        
        # 自动填充可用核心
        for resource_type in [ResourceType.NPU, ResourceType.DSP]:
            if resource_type in scheduler.resources:
                cores = [r.unit_id for r in scheduler.resources[resource_type]]
                if cores:
                    search_space.available_cores[resource_type] = cores
        
        optimizer.define_search_space(task.task_id, search_space)
    
    # 运行优化
    return optimizer.optimize_greedy(time_window=time_window, iterations=iterations)


# 模块初始化检查
def _check_optimization_dependencies():
    """检查优化模块依赖"""
    missing_deps = []
    
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    
    # 高级优化算法需要scipy
    try:
        import scipy
    except ImportError:
        print("📝 提示: 安装scipy可启用更多优化算法 (pip install scipy)")
    
    if missing_deps:
        print(f"⚠️  优化模块缺少依赖: {', '.join(missing_deps)}")
        return False
    
    return True


# 执行依赖检查
_opt_deps_ok = _check_optimization_dependencies()
