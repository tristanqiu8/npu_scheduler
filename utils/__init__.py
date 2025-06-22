"""
Utilities module for NPU Scheduler
工具模块，包含验证器、补丁和其他辅助功能
"""

# 验证器
from .validator import ScheduleValidator, validate_schedule

# 补丁系统
from .patches import patch_scheduler, apply_all_patches

# 版本信息
__version__ = "2.0.0"

# 导出接口
__all__ = [
    'ScheduleValidator',
    'validate_schedule',
    'patch_scheduler', 
    'apply_all_patches',
    '__version__'
]

# 便捷函数
def quick_validate(scheduler, verbose=True):
    """快速验证调度结果"""
    is_valid, errors = validate_schedule(scheduler)
    
    if verbose:
        if is_valid:
            print("✅ 调度验证通过，没有发现错误")
        else:
            print(f"❌ 调度验证失败，发现 {len(errors)} 个错误:")
            for i, error in enumerate(errors, 1):
                print(f"   {i}. {error}")
    
    return is_valid, errors


def apply_recommended_patches(scheduler):
    """应用推荐的补丁配置"""
    try:
        patch_scheduler(scheduler)
        return True
    except Exception as e:
        print(f"⚠️  应用补丁时出错: {e}")
        return False


# 模块初始化检查
def _check_utils_dependencies():
    """检查工具模块依赖"""
    # 工具模块通常只依赖标准库
    return True


# 执行依赖检查
_utils_deps_ok = _check_utils_dependencies()
