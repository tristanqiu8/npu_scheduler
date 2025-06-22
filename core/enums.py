#!/usr/bin/env python3
"""
Enums for NPU Scheduler
定义所有枚举类型
"""

from enum import Enum, IntEnum


class ResourceType(Enum):
    """资源类型枚举"""
    NPU = "NPU"
    DSP = "DSP"


class TaskPriority(IntEnum):
    """任务优先级枚举 (数值越小优先级越高)"""
    CRITICAL = 0    # 关键优先级
    HIGH = 1        # 高优先级
    NORMAL = 2      # 普通优先级
    LOW = 3         # 低优先级


class RuntimeType(Enum):
    """运行时类型枚举"""
    DSP_RUNTIME = "DSP_Runtime"      # 资源绑定运行时
    ACPU_RUNTIME = "ACPU_Runtime"    # 流水线运行时


class SegmentationStrategy(Enum):
    """分段策略枚举"""
    NO_SEGMENTATION = "NoSegmentation"              # 无分段
    ADAPTIVE_SEGMENTATION = "AdaptiveSegmentation"  # 自适应分段
    FORCED_SEGMENTATION = "ForcedSegmentation"      # 强制分段
    CUSTOM_SEGMENTATION = "CustomSegmentation"      # 自定义分段


class SchedulingStrategy(Enum):
    """调度策略枚举"""
    PRIORITY_FIRST = "PriorityFirst"        # 优先级优先
    RESOURCE_BALANCED = "ResourceBalanced"  # 资源均衡
    LATENCY_OPTIMIZED = "LatencyOptimized"  # 延迟优化
    THROUGHPUT_OPTIMIZED = "ThroughputOptimized"  # 吞吐量优化


class ResourceState(Enum):
    """资源状态枚举"""
    IDLE = "Idle"              # 空闲
    BUSY = "Busy"              # 忙碌
    BOUND = "Bound"            # 已绑定
    MAINTENANCE = "Maintenance"  # 维护中


class TaskState(Enum):
    """任务状态枚举"""
    PENDING = "Pending"        # 等待中
    RUNNING = "Running"        # 运行中
    COMPLETED = "Completed"    # 已完成
    FAILED = "Failed"          # 失败
    CANCELLED = "Cancelled"    # 已取消


class OptimizationObjective(Enum):
    """优化目标枚举"""
    MINIMIZE_LATENCY = "MinimizeLatency"          # 最小化延迟
    MAXIMIZE_THROUGHPUT = "MaximizeThroughput"    # 最大化吞吐量
    BALANCE_UTILIZATION = "BalanceUtilization"   # 平衡利用率
    MINIMIZE_VIOLATIONS = "MinimizeViolations"   # 最小化违规


class ValidationLevel(Enum):
    """验证级别枚举"""
    BASIC = "Basic"          # 基础验证
    STANDARD = "Standard"    # 标准验证
    STRICT = "Strict"        # 严格验证
    COMPREHENSIVE = "Comprehensive"  # 全面验证


# 便捷的映射字典
PRIORITY_NAMES = {
    TaskPriority.CRITICAL: "关键",
    TaskPriority.HIGH: "高",
    TaskPriority.NORMAL: "普通", 
    TaskPriority.LOW: "低"
}

RUNTIME_DESCRIPTIONS = {
    RuntimeType.DSP_RUNTIME: "资源绑定模式 - 执行期间独占资源",
    RuntimeType.ACPU_RUNTIME: "流水线模式 - 允许资源共享"
}

SEGMENTATION_DESCRIPTIONS = {
    SegmentationStrategy.NO_SEGMENTATION: "不使用网络分段",
    SegmentationStrategy.ADAPTIVE_SEGMENTATION: "根据负载自适应分段",
    SegmentationStrategy.FORCED_SEGMENTATION: "强制使用分段",
    SegmentationStrategy.CUSTOM_SEGMENTATION: "使用自定义分段配置"
}


def get_priority_name(priority: TaskPriority, lang='zh') -> str:
    """获取优先级名称"""
    if lang == 'zh':
        return PRIORITY_NAMES.get(priority, str(priority.name))
    else:
        return priority.name


def get_higher_priority(priority: TaskPriority) -> TaskPriority:
    """获取更高的优先级"""
    if priority.value > 0:
        return TaskPriority(priority.value - 1)
    return priority


def get_lower_priority(priority: TaskPriority) -> TaskPriority:
    """获取更低的优先级"""
    max_priority_value = max(p.value for p in TaskPriority)
    if priority.value < max_priority_value:
        return TaskPriority(priority.value + 1)
    return priority


def is_higher_priority(p1: TaskPriority, p2: TaskPriority) -> bool:
    """判断p1是否比p2优先级更高"""
    return p1.value < p2.value


def validate_enum_combination(runtime: RuntimeType, segmentation: SegmentationStrategy) -> bool:
    """验证枚举组合的合法性"""
    # DSP_Runtime 通常不使用复杂分段
    if runtime == RuntimeType.DSP_RUNTIME and segmentation == SegmentationStrategy.FORCED_SEGMENTATION:
        return False
    
    return True


if __name__ == "__main__":
    # 测试枚举功能
    print("=== NPU Scheduler 枚举测试 ===")
    
    print("\n优先级测试:")
    for priority in TaskPriority:
        print(f"  {priority.name} ({priority.value}): {get_priority_name(priority)}")
    
    print(f"\n优先级比较:")
    print(f"  CRITICAL > HIGH: {is_higher_priority(TaskPriority.CRITICAL, TaskPriority.HIGH)}")
    print(f"  HIGH > LOW: {is_higher_priority(TaskPriority.HIGH, TaskPriority.LOW)}")
    
    print(f"\n运行时类型:")
    for runtime in RuntimeType:
        print(f"  {runtime.name}: {RUNTIME_DESCRIPTIONS[runtime]}")
    
    print(f"\n分段策略:")
    for strategy in SegmentationStrategy:
        print(f"  {strategy.name}: {SEGMENTATION_DESCRIPTIONS[strategy]}")
    
    print(f"\n枚举组合验证:")
    test_combinations = [
        (RuntimeType.DSP_RUNTIME, SegmentationStrategy.NO_SEGMENTATION),
        (RuntimeType.DSP_RUNTIME, SegmentationStrategy.FORCED_SEGMENTATION),
        (RuntimeType.ACPU_RUNTIME, SegmentationStrategy.ADAPTIVE_SEGMENTATION)
    ]
    
    for runtime, segmentation in test_combinations:
        valid = validate_enum_combination(runtime, segmentation)
        status = "✅" if valid else "❌"
        print(f"  {runtime.value} + {segmentation.value}: {status}")
