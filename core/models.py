#!/usr/bin/env python3
"""
Data Models for NPU Scheduler
NPU调度器的数据模型定义
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from .enums import ResourceType, TaskPriority, RuntimeType, ResourceState, TaskState


@dataclass
class ResourceSegment:
    """资源使用段 - 定义任务在特定资源上的执行信息"""
    
    resource_type: ResourceType
    bandwidth_duration_map: Dict[float, float]  # {带宽: 持续时间}
    start_time: float = 0.0  # 相对于任务开始的偏移时间
    segment_id: str = ""
    
    # 分段相关
    cut_points: List[Tuple[str, float, float]] = field(default_factory=list)  # (操作名, 位置, 开销)
    sub_segments: List['SubSegment'] = field(default_factory=list)
    
    def get_duration(self, bandwidth: float) -> float:
        """根据带宽获取执行时间"""
        return self.bandwidth_duration_map.get(bandwidth, 0.0)
    
    def get_min_duration(self) -> float:
        """获取最小执行时间（最高带宽）"""
        if not self.bandwidth_duration_map:
            return 0.0
        max_bandwidth = max(self.bandwidth_duration_map.keys())
        return self.bandwidth_duration_map[max_bandwidth]
    
    def get_max_duration(self) -> float:
        """获取最大执行时间（最低带宽）"""
        if not self.bandwidth_duration_map:
            return 0.0
        min_bandwidth = min(self.bandwidth_duration_map.keys())
        return self.bandwidth_duration_map[min_bandwidth]
    
    def add_cut_point(self, operation_name: str, position: float, overhead: float):
        """添加分段点"""
        self.cut_points.append((operation_name, position, overhead))
        self.cut_points.sort(key=lambda x: x[1])  # 按位置排序
    
    def generate_sub_segments(self, selected_cuts: List[str] = None) -> List['SubSegment']:
        """根据选中的分段点生成子段"""
        if not selected_cuts:
            selected_cuts = []
        
        # 过滤出选中的分段点
        active_cuts = [(name, pos, overhead) for name, pos, overhead in self.cut_points 
                      if name in selected_cuts]
        active_cuts.sort(key=lambda x: x[1])  # 按位置排序
        
        sub_segments = []
        positions = [0.0] + [cut[1] for cut in active_cuts] + [1.0]
        
        for i in range(len(positions) - 1):
            start_pos = positions[i]
            end_pos = positions[i + 1]
            segment_ratio = end_pos - start_pos
            
            # 计算子段的执行时间
            sub_bandwidth_map = {}
            for bandwidth, duration in self.bandwidth_duration_map.items():
                sub_bandwidth_map[bandwidth] = duration * segment_ratio
            
            # 如果有分段开销，添加到第一个子段之后的所有子段
            overhead = 0.0
            if i > 0 and i <= len(active_cuts):
                overhead = active_cuts[i-1][2]  # 使用前一个分段点的开销
            
            sub_segment = SubSegment(
                parent_segment_id=self.segment_id,
                sub_id=f"{self.segment_id}_sub_{i}",
                resource_type=self.resource_type,
                bandwidth_duration_map=sub_bandwidth_map,
                start_position=start_pos,
                end_position=end_pos,
                segmentation_overhead=overhead
            )
            sub_segments.append(sub_segment)
        
        self.sub_segments = sub_segments
        return sub_segments


@dataclass
class SubSegment:
    """子段 - 分段后的执行单元"""
    
    parent_segment_id: str
    sub_id: str
    resource_type: ResourceType
    bandwidth_duration_map: Dict[float, float]
    start_position: float  # 在原段中的相对位置 (0.0-1.0)
    end_position: float
    segmentation_overhead: float = 0.0  # 分段开销（毫秒）
    
    def get_duration(self, bandwidth: float) -> float:
        """获取执行时间（包含分段开销）"""
        base_duration = self.bandwidth_duration_map.get(bandwidth, 0.0)
        return base_duration + self.segmentation_overhead
    
    def get_effective_duration(self, bandwidth: float) -> float:
        """获取有效执行时间（不含开销）"""
        return self.bandwidth_duration_map.get(bandwidth, 0.0)


@dataclass
class ResourceUnit:
    """资源单元 - 表示单个计算单元"""
    
    unit_id: str
    resource_type: ResourceType
    bandwidth: float  # GOPS或类似单位
    state: ResourceState = ResourceState.IDLE
    
    # 可用性追踪
    available_time: float = 0.0  # 下次可用时间
    current_task: Optional[str] = None  # 当前执行的任务ID
    
    # 统计信息
    total_usage_time: float = 0.0
    task_count: int = 0
    
    def is_available(self, current_time: float) -> bool:
        """检查资源是否可用"""
        return self.available_time <= current_time and self.state == ResourceState.IDLE
    
    def reserve(self, task_id: str, start_time: float, duration: float):
        """预订资源"""
        self.current_task = task_id
        self.available_time = start_time + duration
        self.state = ResourceState.BUSY
        self.total_usage_time += duration
        self.task_count += 1
    
    def release(self):
        """释放资源"""
        self.current_task = None
        self.state = ResourceState.IDLE
    
    def get_utilization(self, total_time: float) -> float:
        """计算资源利用率"""
        if total_time <= 0:
            return 0.0
        return min(100.0, (self.total_usage_time / total_time) * 100.0)


@dataclass
class TaskScheduleInfo:
    """任务调度信息"""
    
    task_id: str
    start_time: float
    end_time: float
    assigned_resources: Dict[ResourceType, str]  # {资源类型: 资源ID}
    
    # 分段调度信息
    sub_segment_schedule: List[Tuple[str, float, float]] = field(default_factory=list)  # [(子段ID, 开始时间, 结束时间)]
    segmentation_overhead: float = 0.0
    
    # 运行时信息
    actual_latency: float = 0.0
    priority_delay: float = 0.0  # 由于优先级导致的延迟
    resource_wait_time: float = 0.0  # 等待资源的时间
    
    def __post_init__(self):
        """计算实际延迟"""
        self.actual_latency = self.end_time - self.start_time
    
    def add_sub_segment_schedule(self, sub_segment_id: str, start_time: float, end_time: float):
        """添加子段调度信息"""
        self.sub_segment_schedule.append((sub_segment_id, start_time, end_time))
        self.sub_segment_schedule.sort(key=lambda x: x[1])  # 按开始时间排序
    
    def get_total_execution_time(self) -> float:
        """获取总执行时间"""
        return self.end_time - self.start_time
    
    def get_resource_usage_summary(self) -> Dict[str, float]:
        """获取资源使用摘要"""
        summary = {}
        total_time = self.get_total_execution_time()
        
        for resource_type, resource_id in self.assigned_resources.items():
            summary[resource_id] = total_time
        
        return summary


@dataclass
class ResourceBinding:
    """资源绑定信息 - DSP_Runtime使用"""
    
    binding_id: str
    bound_resources: Set[str]  # 绑定的资源ID集合
    binding_start: float
    binding_end: float
    task_id: str
    runtime_type: RuntimeType = RuntimeType.DSP_RUNTIME
    
    def __post_init__(self):
        """验证绑定有效性"""
        if self.binding_end <= self.binding_start:
            raise ValueError(f"绑定结束时间必须大于开始时间: {self.binding_start} -> {self.binding_end}")
        
        if len(self.bound_resources) == 0:
            raise ValueError("绑定的资源集合不能为空")
    
    def get_duration(self) -> float:
        """获取绑定持续时间"""
        return self.binding_end - self.binding_start
    
    def overlaps_with(self, other: 'ResourceBinding') -> bool:
        """检查是否与另一个绑定重叠"""
        return not (self.binding_end <= other.binding_start or 
                   other.binding_end <= self.binding_start)
    
    def conflicts_with(self, other: 'ResourceBinding') -> bool:
        """检查是否与另一个绑定冲突（时间重叠且有共同资源）"""
        if not self.overlaps_with(other):
            return False
        
        return bool(self.bound_resources & other.bound_resources)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    
    # 基础指标
    makespan: float = 0.0  # 总完成时间
    average_latency: float = 0.0
    total_tasks: int = 0
    
    # 资源利用率
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    average_utilization: float = 0.0
    
    # 违规统计
    fps_violations: int = 0
    latency_violations: int = 0
    priority_violations: int = 0
    
    # 分段统计
    segmented_tasks: int = 0
    total_segmentation_overhead: float = 0.0
    average_segmentation_benefit: float = 0.0
    
    # 优先级统计
    priority_distribution: Dict[TaskPriority, int] = field(default_factory=dict)
    
    def calculate_efficiency_score(self) -> float:
        """计算效率评分 (0-100)"""
        if self.total_tasks == 0:
            return 0.0
        
        # 基础效率：利用率权重50%
        utilization_score = min(100.0, self.average_utilization)
        
        # 违规惩罚：每个违规-10分
        total_violations = self.fps_violations + self.latency_violations + self.priority_violations
        violation_penalty = min(50.0, total_violations * 10.0)
        
        # 分段奖励：有效分段+5分
        segmentation_bonus = min(10.0, self.segmented_tasks * 2.0)
        
        score = (utilization_score * 0.5) - violation_penalty + segmentation_bonus
        return max(0.0, min(100.0, score))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'makespan': self.makespan,
            'average_latency': self.average_latency,
            'total_tasks': self.total_tasks,
            'resource_utilization': self.resource_utilization,
            'average_utilization': self.average_utilization,
            'fps_violations': self.fps_violations,
            'latency_violations': self.latency_violations,
            'priority_violations': self.priority_violations,
            'segmented_tasks': self.segmented_tasks,
            'total_segmentation_overhead': self.total_segmentation_overhead,
            'average_segmentation_benefit': self.average_segmentation_benefit,
            'priority_distribution': {k.name: v for k, v in self.priority_distribution.items()},
            'efficiency_score': self.calculate_efficiency_score()
        }


@dataclass
class OptimizationResult:
    """优化结果"""
    
    # 优化配置
    task_configs: Dict[str, 'TaskConfig'] = field(default_factory=dict)
    
    # 性能指标
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    
    # 优化过程信息
    iterations: int = 0
    convergence_time: float = 0.0
    optimization_method: str = ""
    
    def get_improvement_summary(self, baseline_metrics: PerformanceMetrics) -> Dict[str, float]:
        """获取相对于基线的改进摘要"""
        if baseline_metrics.total_tasks == 0:
            return {}
        
        return {
            'makespan_improvement': ((baseline_metrics.makespan - self.metrics.makespan) / baseline_metrics.makespan) * 100,
            'latency_improvement': ((baseline_metrics.average_latency - self.metrics.average_latency) / baseline_metrics.average_latency) * 100,
            'utilization_improvement': self.metrics.average_utilization - baseline_metrics.average_utilization,
            'violation_reduction': (baseline_metrics.fps_violations + baseline_metrics.latency_violations) - (self.metrics.fps_violations + self.metrics.latency_violations),
            'efficiency_improvement': self.metrics.calculate_efficiency_score() - baseline_metrics.calculate_efficiency_score()
        }


@dataclass  
class TaskConfig:
    """任务配置（用于优化）"""
    
    task_id: str
    priority: TaskPriority
    runtime_type: RuntimeType
    segmentation_configs: Dict[str, int] = field(default_factory=dict)  # {段ID: 配置索引}
    core_assignments: Dict[str, str] = field(default_factory=dict)  # {段ID: 核心ID}
    
    def __hash__(self):
        """使其可哈希，用于集合和字典"""
        return hash((
            self.task_id,
            self.priority,
            self.runtime_type,
            tuple(sorted(self.segmentation_configs.items())),
            tuple(sorted(self.core_assignments.items()))
        ))
    
    def copy(self) -> 'TaskConfig':
        """创建配置副本"""
        return TaskConfig(
            task_id=self.task_id,
            priority=self.priority,
            runtime_type=self.runtime_type,
            segmentation_configs=self.segmentation_configs.copy(),
            core_assignments=self.core_assignments.copy()
        )


if __name__ == "__main__":
    # 测试数据模型
    print("=== NPU Scheduler 数据模型测试 ===")
    
    # 测试ResourceSegment
    segment = ResourceSegment(
        resource_type=ResourceType.NPU,
        bandwidth_duration_map={2.0: 40, 4.0: 20, 8.0: 10},
        segment_id="test_segment"
    )
    
    segment.add_cut_point("op1", 0.3, 0.15)
    segment.add_cut_point("op2", 0.7, 0.12)
    
    print(f"\n段信息:")
    print(f"  最小时间: {segment.get_min_duration()}ms")
    print(f"  最大时间: {segment.get_max_duration()}ms")
    print(f"  分段点数: {len(segment.cut_points)}")
    
    # 测试子段生成
    sub_segments = segment.generate_sub_segments(["op1", "op2"])
    print(f"  生成子段: {len(sub_segments)}个")
    
    # 测试ResourceUnit
    npu = ResourceUnit("NPU_0", ResourceType.NPU, bandwidth=8.0)
    print(f"\n资源单元:")
    print(f"  可用性: {npu.is_available(0.0)}")
    
    npu.reserve("T1", 0.0, 10.0)
    print(f"  预订后可用时间: {npu.available_time}")
    print(f"  利用率: {npu.get_utilization(100.0):.1f}%")
    
    print("\n✅ 数据模型测试完成")
