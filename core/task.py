#!/usr/bin/env python3
"""
Neural Network Task Class
神经网络任务类 - 定义AI任务的结构和属性
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field

from .enums import TaskPriority, RuntimeType, ResourceType, SegmentationStrategy, TaskState
from .models import ResourceSegment, SubSegment


class NNTask:
    """神经网络任务类"""
    
    def __init__(self, 
                 task_id: str,
                 name: str,
                 priority: TaskPriority = TaskPriority.NORMAL,
                 runtime_type: RuntimeType = RuntimeType.ACPU_RUNTIME,
                 segmentation_strategy: SegmentationStrategy = SegmentationStrategy.NO_SEGMENTATION):
        
        # 基础属性
        self.task_id = task_id
        self.name = name
        self.priority = priority
        self.runtime_type = runtime_type
        self.segmentation_strategy = segmentation_strategy
        
        # 资源需求
        self.segments: List[ResourceSegment] = []
        
        # 性能需求
        self.fps_requirement: float = 10.0      # 默认10 FPS
        self.latency_requirement: float = 100.0  # 默认100ms延迟
        self.min_interval_ms: float = 100.0      # 最小执行间隔
        
        # 依赖关系
        self.dependencies: List[str] = []        # 依赖的任务ID列表
        
        # 分段相关
        self.preset_cut_configurations: Dict[str, List[List[str]]] = {}  # {段ID: [配置列表]}
        self.selected_cut_config_index: Dict[str, int] = {}              # {段ID: 选中的配置索引}
        self.current_segmentation: Dict[str, List[SubSegment]] = {}      # 当前的分段结果
        
        # 调度状态
        self.state: TaskState = TaskState.PENDING
        self.last_execution_time: float = -float('inf')
        self.schedule_info: Optional['TaskScheduleInfo'] = None
        
        # 统计信息
        self.execution_count: int = 0
        self.total_execution_time: float = 0.0
        self.total_segmentation_overhead: float = 0.0
    
    def set_npu_only(self, bandwidth_duration_map: Dict[float, float], segment_id: str):
        """设置NPU专用配置"""
        segment = ResourceSegment(
            resource_type=ResourceType.NPU,
            bandwidth_duration_map=bandwidth_duration_map,
            start_time=0.0,
            segment_id=segment_id
        )
        self.segments = [segment]  # 清空并设置为唯一段
    
    def set_dsp_npu_sequence(self, sequence_config: List[Tuple]):
        """设置DSP+NPU顺序执行配置
        
        Args:
            sequence_config: [(资源类型, 带宽-时间映射, 开始时间, 段ID), ...]
        """
        self.segments = []
        
        for resource_type, bandwidth_map, start_time, segment_id in sequence_config:
            segment = ResourceSegment(
                resource_type=resource_type,
                bandwidth_duration_map=bandwidth_map,
                start_time=start_time,
                segment_id=segment_id
            )
            self.segments.append(segment)
    
    def set_performance_requirements(self, fps: float, latency: float):
        """设置性能需求"""
        self.fps_requirement = fps
        self.latency_requirement = latency
        
        # 更新最小执行间隔
        if fps > 0:
            self.min_interval_ms = 1000.0 / fps
        else:
            self.min_interval_ms = 100.0  # 默认值
    
    def add_dependency(self, task_id: str):
        """添加任务依赖"""
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)
    
    def remove_dependency(self, task_id: str):
        """移除任务依赖"""
        if task_id in self.dependencies:
            self.dependencies.remove(task_id)
    
    def add_cut_points_to_segment(self, segment_id: str, cut_points: List[Tuple[str, float, float]]):
        """为指定段添加分段点
        
        Args:
            segment_id: 段ID
            cut_points: [(操作名, 位置(0-1), 开销(ms)), ...]
        """
        segment = self._find_segment_by_id(segment_id)
        if segment:
            for op_name, position, overhead in cut_points:
                segment.add_cut_point(op_name, position, overhead)
    
    def set_preset_cut_configurations(self, segment_id: str, configurations: List[List[str]]):
        """设置预设的分段配置
        
        Args:
            segment_id: 段ID
            configurations: [配置1, 配置2, ...], 每个配置是操作名列表
        """
        self.preset_cut_configurations[segment_id] = configurations
        self.selected_cut_config_index[segment_id] = 0  # 默认选择第一个配置
    
    def select_cut_configuration(self, segment_id: str, config_index: int):
        """选择分段配置"""
        if segment_id in self.preset_cut_configurations:
            configs = self.preset_cut_configurations[segment_id]
            if 0 <= config_index < len(configs):
                self.selected_cut_config_index[segment_id] = config_index
                self._apply_segmentation_configuration(segment_id, config_index)
    
    def _apply_segmentation_configuration(self, segment_id: str, config_index: int):
        """应用分段配置"""
        segment = self._find_segment_by_id(segment_id)
        if not segment:
            return
        
        if segment_id not in self.preset_cut_configurations:
            return
        
        configurations = self.preset_cut_configurations[segment_id]
        if config_index >= len(configurations):
            return
        
        selected_cuts = configurations[config_index]
        
        # 生成子段
        sub_segments = segment.generate_sub_segments(selected_cuts)
        self.current_segmentation[segment_id] = sub_segments
        
        # 计算分段开销
        overhead = sum(ss.segmentation_overhead for ss in sub_segments)
        self.total_segmentation_overhead += overhead
    
    def get_sub_segments_for_scheduling(self) -> List[SubSegment]:
        """获取用于调度的所有子段"""
        all_sub_segments = []
        
        for segment_id, sub_segments in self.current_segmentation.items():
            all_sub_segments.extend(sub_segments)
        
        return all_sub_segments
    
    def _find_segment_by_id(self, segment_id: str) -> Optional[ResourceSegment]:
        """根据ID查找段"""
        for segment in self.segments:
            if segment.segment_id == segment_id:
                return segment
        return None
    
    @property
    def is_segmented(self) -> bool:
        """检查任务是否已分段"""
        return len(self.current_segmentation) > 0 and any(
            len(sub_segs) > 1 for sub_segs in self.current_segmentation.values()
        )
    
    def requires_resource_binding(self) -> bool:
        """检查是否需要资源绑定"""
        return (self.runtime_type == RuntimeType.DSP_RUNTIME and 
                len(self.segments) > 1)
    
    def get_estimated_duration(self, resource_bandwidths: Dict[ResourceType, float]) -> float:
        """估算任务执行时间"""
        if self.is_segmented:
            return self._estimate_segmented_duration(resource_bandwidths)
        else:
            return self._estimate_regular_duration(resource_bandwidths)
    
    def _estimate_regular_duration(self, resource_bandwidths: Dict[ResourceType, float]) -> float:
        """估算常规任务执行时间"""
        max_duration = 0.0
        
        for segment in self.segments:
            resource_type = segment.resource_type
            if resource_type in resource_bandwidths:
                bandwidth = resource_bandwidths[resource_type]
                duration = segment.get_duration(bandwidth)
                segment_end = segment.start_time + duration
                max_duration = max(max_duration, segment_end)
        
        return max_duration
    
    def _estimate_segmented_duration(self, resource_bandwidths: Dict[ResourceType, float]) -> float:
        """估算分段任务执行时间"""
        max_duration = 0.0
        
        for segment_id, sub_segments in self.current_segmentation.items():
            for sub_segment in sub_segments:
                resource_type = sub_segment.resource_type
                if resource_type in resource_bandwidths:
                    bandwidth = resource_bandwidths[resource_type]
                    duration = sub_segment.get_duration(bandwidth)  # 包含分段开销
                    max_duration = max(max_duration, duration)
        
        return max_duration
    
    def is_valid(self) -> bool:
        """检查任务配置是否有效"""
        if not self.segments:
            return False
        
        # 检查每个段是否有效
        for segment in self.segments:
            if not segment.bandwidth_duration_map:
                return False
            
            # 检查带宽和时间值是否合理
            for bandwidth, duration in segment.bandwidth_duration_map.items():
                if bandwidth <= 0 or duration <= 0:
                    return False
        
        # 检查性能需求是否合理
        if self.fps_requirement <= 0 or self.latency_requirement <= 0:
            return False
        
        return True
    
    def get_resource_requirements(self) -> Set[ResourceType]:
        """获取任务需要的资源类型"""
        return {segment.resource_type for segment in self.segments}
    
    def clone(self) -> 'NNTask':
        """克隆任务"""
        cloned = NNTask(
            task_id=f"{self.task_id}_clone",
            name=f"{self.name}_clone",
            priority=self.priority,
            runtime_type=self.runtime_type,
            segmentation_strategy=self.segmentation_strategy
        )
        
        # 复制段信息
        cloned.segments = []
        for segment in self.segments:
            cloned_segment = ResourceSegment(
                resource_type=segment.resource_type,
                bandwidth_duration_map=segment.bandwidth_duration_map.copy(),
                start_time=segment.start_time,
                segment_id=segment.segment_id
            )
            cloned_segment.cut_points = segment.cut_points.copy()
            cloned.segments.append(cloned_segment)
        
        # 复制其他属性
        cloned.fps_requirement = self.fps_requirement
        cloned.latency_requirement = self.latency_requirement
        cloned.dependencies = self.dependencies.copy()
        cloned.preset_cut_configurations = {
            k: [config.copy() for config in v] 
            for k, v in self.preset_cut_configurations.items()
        }
        
        return cloned
    
    def reset_execution_state(self):
        """重置执行状态"""
        self.state = TaskState.PENDING
        self.last_execution_time = -float('inf')
        self.schedule_info = None
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.current_segmentation = {}
    
    def update_execution_stats(self, execution_time: float):
        """更新执行统计"""
        self.execution_count += 1
        self.total_execution_time += execution_time
        self.last_execution_time = execution_time
    
    def get_performance_summary(self) -> Dict[str, float]:
        """获取性能摘要"""
        avg_execution_time = 0.0
        if self.execution_count > 0:
            avg_execution_time = self.total_execution_time / self.execution_count
        
        return {
            'execution_count': self.execution_count,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': avg_execution_time,
            'total_segmentation_overhead': self.total_segmentation_overhead,
            'fps_requirement': self.fps_requirement,
            'latency_requirement': self.latency_requirement
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        segments_info = f"{len(self.segments)} segments"
        if self.is_segmented:
            total_sub_segments = sum(len(sub_segs) for sub_segs in self.current_segmentation.values())
            segments_info += f" ({total_sub_segments} sub-segments)"
        
        return (f"NNTask({self.task_id}, {self.name}, "
                f"priority={self.priority.name}, "
                f"runtime={self.runtime_type.value}, "
                f"{segments_info})")
    
    def __repr__(self) -> str:
        return self.__str__()


# 便捷的任务创建函数
def create_vision_task(task_id: str, name: str, priority: TaskPriority = TaskPriority.HIGH) -> NNTask:
    """创建视觉处理任务"""
    task = NNTask(task_id, name, priority=priority, runtime_type=RuntimeType.ACPU_RUNTIME)
    task.set_npu_only({2.0: 50, 4.0: 30, 8.0: 20}, "vision_processing")
    task.set_performance_requirements(fps=30, latency=33)
    return task


def create_control_task(task_id: str, name: str, priority: TaskPriority = TaskPriority.CRITICAL) -> NNTask:
    """创建控制任务"""
    task = NNTask(task_id, name, priority=priority, runtime_type=RuntimeType.DSP_RUNTIME)
    task.set_npu_only({4.0: 10, 8.0: 6}, "control_algorithm")
    task.set_performance_requirements(fps=100, latency=10)
    return task


def create_perception_task(task_id: str, name: str, priority: TaskPriority = TaskPriority.HIGH) -> NNTask:
    """创建感知任务"""
    task = NNTask(task_id, name, priority=priority, runtime_type=RuntimeType.DSP_RUNTIME)
    task.set_dsp_npu_sequence([
        (ResourceType.DSP, {8.0: 5}, 0, "preprocessing"),
        (ResourceType.NPU, {4.0: 20}, 5, "inference")
    ])
    task.set_performance_requirements(fps=20, latency=50)
    return task


def create_background_task(task_id: str, name: str, priority: TaskPriority = TaskPriority.LOW) -> NNTask:
    """创建后台任务"""
    task = NNTask(task_id, name, priority=priority, runtime_type=RuntimeType.ACPU_RUNTIME)
    task.set_npu_only({2.0: 100, 4.0: 60, 8.0: 40}, "background_processing")
    task.set_performance_requirements(fps=5, latency=200)
    return task


if __name__ == "__main__":
    # 测试任务类
    print("=== NNTask 测试 ===")
    
    # 创建基础任务
    task = NNTask("T1", "TestTask", priority=TaskPriority.HIGH)
    task.set_npu_only({4.0: 20}, "test_segment")
    task.set_performance_requirements(fps=30, latency=50)
    
    print(f"任务: {task}")
    print(f"有效性: {task.is_valid()}")
    print(f"资源需求: {task.get_resource_requirements()}")
    
    # 测试分段功能
    print(f"\n=== 分段测试 ===")
    task.add_cut_points_to_segment("test_segment", [
        ("op1", 0.3, 0.15),
        ("op2", 0.7, 0.12)
    ])
    
    task.set_preset_cut_configurations("test_segment", [
        [],              # 无分段
        ["op1"],         # 单点分段
        ["op1", "op2"]   # 双点分段
    ])
    
    # 应用分段配置
    task.select_cut_configuration("test_segment", 2)  # 选择双点分段
    
    print(f"是否分段: {task.is_segmented}")
    print(f"子段数量: {len(task.get_sub_segments_for_scheduling())}")
    
    # 测试便捷创建函数
    print(f"\n=== 便捷创建函数测试 ===")
    
    vision_task = create_vision_task("V1", "VisionProcessing")
    control_task = create_control_task("C1", "ControlSystem")
    perception_task = create_perception_task("P1", "SensorFusion")
    
    tasks = [vision_task, control_task, perception_task]
    
    for task in tasks:
        print(f"{task.task_id}: {task.priority.name}, {len(task.segments)} segments")
    
    print("✅ 任务类测试完成")
