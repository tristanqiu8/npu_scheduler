#!/usr/bin/env python3
"""
Multi-Resource Scheduler
多资源调度器 - NPU调度器的核心调度算法实现
"""

import time
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import numpy as np

from .enums import ResourceType, TaskPriority, RuntimeType, ResourceState
from .models import (
    ResourceUnit, TaskScheduleInfo, ResourceBinding, 
    PerformanceMetrics, SubSegment
)


class ResourcePriorityQueue:
    """单个资源的优先级队列"""
    
    def __init__(self, resource_id: str):
        self.resource_id = resource_id
        self.available_time = 0.0
        
        # 优先级队列
        self.queues = {priority: [] for priority in TaskPriority}
        
        # 资源绑定状态
        self.bound_until = 0.0
        self.binding_task_id = None
        
        # 子段预约
        self.sub_segment_reservations = []  # [(sub_seg_id, start_time, end_time)]
    
    def add_task(self, task, current_time: float):
        """添加任务到优先级队列"""
        self.queues[task.priority].append((task, current_time))
    
    def get_next_task(self, current_time: float):
        """获取下一个要执行的任务"""
        for priority in TaskPriority:
            queue = self.queues[priority]
            if queue:
                # 获取等待时间最长的任务
                return queue.pop(0)
        return None
    
    def is_available(self, current_time: float) -> bool:
        """检查资源是否可用"""
        return (self.available_time <= current_time and 
                self.bound_until <= current_time)
    
    def reserve(self, task_id: str, start_time: float, duration: float):
        """预订资源"""
        self.available_time = start_time + duration
    
    def bind_resource(self, task_id: str, start_time: float, duration: float):
        """绑定资源（DSP_Runtime使用）"""
        self.binding_task_id = task_id
        self.bound_until = start_time + duration
        self.available_time = max(self.available_time, start_time + duration)
    
    def release_binding(self):
        """释放资源绑定"""
        self.binding_task_id = None
        self.bound_until = 0.0


class MultiResourceScheduler:
    """多资源调度器主类"""
    
    def __init__(self, enable_segmentation: bool = False):
        # 资源管理
        self.resources: Dict[ResourceType, List[ResourceUnit]] = {
            ResourceType.NPU: [],
            ResourceType.DSP: []
        }
        
        # 任务管理
        self.tasks: Dict[str, 'NNTask'] = {}
        
        # 调度状态
        self.schedule_history: List[TaskScheduleInfo] = []
        self.active_bindings: List[ResourceBinding] = []
        
        # 优先级队列
        self.priority_queues: Dict[str, ResourcePriorityQueue] = {}
        
        # 功能开关
        self.enable_segmentation = enable_segmentation
        
        # 性能统计
        self.segmentation_stats = {
            'segmented_tasks': 0,
            'total_overhead': 0.0,
            'average_benefit': 0.0
        }
        
        # 初始化默认资源
        self._initialize_default_resources()
    
    def _initialize_default_resources(self):
        """初始化默认资源配置"""
        # 添加NPU资源
        for i in range(4):
            bandwidth = [2.0, 4.0, 4.0, 8.0][i]  # 不同性能的NPU
            npu = ResourceUnit(f"NPU_{i}", ResourceType.NPU, bandwidth=bandwidth)
            self.add_resource(npu)
        
        # 添加DSP资源
        for i in range(2):
            bandwidth = [4.0, 8.0][i]
            dsp = ResourceUnit(f"DSP_{i}", ResourceType.DSP, bandwidth=bandwidth)
            self.add_resource(dsp)
    
    def add_resource(self, resource: ResourceUnit):
        """添加资源"""
        self.resources[resource.resource_type].append(resource)
        
        # 创建对应的优先级队列
        self.priority_queues[resource.unit_id] = ResourcePriorityQueue(resource.unit_id)
    
    def add_task(self, task: 'NNTask'):
        """添加任务"""
        self.tasks[task.task_id] = task
        
        # 设置任务的性能需求默认值
        if not hasattr(task, 'fps_requirement'):
            task.fps_requirement = 10  # 默认10 FPS
        if not hasattr(task, 'latency_requirement'):
            task.latency_requirement = 100  # 默认100ms延迟
        if not hasattr(task, 'min_interval_ms'):
            task.min_interval_ms = 1000.0 / task.fps_requirement if task.fps_requirement > 0 else 100.0
        if not hasattr(task, 'dependencies'):
            task.dependencies = []
    
    def priority_aware_schedule_with_segmentation(self, time_window: float = 1000.0) -> List[TaskScheduleInfo]:
        """优先级感知的分段调度算法"""
        
        # 重置调度状态
        self._reset_scheduling_state()
        
        # 任务执行计数（用于FPS计算）
        task_execution_counts = defaultdict(int)
        current_time = 0.0
        
        print(f"🚀 开始调度算法，时间窗口: {time_window}ms")
        
        while current_time < time_window:
            # 清理过期的绑定
            self._cleanup_expired_bindings(current_time)
            
            # 找到所有准备就绪的任务
            ready_tasks = self._find_ready_tasks(current_time, task_execution_counts)
            
            if not ready_tasks:
                # 没有就绪任务，跳到下一个事件时间
                next_time = self._find_next_event_time(current_time, time_window)
                if next_time > current_time:
                    current_time = next_time
                else:
                    current_time += 1.0  # 防止死循环
                continue
            
            # 调度就绪任务
            scheduled_any = False
            
            for task in ready_tasks:
                if self._schedule_single_task(task, current_time):
                    task_execution_counts[task.task_id] += 1
                    task.last_execution_time = current_time
                    scheduled_any = True
                    
                    if hasattr(self, 'verbose') and self.verbose:
                        print(f"   ✅ 调度任务 {task.task_id} 在时间 {current_time:.1f}ms")
                    
                    break  # 每次只调度一个任务
            
            if not scheduled_any:
                current_time += 1.0  # 推进时间
        
        print(f"✅ 调度完成，共调度 {len(self.schedule_history)} 个任务")
        
        return self.schedule_history
    
    def _reset_scheduling_state(self):
        """重置调度状态"""
        # 重置资源队列
        for queue in self.priority_queues.values():
            queue.available_time = 0.0
            queue.release_binding()
            for priority_queue in queue.queues.values():
                priority_queue.clear()
        
        # 重置任务状态
        for task in self.tasks.values():
            task.last_execution_time = -float('inf')
            if hasattr(task, 'schedule_info'):
                task.schedule_info = None
        
        # 清空历史记录
        self.schedule_history.clear()
        self.active_bindings.clear()
        
        # 重置统计信息
        self.segmentation_stats = {
            'segmented_tasks': 0,
            'total_overhead': 0.0,
            'average_benefit': 0.0
        }
    
    def _cleanup_expired_bindings(self, current_time: float):
        """清理过期的资源绑定"""
        self.active_bindings = [
            binding for binding in self.active_bindings
            if binding.binding_end > current_time
        ]
    
    def _find_ready_tasks(self, current_time: float, task_execution_counts: Dict[str, int]) -> List['NNTask']:
        """找到准备就绪的任务"""
        ready_tasks = []
        
        for task in self.tasks.values():
            # 检查最小间隔
            if task.last_execution_time + task.min_interval_ms > current_time:
                continue
            
            # 检查依赖关系
            deps_satisfied = True
            for dep_id in task.dependencies:
                if dep_id in self.tasks:
                    if task_execution_counts.get(dep_id, 0) <= task_execution_counts.get(task.task_id, 0):
                        deps_satisfied = False
                        break
            
            if deps_satisfied:
                ready_tasks.append(task)
        
        # 按优先级排序（优先级值越小，优先级越高）
        ready_tasks.sort(key=lambda t: (t.priority.value, t.task_id))
        
        return ready_tasks
    
    def _find_next_event_time(self, current_time: float, time_window: float) -> float:
        """找到下一个事件发生的时间"""
        next_time = time_window
        
        # 检查资源可用时间
        for queue in self.priority_queues.values():
            if queue.available_time > current_time:
                next_time = min(next_time, queue.available_time)
            if queue.bound_until > current_time:
                next_time = min(next_time, queue.bound_until)
        
        # 检查任务准备时间
        for task in self.tasks.values():
            if task.last_execution_time > -float('inf'):
                next_ready = task.last_execution_time + task.min_interval_ms
                if next_ready > current_time:
                    next_time = min(next_time, next_ready)
        
        return next_time
    
    def _schedule_single_task(self, task: 'NNTask', current_time: float) -> bool:
        """调度单个任务"""
        try:
            # 根据运行时类型选择调度策略
            if task.runtime_type == RuntimeType.DSP_RUNTIME:
                return self._schedule_dsp_runtime_task(task, current_time)
            else:
                return self._schedule_acpu_runtime_task(task, current_time)
        
        except Exception as e:
            if hasattr(self, 'verbose') and self.verbose:
                print(f"⚠️  调度任务 {task.task_id} 失败: {e}")
            return False
    
    def _schedule_dsp_runtime_task(self, task: 'NNTask', current_time: float) -> bool:
        """调度DSP运行时任务（资源绑定模式）"""
        
        # 找到所需的资源
        required_resources = []
        for segment in task.segments:
            resources_of_type = [r for r in self.resources[segment.resource_type] 
                               if self.priority_queues[r.unit_id].is_available(current_time)]
            
            if not resources_of_type:
                return False  # 没有可用资源
            
            # 选择最高带宽的资源
            best_resource = max(resources_of_type, key=lambda r: r.bandwidth)
            required_resources.append((segment, best_resource))
        
        # 计算执行时间和绑定时间
        max_end_time = current_time
        resource_assignments = {}
        
        for segment, resource in required_resources:
            duration = segment.get_duration(resource.bandwidth)
            segment_start = current_time + segment.start_time
            segment_end = segment_start + duration
            
            max_end_time = max(max_end_time, segment_end)
            resource_assignments[segment.resource_type] = resource.unit_id
        
        # 绑定所有资源
        binding_duration = max_end_time - current_time
        bound_resource_ids = set(resource_assignments.values())
        
        # 创建资源绑定
        binding = ResourceBinding(
            binding_id=f"binding_{task.task_id}_{len(self.active_bindings)}",
            bound_resources=bound_resource_ids,
            binding_start=current_time,
            binding_end=max_end_time,
            task_id=task.task_id,
            runtime_type=RuntimeType.DSP_RUNTIME
        )
        
        self.active_bindings.append(binding)
        
        # 更新资源状态
        for resource_id in bound_resource_ids:
            queue = self.priority_queues[resource_id]
            queue.bind_resource(task.task_id, current_time, binding_duration)
        
        # 创建调度信息
        schedule_info = TaskScheduleInfo(
            task_id=task.task_id,
            start_time=current_time,
            end_time=max_end_time,
            assigned_resources=resource_assignments
        )
        
        self.schedule_history.append(schedule_info)
        task.schedule_info = schedule_info
        
        return True
    
    def _schedule_acpu_runtime_task(self, task: 'NNTask', current_time: float) -> bool:
        """调度ACPU运行时任务（流水线模式）"""
        
        # 为每个段找到可用资源
        resource_assignments = {}
        segment_schedules = []
        
        for segment in task.segments:
            # 找到该类型的可用资源
            available_resources = [
                r for r in self.resources[segment.resource_type]
                if self.priority_queues[r.unit_id].is_available(current_time + segment.start_time)
            ]
            
            if not available_resources:
                return False  # 没有可用资源
            
            # 选择最早可用的资源
            best_resource = min(available_resources, 
                              key=lambda r: self.priority_queues[r.unit_id].available_time)
            
            # 计算实际开始时间
            queue = self.priority_queues[best_resource.unit_id]
            actual_start = max(current_time + segment.start_time, queue.available_time)
            duration = segment.get_duration(best_resource.bandwidth)
            actual_end = actual_start + duration
            
            # 预订资源
            queue.reserve(task.task_id, actual_start, duration)
            
            resource_assignments[segment.resource_type] = best_resource.unit_id
            segment_schedules.append((segment, actual_start, actual_end))
        
        # 计算总的开始和结束时间
        start_time = min(start for _, start, _ in segment_schedules)
        end_time = max(end for _, _, end in segment_schedules)
        
        # 创建调度信息
        schedule_info = TaskScheduleInfo(
            task_id=task.task_id,
            start_time=start_time,
            end_time=end_time,
            assigned_resources=resource_assignments
        )
        
        self.schedule_history.append(schedule_info)
        task.schedule_info = schedule_info
        
        return True
    
    def get_resource_utilization(self, total_time: float) -> Dict[str, float]:
        """计算资源利用率"""
        utilization = {}
        
        for resource_type, resources in self.resources.items():
            for resource in resources:
                usage_time = 0.0
                
                # 计算该资源的总使用时间
                for schedule in self.schedule_history:
                    if resource.unit_id in schedule.assigned_resources.values():
                        usage_time += schedule.end_time - schedule.start_time
                
                # 计算利用率百分比
                if total_time > 0:
                    utilization[resource.unit_id] = (usage_time / total_time) * 100.0
                else:
                    utilization[resource.unit_id] = 0.0
        
        return utilization
    
    def get_performance_metrics(self, time_window: float) -> PerformanceMetrics:
        """获取性能指标"""
        metrics = PerformanceMetrics()
        
        if not self.schedule_history:
            return metrics
        
        # 基础指标
        metrics.total_tasks = len(self.schedule_history)
        metrics.makespan = max(s.end_time for s in self.schedule_history)
        
        latencies = [s.end_time - s.start_time for s in self.schedule_history]
        metrics.average_latency = sum(latencies) / len(latencies)
        
        # 资源利用率
        metrics.resource_utilization = self.get_resource_utilization(metrics.makespan)
        if metrics.resource_utilization:
            metrics.average_utilization = sum(metrics.resource_utilization.values()) / len(metrics.resource_utilization)
        
        # 计算违规数量
        task_counts = defaultdict(int)
        for schedule in self.schedule_history:
            task_counts[schedule.task_id] += 1
        
        for task_id, task in self.tasks.items():
            # FPS违规检查
            if hasattr(task, 'fps_requirement') and task.fps_requirement > 0:
                count = task_counts.get(task_id, 0)
                achieved_fps = count / (time_window / 1000.0)
                if achieved_fps < task.fps_requirement * 0.95:  # 5%容忍度
                    metrics.fps_violations += 1
            
            # 延迟违规检查
            if hasattr(task, 'latency_requirement') and task.latency_requirement > 0:
                for schedule in self.schedule_history:
                    if schedule.task_id == task_id:
                        if schedule.end_time - schedule.start_time > task.latency_requirement:
                            metrics.latency_violations += 1
                            break
        
        # 优先级分布
        for task in self.tasks.values():
            priority = task.priority
            metrics.priority_distribution[priority] = metrics.priority_distribution.get(priority, 0) + 1
        
        # 分段统计
        metrics.segmented_tasks = self.segmentation_stats['segmented_tasks']
        metrics.total_segmentation_overhead = self.segmentation_stats['total_overhead']
        metrics.average_segmentation_benefit = self.segmentation_stats['average_benefit']
        
        return metrics
    
    def print_schedule_summary(self):
        """打印调度摘要"""
        if not self.schedule_history:
            print("❌ 没有调度结果")
            return
        
        print(f"\n📊 调度摘要:")
        print(f"  总任务数: {len(self.schedule_history)}")
        print(f"  总完成时间: {max(s.end_time for s in self.schedule_history):.1f}ms")
        
        # 按优先级统计
        priority_counts = defaultdict(int)
        for schedule in self.schedule_history:
            task = self.tasks[schedule.task_id]
            priority_counts[task.priority.name] += 1
        
        print(f"  优先级分布:")
        for priority, count in priority_counts.items():
            print(f"    {priority}: {count} 个任务")
        
        # 资源利用率
        total_time = max(s.end_time for s in self.schedule_history)
        utilization = self.get_resource_utilization(total_time)
        avg_utilization = sum(utilization.values()) / len(utilization) if utilization else 0
        
        print(f"  平均资源利用率: {avg_utilization:.1f}%")


if __name__ == "__main__":
    # 简单测试
    print("=== MultiResourceScheduler 测试 ===")
    
    scheduler = MultiResourceScheduler()
    
    print(f"NPU 资源数: {len(scheduler.resources[ResourceType.NPU])}")
    print(f"DSP 资源数: {len(scheduler.resources[ResourceType.DSP])}")
    print(f"优先级队列数: {len(scheduler.priority_queues)}")
    
    print("✅ 调度器初始化成功")
