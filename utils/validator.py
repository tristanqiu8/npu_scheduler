#!/usr/bin/env python3
"""
Schedule Validator
调度验证器，用于检查调度结果的正确性
"""

from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from core.enums import ResourceType, TaskPriority


@dataclass
class ValidationError:
    """验证错误信息"""
    error_type: str
    message: str
    task_ids: List[str]
    resource_id: str = None
    time_range: Tuple[float, float] = None
    
    def __str__(self):
        return f"[{self.error_type}] {self.message}"


class ScheduleValidator:
    """调度验证器"""
    
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.errors = []
    
    def validate_all(self) -> Tuple[bool, List[ValidationError]]:
        """执行所有验证检查"""
        self.errors = []
        
        # 基础验证
        self._validate_resource_conflicts()
        self._validate_task_requirements()
        self._validate_resource_bindings()
        self._validate_priority_order()
        
        # 性能验证
        self._validate_fps_requirements()
        self._validate_latency_requirements()
        
        # 高级验证
        self._validate_segmentation_correctness()
        self._validate_resource_utilization()
        
        return len(self.errors) == 0, self.errors
    
    def _validate_resource_conflicts(self):
        """验证资源冲突"""
        # 按资源分组检查时间重叠
        resource_schedules = {}
        
        for schedule in self.scheduler.schedule_history:
            for res_type, res_id in schedule.assigned_resources.items():
                if res_id not in resource_schedules:
                    resource_schedules[res_id] = []
                
                # 计算实际执行时间
                task = self.scheduler.tasks[schedule.task_id]
                for seg in task.segments:
                    if seg.resource_type == res_type:
                        start_time = schedule.start_time + seg.start_time
                        resource_unit = next((r for r in self.scheduler.resources[res_type] 
                                            if r.unit_id == res_id), None)
                        if resource_unit:
                            duration = seg.get_duration(resource_unit.bandwidth)
                            end_time = start_time + duration
                            
                            resource_schedules[res_id].append({
                                'task_id': schedule.task_id,
                                'start': start_time,
                                'end': end_time,
                                'schedule': schedule
                            })
        
        # 检查每个资源的时间冲突
        for res_id, schedules in resource_schedules.items():
            schedules.sort(key=lambda x: x['start'])
            
            for i in range(len(schedules) - 1):
                current = schedules[i]
                next_task = schedules[i + 1]
                
                if current['end'] > next_task['start']:
                    # 发现时间重叠
                    overlap_start = next_task['start']
                    overlap_end = min(current['end'], next_task['end'])
                    
                    self.errors.append(ValidationError(
                        error_type="RESOURCE_CONFLICT",
                        message=f"资源 {res_id} 上任务时间重叠: {current['task_id']} 和 {next_task['task_id']}",
                        task_ids=[current['task_id'], next_task['task_id']],
                        resource_id=res_id,
                        time_range=(overlap_start, overlap_end)
                    ))
    
    def _validate_task_requirements(self):
        """验证任务基本需求"""
        for schedule in self.scheduler.schedule_history:
            task = self.scheduler.tasks[schedule.task_id]
            
            # 检查任务是否分配了足够的资源
            required_resources = set()
            for seg in task.segments:
                required_resources.add(seg.resource_type)
            
            assigned_resources = set(schedule.assigned_resources.keys())
            
            if not required_resources.issubset(assigned_resources):
                missing_resources = required_resources - assigned_resources
                self.errors.append(ValidationError(
                    error_type="MISSING_RESOURCES",
                    message=f"任务 {task.task_id} 缺少必需的资源: {[r.value for r in missing_resources]}",
                    task_ids=[task.task_id]
                ))
    
    def _validate_resource_bindings(self):
        """验证资源绑定正确性"""
        if not hasattr(self.scheduler, 'active_bindings'):
            return
        
        for binding in self.scheduler.active_bindings:
            # 检查绑定期间是否有任务正在运行
            for schedule in self.scheduler.schedule_history:
                task = self.scheduler.tasks[schedule.task_id]
                
                # 检查是否有任务在绑定期间使用绑定的资源
                if (schedule.start_time < binding.binding_end and 
                    schedule.end_time > binding.binding_start):
                    
                    for res_type, res_id in schedule.assigned_resources.items():
                        if res_id in binding.bound_resources:
                            # 验证运行时类型
                            from core.enums import RuntimeType
                            if task.runtime_type != RuntimeType.DSP_RUNTIME:
                                self.errors.append(ValidationError(
                                    error_type="INVALID_BINDING",
                                    message=f"非DSP运行时任务 {task.task_id} 不应使用绑定资源 {res_id}",
                                    task_ids=[task.task_id],
                                    resource_id=res_id,
                                    time_range=(binding.binding_start, binding.binding_end)
                                ))
    
    def _validate_priority_order(self):
        """验证优先级顺序"""
        # 按开始时间排序检查优先级
        sorted_schedules = sorted(self.scheduler.schedule_history, key=lambda s: s.start_time)
        
        for i in range(len(sorted_schedules) - 1):
            current_schedule = sorted_schedules[i]
            next_schedule = sorted_schedules[i + 1]
            
            current_task = self.scheduler.tasks[current_schedule.task_id]
            next_task = self.scheduler.tasks[next_schedule.task_id]
            
            # 检查是否有低优先级任务在高优先级任务之前开始
            if (current_task.priority.value > next_task.priority.value and 
                current_schedule.start_time < next_schedule.start_time):
                
                # 进一步检查是否有资源冲突
                current_resources = set(current_schedule.assigned_resources.values())
                next_resources = set(next_schedule.assigned_resources.values())
                
                if current_resources & next_resources:  # 有共同资源
                    self.errors.append(ValidationError(
                        error_type="PRIORITY_VIOLATION",
                        message=f"低优先级任务 {current_task.task_id}({current_task.priority.name}) "
                               f"在高优先级任务 {next_task.task_id}({next_task.priority.name}) 之前执行",
                        task_ids=[current_task.task_id, next_task.task_id]
                    ))
    
    def _validate_fps_requirements(self):
        """验证FPS需求"""
        # 按任务分组计算实际FPS
        task_schedules = {}
        for schedule in self.scheduler.schedule_history:
            task_id = schedule.task_id
            if task_id not in task_schedules:
                task_schedules[task_id] = []
            task_schedules[task_id].append(schedule)
        
        for task_id, schedules in task_schedules.items():
            task = self.scheduler.tasks[task_id]
            if hasattr(task, 'fps_requirement') and task.fps_requirement:
                # 计算实际执行间隔
                if len(schedules) > 1:
                    schedules.sort(key=lambda s: s.start_time)
                    intervals = []
                    for i in range(1, len(schedules)):
                        interval = schedules[i].start_time - schedules[i-1].start_time
                        intervals.append(interval)
                    
                    avg_interval = sum(intervals) / len(intervals)
                    actual_fps = 1000.0 / avg_interval  # 转换为FPS
                    
                    if actual_fps < task.fps_requirement * 0.9:  # 允许10%误差
                        self.errors.append(ValidationError(
                            error_type="FPS_VIOLATION",
                            message=f"任务 {task_id} FPS不足: 需要 {task.fps_requirement}, 实际 {actual_fps:.1f}",
                            task_ids=[task_id]
                        ))
    
    def _validate_latency_requirements(self):
        """验证延迟需求"""
        for schedule in self.scheduler.schedule_history:
            task = self.scheduler.tasks[schedule.task_id]
            if hasattr(task, 'latency_requirement') and task.latency_requirement:
                actual_latency = schedule.end_time - schedule.start_time
                
                if actual_latency > task.latency_requirement:
                    self.errors.append(ValidationError(
                        error_type="LATENCY_VIOLATION",
                        message=f"任务 {task.task_id} 延迟超标: 需要 ≤{task.latency_requirement}ms, 实际 {actual_latency:.1f}ms",
                        task_ids=[task.task_id]
                    ))
    
    def _validate_segmentation_correctness(self):
        """验证分段正确性"""
        for schedule in self.scheduler.schedule_history:
            task = self.scheduler.tasks[schedule.task_id]
            
            if task.is_segmented and hasattr(schedule, 'sub_segment_schedule'):
                if not schedule.sub_segment_schedule:
                    self.errors.append(ValidationError(
                        error_type="SEGMENTATION_ERROR",
                        message=f"分段任务 {task.task_id} 没有子段调度信息",
                        task_ids=[task.task_id]
                    ))
                else:
                    # 验证子段时间连续性
                    sorted_segments = sorted(schedule.sub_segment_schedule, key=lambda x: x[1])
                    for i in range(len(sorted_segments) - 1):
                        current_end = sorted_segments[i][2]
                        next_start = sorted_segments[i + 1][1]
                        
                        if next_start < current_end:
                            self.errors.append(ValidationError(
                                error_type="SEGMENT_OVERLAP",
                                message=f"任务 {task.task_id} 的子段时间重叠",
                                task_ids=[task.task_id]
                            ))
    
    def _validate_resource_utilization(self):
        """验证资源利用率"""
        if not self.scheduler.schedule_history:
            return
        
        total_time = max(s.end_time for s in self.scheduler.schedule_history)
        
        # 计算每个资源的利用率
        resource_usage = {}
        for schedule in self.scheduler.schedule_history:
            for res_type, res_id in schedule.assigned_resources.items():
                if res_id not in resource_usage:
                    resource_usage[res_id] = 0
                resource_usage[res_id] += schedule.end_time - schedule.start_time
        
        # 检查利用率异常
        for res_id, usage in resource_usage.items():
            utilization = (usage / total_time) * 100
            
            if utilization > 100:  # 理论上不应该超过100%
                self.errors.append(ValidationError(
                    error_type="OVERUTILIZATION",
                    message=f"资源 {res_id} 利用率超过100%: {utilization:.1f}%",
                    task_ids=[],
                    resource_id=res_id
                ))
    
    def print_validation_report(self):
        """打印验证报告"""
        if not self.errors:
            print("✅ 调度验证通过，没有发现错误")
            return
        
        print(f"❌ 调度验证失败，发现 {len(self.errors)} 个错误:\n")
        
        # 按错误类型分组
        error_groups = {}
        for error in self.errors:
            error_type = error.error_type
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(error)
        
        for error_type, errors in error_groups.items():
            print(f"📋 {error_type} ({len(errors)} 个错误):")
            for i, error in enumerate(errors, 1):
                print(f"   {i}. {error.message}")
                if error.task_ids:
                    print(f"      涉及任务: {', '.join(error.task_ids)}")
                if error.resource_id:
                    print(f"      涉及资源: {error.resource_id}")
                if error.time_range:
                    print(f"      时间范围: {error.time_range[0]:.1f} - {error.time_range[1]:.1f}ms")
            print()


def validate_schedule(scheduler, verbose=True) -> Tuple[bool, List[ValidationError]]:
    """便捷的调度验证函数"""
    validator = ScheduleValidator(scheduler)
    is_valid, errors = validator.validate_all()
    
    if verbose:
        validator.print_validation_report()
    
    return is_valid, errors


def quick_check(scheduler) -> bool:
    """快速检查调度是否有基本错误"""
    validator = ScheduleValidator(scheduler)
    validator._validate_resource_conflicts()
    validator._validate_task_requirements()
    
    return len(validator.errors) == 0


if __name__ == "__main__":
    # 测试验证器功能
    print("=== 调度验证器测试 ===")
    print("请通过主程序运行来测试验证功能")
