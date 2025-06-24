#!/usr/bin/env python3
"""
精度调度补丁 - 将成功的精度调度方案应用到现有系统
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict, Optional, Tuple
from decimal import Decimal, ROUND_UP
from dataclasses import dataclass
from enums import ResourceType
from models import TaskScheduleInfo

# 常量定义
TIMING_PRECISION = Decimal('0.1')  # 0.1ms精度
SAFETY_BUFFER = Decimal('0.1')     # 0.1ms安全缓冲

@dataclass
class PrecisionScheduleEvent:
    """精度调度事件"""
    task_id: str
    sub_segment_id: str
    resource_id: str
    resource_type: ResourceType
    start_time: Decimal
    end_time: Decimal
    

class PrecisionTimeManager:
    """精度时间管理器"""
    
    def __init__(self):
        self.scheduled_events: List[PrecisionScheduleEvent] = []
        
    def to_decimal(self, value: float) -> Decimal:
        """转换为Decimal并量化"""
        return Decimal(str(value)).quantize(TIMING_PRECISION, rounding=ROUND_UP)
    
    def check_resource_available(self, resource_id: str, start: Decimal, end: Decimal) -> bool:
        """检查资源在指定时间段是否可用"""
        for event in self.scheduled_events:
            if event.resource_id == resource_id:
                # 检查时间重叠（包含安全缓冲）
                if not (end + SAFETY_BUFFER <= event.start_time or 
                       start >= event.end_time + SAFETY_BUFFER):
                    return False
        return True
    
    def find_next_available_time(self, resource_id: str, duration: Decimal, 
                                earliest_start: Decimal) -> Decimal:
        """找到资源的下一个可用时间"""
        # 获取该资源的所有事件，按时间排序
        resource_events = sorted(
            [e for e in self.scheduled_events if e.resource_id == resource_id],
            key=lambda e: e.start_time
        )
        
        if not resource_events:
            return earliest_start
        
        current_time = earliest_start
        
        for event in resource_events:
            # 检查当前时间段是否可以容纳任务
            if current_time + duration + SAFETY_BUFFER <= event.start_time:
                return current_time
            
            # 移动到事件结束后
            current_time = event.end_time + SAFETY_BUFFER
        
        return current_time
    
    def reserve_time_slot(self, task_id: str, sub_segment_id: str, 
                         resource_id: str, resource_type: ResourceType,
                         duration: Decimal, earliest_start: Decimal) -> Optional[Tuple[Decimal, Decimal]]:
        """预留时间槽"""
        # 找到可用时间
        start_time = self.find_next_available_time(resource_id, duration, earliest_start)
        end_time = start_time + duration
        
        # 再次验证（双重保险）
        if not self.check_resource_available(resource_id, start_time, end_time):
            return None
        
        # 创建事件记录
        event = PrecisionScheduleEvent(
            task_id=task_id,
            sub_segment_id=sub_segment_id,
            resource_id=resource_id,
            resource_type=resource_type,
            start_time=start_time,
            end_time=end_time
        )
        
        self.scheduled_events.append(event)
        return (start_time, end_time)
    
    def clear_events_after(self, time: float):
        """清除指定时间后的所有事件（用于重新调度）"""
        time_dec = self.to_decimal(time)
        self.scheduled_events = [e for e in self.scheduled_events if e.start_time < time_dec]


def apply_precision_scheduling_patch(scheduler):
    """应用精度调度补丁到现有调度器"""
    
    print("✅ Applying precision scheduling patch...")
    
    # 添加精度时间管理器
    scheduler._precision_time_manager = PrecisionTimeManager()
    
    # 保存原始方法
    original_schedule_method = scheduler.priority_aware_schedule_with_segmentation
    
    def precision_aware_schedule_with_segmentation(time_window: float = 1000.0):
        """使用精度时间管理的调度方法"""
        
        # 清理状态
        scheduler._precision_time_manager.scheduled_events.clear()
        scheduler.schedule_history.clear()
        scheduler.active_bindings.clear()
        
        # 重置资源队列
        for queue in scheduler.resource_queues.values():
            queue.available_time = 0.0
            queue.release_binding()
            
        # 重置任务状态
        for task in scheduler.tasks.values():
            task.schedule_info = None
            task.last_execution_time = -float('inf')
            task.ready_time = 0
            
        current_time = 0.0
        scheduled_events = []
        task_execution_count = {task_id: 0 for task_id in scheduler.tasks}
        
        while current_time < time_window:
            # 清理过期的绑定
            scheduler.cleanup_expired_bindings(current_time)
            
            # 找到就绪的任务
            ready_tasks = []
            for task in scheduler.tasks.values():
                # 检查最小间隔
                if task.last_execution_time + task.min_interval_ms > current_time + 0.01:
                    continue
                
                # 检查依赖
                deps_satisfied = True
                for dep_id in task.dependencies:
                    if dep_id in scheduler.tasks:
                        if task_execution_count[dep_id] <= task_execution_count[task.task_id]:
                            deps_satisfied = False
                            break
                
                if deps_satisfied:
                    ready_tasks.append(task)
            
            # 按优先级排序
            ready_tasks.sort(key=lambda t: (t.priority.value, t.task_id))
            
            scheduled_any = False
            
            for task in ready_tasks:
                # 获取任务的子段
                sub_segments = task.get_sub_segments_for_scheduling()
                
                if not sub_segments:
                    continue
                
                # 尝试调度所有子段
                segment_schedule = []
                all_scheduled = True
                start_time_dec = scheduler._precision_time_manager.to_decimal(current_time)
                
                for sub_seg in sub_segments:
                    # 找到可用资源
                    best_resource = None
                    best_time = None
                    
                    for resource in scheduler.resources.get(sub_seg.resource_type, []):
                        # 计算持续时间
                        duration = sub_seg.get_duration(resource.bandwidth)
                        duration_dec = scheduler._precision_time_manager.to_decimal(duration)
                        
                        # 计算最早开始时间
                        earliest = start_time_dec + scheduler._precision_time_manager.to_decimal(sub_seg.start_time)
                        
                        # 尝试预留时间槽
                        time_slot = scheduler._precision_time_manager.reserve_time_slot(
                            task.task_id,
                            sub_seg.sub_id,
                            resource.unit_id,
                            sub_seg.resource_type,
                            duration_dec,
                            earliest
                        )
                        
                        if time_slot:
                            if best_time is None or time_slot[0] < best_time[0]:
                                # 如果之前有更好的选择，需要撤销
                                if best_resource:
                                    scheduler._precision_time_manager.scheduled_events.pop()
                                
                                best_resource = resource
                                best_time = time_slot
                            else:
                                # 撤销这次预留
                                scheduler._precision_time_manager.scheduled_events.pop()
                    
                    if best_resource:
                        segment_schedule.append({
                            'sub_segment': sub_seg,
                            'resource': best_resource,
                            'start_time': float(best_time[0]),
                            'end_time': float(best_time[1])
                        })
                    else:
                        all_scheduled = False
                        # 撤销已经调度的段
                        for _ in segment_schedule:
                            scheduler._precision_time_manager.scheduled_events.pop()
                        break
                
                if all_scheduled and segment_schedule:
                    # 创建调度信息
                    task_start = min(s['start_time'] for s in segment_schedule)
                    task_end = max(s['end_time'] for s in segment_schedule)
                    
                    # 确定分配的资源
                    assigned_resources = {}
                    for seg_info in segment_schedule:
                        res_type = seg_info['sub_segment'].resource_type
                        assigned_resources[res_type] = seg_info['resource'].unit_id
                    
                    # 创建子段调度信息
                    sub_segment_schedule = [
                        (seg_info['sub_segment'].sub_id, 
                         seg_info['start_time'], 
                         seg_info['end_time'])
                        for seg_info in segment_schedule
                    ]
                    
                    schedule_info = TaskScheduleInfo(
                        task_id=task.task_id,
                        start_time=task_start,
                        end_time=task_end,
                        assigned_resources=assigned_resources,
                        actual_latency=task_end - current_time,
                        runtime_type=task.runtime_type,
                        used_cuts=task.current_segmentation.copy(),
                        segmentation_overhead=task.total_segmentation_overhead,
                        sub_segment_schedule=sub_segment_schedule
                    )
                    
                    # 更新任务状态
                    task.schedule_info = schedule_info
                    task.last_execution_time = task_start
                    task_execution_count[task.task_id] += 1
                    
                    # 更新资源队列
                    for seg_info in segment_schedule:
                        queue = scheduler.resource_queues[seg_info['resource'].unit_id]
                        queue.available_time = seg_info['end_time']
                    
                    scheduler.schedule_history.append(schedule_info)
                    scheduled_events.append(schedule_info)
                    scheduled_any = True
                    
                    break  # 只调度一个任务，然后重新评估
            
            if not scheduled_any:
                # 推进时间到下一个可能的调度点
                next_time = current_time + 1.0
                
                # 检查资源可用时间
                for queue in scheduler.resource_queues.values():
                    if queue.available_time > current_time:
                        next_time = min(next_time, queue.available_time)
                
                # 检查任务就绪时间
                for task in scheduler.tasks.values():
                    if task.last_execution_time > -float('inf'):
                        ready_time = task.last_execution_time + task.min_interval_ms
                        if ready_time > current_time:
                            next_time = min(next_time, ready_time)
                
                current_time = min(next_time, time_window)
            else:
                current_time += 0.1  # 小步推进
        
        print(f"✅ Precision scheduling completed: {len(scheduled_events)} events scheduled")
        
        # 验证结果
        issues = validate_precision_schedule(scheduler)
        if issues:
            print(f"⚠️ Found {len(issues)} timing issues:")
            for issue in issues[:5]:  # 只显示前5个
                print(f"  - {issue}")
        else:
            print("✅ No timing conflicts detected!")
        
        return scheduled_events
    
    # 替换原方法
    scheduler.priority_aware_schedule_with_segmentation = precision_aware_schedule_with_segmentation
    
    print("✅ Precision scheduling patch applied successfully")
    print("  - Timing precision: 0.1ms")
    print("  - Safety buffer: 0.1ms")
    print("  - Conflict detection: Enhanced")


def validate_precision_schedule(scheduler) -> List[str]:
    """验证精度调度结果"""
    issues = []
    
    if not hasattr(scheduler, '_precision_time_manager'):
        return ["Precision time manager not found"]
    
    # 按资源分组事件
    by_resource = {}
    for event in scheduler._precision_time_manager.scheduled_events:
        if event.resource_id not in by_resource:
            by_resource[event.resource_id] = []
        by_resource[event.resource_id].append(event)
    
    # 检查每个资源的时间冲突
    for resource_id, events in by_resource.items():
        sorted_events = sorted(events, key=lambda e: e.start_time)
        
        for i in range(len(sorted_events) - 1):
            current = sorted_events[i]
            next_event = sorted_events[i + 1]
            
            if current.end_time > next_event.start_time:
                overlap = float(current.end_time - next_event.start_time)
                issues.append(
                    f"{resource_id}: {current.task_id}-{current.sub_segment_id} "
                    f"({float(current.start_time):.1f}-{float(current.end_time):.1f}) overlaps "
                    f"{next_event.task_id}-{next_event.sub_segment_id} "
                    f"({float(next_event.start_time):.1f}-{float(next_event.end_time):.1f}) "
                    f"by {overlap:.1f}ms"
                )
    
    return issues


if __name__ == "__main__":
    print("Precision Scheduler Patch - Ready to apply")
    print("Usage: from precision_scheduler_patch import apply_precision_scheduling_patch")
    print("       apply_precision_scheduling_patch(your_scheduler)")
