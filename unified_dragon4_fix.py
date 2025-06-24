#!/usr/bin/env python3
"""
统一的Dragon4系统修复方案
解决多个补丁之间的冲突，确保资源冲突彻底消除
"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from enums import ResourceType, TaskPriority, RuntimeType
from models import TaskScheduleInfo


def apply_unified_dragon4_fix(scheduler):
    """应用统一的Dragon4修复方案
    
    这个修复方案会替代所有其他补丁，确保没有资源冲突
    """
    print("🔧 应用统一Dragon4修复方案...")
    
    # 1. 保存原始方法
    original_schedule_method = scheduler.priority_aware_schedule_with_segmentation
    original_find_resources = scheduler.find_available_resources_for_task_with_segmentation
    
    # 2. 创建统一的时间管理器
    scheduler._unified_time_manager = UnifiedTimeManager()
    
    # 3. 替换资源查找方法
    def unified_find_available_resources(task, current_time):
        """统一的资源查找方法，确保无冲突"""
        return scheduler._unified_time_manager.find_resources_for_task(
            scheduler, task, current_time
        )
    
    # 4. 替换调度方法
    def unified_priority_schedule(time_window: float = 1000.0):
        """统一的优先级调度方法"""
        return scheduler._unified_time_manager.schedule_with_strict_timing(
            scheduler, time_window
        )
    
    # 5. 应用替换
    scheduler.find_available_resources_for_task_with_segmentation = unified_find_available_resources
    scheduler.priority_aware_schedule_with_segmentation = unified_priority_schedule
    
    print("✅ 统一Dragon4修复已应用")
    print("  - 严格时间管理")
    print("  - 零资源冲突保证")
    print("  - 优先级感知调度")


class UnifiedTimeManager:
    """统一时间管理器 - 确保零资源冲突"""
    
    def __init__(self):
        self.time_precision = Decimal('0.1')  # 0.1ms精度
        self.safety_buffer = Decimal('0.1')   # 0.1ms安全缓冲
        self.resource_timeline = defaultdict(list)  # 资源时间线
        
    def to_decimal(self, value: float) -> Decimal:
        """转换为高精度Decimal"""
        return Decimal(str(value)).quantize(self.time_precision, rounding=ROUND_HALF_UP)
    
    def find_resources_for_task(self, scheduler, task, current_time):
        """为任务查找可用资源，确保无冲突"""
        
        assigned_resources = {}
        current_time_dec = self.to_decimal(current_time)
        
        # 计算任务所需的资源和持续时间
        resource_requirements = {}
        
        for segment in task.segments:
            res_type = segment.resource_type
            
            # 找到合适的资源
            available_resource = None
            min_available_time = None
            
            for resource in scheduler.resources[res_type]:
                # 计算这个资源何时可用
                next_available = self.get_resource_next_available_time(
                    resource.unit_id, current_time_dec
                )
                
                if min_available_time is None or next_available < min_available_time:
                    min_available_time = next_available
                    available_resource = resource
            
            if available_resource is None:
                return None
            
            # 计算任务在这个资源上的持续时间
            duration = segment.get_duration(available_resource.bandwidth)
            duration_dec = self.to_decimal(duration)
            
            resource_requirements[res_type] = {
                'resource_id': available_resource.unit_id,
                'resource': available_resource,
                'segment': segment,
                'duration': duration_dec,
                'earliest_start': max(current_time_dec, min_available_time)
            }
        
        # 检查所有资源是否能在合理时间内可用
        max_wait_time = self.to_decimal(10.0)  # 最多等待10ms
        
        for res_type, req in resource_requirements.items():
            if req['earliest_start'] > current_time_dec + max_wait_time:
                return None  # 等待时间太长
            
            assigned_resources[res_type] = req['resource_id']
        
        return assigned_resources if assigned_resources else None
    
    def get_resource_next_available_time(self, resource_id: str, current_time: Decimal) -> Decimal:
        """获取资源的下一个可用时间"""
        
        if resource_id not in self.resource_timeline:
            return current_time
        
        # 获取该资源的所有占用时间段，按时间排序
        events = sorted(self.resource_timeline[resource_id], key=lambda x: x['start'])
        
        # 找到当前时间之后的第一个可用时间
        for event in events:
            if event['start'] > current_time:
                # 检查当前时间到这个事件开始是否有足够空间
                if event['start'] >= current_time + self.safety_buffer:
                    return current_time
                else:
                    # 继续找下一个空隙
                    current_time = event['end'] + self.safety_buffer
            elif event['end'] > current_time:
                # 当前时间在这个事件中间，需要等到事件结束
                current_time = event['end'] + self.safety_buffer
        
        return current_time
    
    def reserve_resource_time(self, resource_id: str, start_time: Decimal, 
                            end_time: Decimal, task_id: str):
        """预留资源时间"""
        
        if resource_id not in self.resource_timeline:
            self.resource_timeline[resource_id] = []
        
        # 添加新的占用时间段
        self.resource_timeline[resource_id].append({
            'start': start_time,
            'end': end_time,
            'task_id': task_id
        })
        
        # 重新排序以保持时间顺序
        self.resource_timeline[resource_id].sort(key=lambda x: x['start'])
    
    def schedule_with_strict_timing(self, scheduler, time_window: float):
        """使用严格时间管理的调度方法"""
        
        # 清理状态
        self.resource_timeline.clear()
        scheduler.schedule_history = []
        scheduler.active_bindings = []
        
        # 重置资源队列
        for queue in scheduler.resource_queues.values():
            queue.available_time = 0.0
            if hasattr(queue, 'release_binding'):
                queue.release_binding()
        
        # 重置任务状态
        for task in scheduler.tasks.values():
            task.schedule_info = None
            task.last_execution_time = -float('inf')
            task.ready_time = 0
        
        current_time = 0.0
        scheduled_events = []
        task_execution_count = {task_id: 0 for task_id in scheduler.tasks}
        time_window_dec = self.to_decimal(time_window)
        
        print(f"开始统一调度 (时间窗口: {time_window}ms)...")
        
        while self.to_decimal(current_time) < time_window_dec:
            
            # 清理过期绑定
            scheduler.cleanup_expired_bindings(current_time)
            
            # 获取就绪任务并按优先级排序
            ready_tasks = self.get_ready_tasks(scheduler, current_time, task_execution_count)
            
            if not ready_tasks:
                # 没有就绪任务，推进时间
                current_time = self.find_next_meaningful_time(scheduler, current_time)
                if current_time >= time_window:
                    break
                continue
            
            # 尝试调度最高优先级的任务
            scheduled_any = False
            
            for task in ready_tasks:
                success, new_current_time = self.try_schedule_task(
                    scheduler, task, current_time, task_execution_count
                )
                
                if success:
                    scheduled_events.extend(scheduler.schedule_history[-1:])  # 添加最新的调度事件
                    current_time = new_current_time
                    scheduled_any = True
                    break  # 只调度一个任务，然后重新评估
            
            if not scheduled_any:
                # 没有任务能调度，推进时间
                current_time += 1.0
        
        print(f"✅ 统一调度完成: {len(scheduled_events)} 个事件")
        
        # 验证结果
        conflicts = self.detect_conflicts()
        if conflicts:
            print(f"❌ 发现 {len(conflicts)} 个冲突 - 这不应该发生!")
            for conflict in conflicts[:3]:
                print(f"  {conflict}")
        else:
            print("✅ 验证通过: 无资源冲突")
        
        return scheduled_events
    
    def get_ready_tasks(self, scheduler, current_time, task_execution_count):
        """获取就绪的任务列表"""
        
        ready_tasks = []
        
        for task in scheduler.tasks.values():
            # 检查FPS间隔
            if task.last_execution_time > -float('inf'):
                min_interval = 1000.0 / task.fps_requirement
                if current_time - task.last_execution_time < min_interval - 0.1:
                    continue
            
            # 检查依赖关系
            deps_satisfied = True
            for dep_id in task.dependencies:
                if dep_id in scheduler.tasks:
                    if task_execution_count[dep_id] <= task_execution_count[task.task_id]:
                        deps_satisfied = False
                        break
            
            if deps_satisfied:
                ready_tasks.append(task)
        
        # 严格按优先级排序（优先级值越小越高）
        ready_tasks.sort(key=lambda t: (t.priority.value, t.task_id))
        
        return ready_tasks
    
    def try_schedule_task(self, scheduler, task, current_time, task_execution_count):
        """尝试调度单个任务"""
        
        # 查找可用资源
        assigned_resources = self.find_resources_for_task(scheduler, task, current_time)
        
        if not assigned_resources:
            return False, current_time
        
        # 计算任务的实际开始时间和持续时间
        actual_start_time = current_time
        task_duration = 0
        
        # 预留资源时间
        resource_reservations = []
        
        for segment in task.segments:
            if segment.resource_type in assigned_resources:
                resource_id = assigned_resources[segment.resource_type]
                
                # 找到对应的资源
                resource = None
                for res in scheduler.resources[segment.resource_type]:
                    if res.unit_id == resource_id:
                        resource = res
                        break
                
                if resource:
                    # 计算段的开始时间和持续时间
                    segment_start = actual_start_time + segment.start_time
                    duration = segment.get_duration(resource.bandwidth)
                    segment_end = segment_start + duration
                    
                    # 找到实际可用的时间槽
                    start_dec = self.to_decimal(segment_start)
                    duration_dec = self.to_decimal(duration)
                    
                    available_start = self.get_resource_next_available_time(resource_id, start_dec)
                    available_end = available_start + duration_dec
                    
                    # 预留时间
                    resource_reservations.append({
                        'resource_id': resource_id,
                        'start': available_start,
                        'end': available_end,
                        'task_id': task.task_id
                    })
                    
                    # 更新任务持续时间
                    actual_segment_end = float(available_end)
                    task_duration = max(task_duration, actual_segment_end - actual_start_time)
        
        # 正式预留所有资源
        for reservation in resource_reservations:
            self.reserve_resource_time(
                reservation['resource_id'],
                reservation['start'],
                reservation['end'],
                reservation['task_id']
            )
        
        # 创建调度信息
        schedule_info = TaskScheduleInfo(
            task_id=task.task_id,
            start_time=actual_start_time,
            end_time=actual_start_time + task_duration,
            assigned_resources=assigned_resources,
            actual_latency=task_duration,
            runtime_type=task.runtime_type,
            used_cuts=[],
            segmentation_overhead=0.0,
            sub_segment_schedule=[]
        )
        
        # 更新任务状态
        task.schedule_info = schedule_info
        task.last_execution_time = actual_start_time
        task_execution_count[task.task_id] += 1
        
        # 更新资源队列状态
        for res_type, res_id in assigned_resources.items():
            queue = scheduler.resource_queues[res_id]
            queue.available_time = actual_start_time + task_duration
        
        # 记录调度
        scheduler.schedule_history.append(schedule_info)
        
        return True, actual_start_time + 0.1  # 小步推进
    
    def find_next_meaningful_time(self, scheduler, current_time):
        """找到下一个有意义的时间点"""
        
        next_time = current_time + 10.0  # 默认推进10ms
        
        # 检查任务的下一个就绪时间
        for task in scheduler.tasks.values():
            if task.last_execution_time > -float('inf'):
                min_interval = 1000.0 / task.fps_requirement
                next_ready_time = task.last_execution_time + min_interval
                if next_ready_time > current_time:
                    next_time = min(next_time, next_ready_time)
        
        # 检查资源何时可用
        for resource_id in self.resource_timeline:
            next_available = float(self.get_resource_next_available_time(
                resource_id, self.to_decimal(current_time)
            ))
            if next_available > current_time:
                next_time = min(next_time, next_available)
        
        return next_time
    
    def detect_conflicts(self):
        """检测资源冲突"""
        
        conflicts = []
        
        for resource_id, events in self.resource_timeline.items():
            sorted_events = sorted(events, key=lambda x: x['start'])
            
            for i in range(len(sorted_events) - 1):
                current = sorted_events[i]
                next_event = sorted_events[i + 1]
                
                if current['end'] > next_event['start']:
                    overlap = float(current['end'] - next_event['start'])
                    conflicts.append(
                        f"资源冲突 {resource_id}: {current['task_id']} "
                        f"({float(current['start']):.1f}-{float(current['end']):.1f}ms) "
                        f"与 {next_event['task_id']} "
                        f"({float(next_event['start']):.1f}-{float(next_event['end']):.1f}ms) "
                        f"重叠 {overlap:.1f}ms"
                    )
        
        return conflicts


def validate_unified_schedule(scheduler):
    """验证统一调度的结果"""
    
    if not hasattr(scheduler, '_unified_time_manager'):
        return False, ["统一时间管理器未找到"]
    
    conflicts = scheduler._unified_time_manager.detect_conflicts()
    return len(conflicts) == 0, conflicts


if __name__ == "__main__":
    print("统一Dragon4修复方案")
    print("使用方法:")
    print("  from unified_dragon4_fix import apply_unified_dragon4_fix")
    print("  apply_unified_dragon4_fix(scheduler)")
    print("  # 现在调度器将保证零资源冲突")
