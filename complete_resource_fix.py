#!/usr/bin/env python3
"""
完整的资源冲突修复方案
解决调度器中多个任务同时执行在同一资源上的问题
"""

from typing import List, Dict, Optional, Tuple
from enums import TaskPriority, ResourceType, RuntimeType
from models import TaskScheduleInfo
from collections import defaultdict


def apply_complete_resource_fix(scheduler):
    """应用完整的资源冲突修复"""
    print("🔧 应用完整资源冲突修复...")
    
    # 1. 修复资源可用性检查
    fix_resource_availability_check(scheduler)
    
    # 2. 修复优先级调度逻辑
    fix_priority_scheduling_logic(scheduler)
    
    # 3. 添加严格的资源冲突检测
    add_strict_conflict_detection(scheduler)
    
    print("✅ 完整资源冲突修复已应用")


def fix_resource_availability_check(scheduler):
    """修复资源可用性检查逻辑"""
    
    # 保存原始方法
    original_find_resources = scheduler.find_available_resources_for_task_with_segmentation
    
    def strict_find_available_resources(task, current_time):
        """严格的资源可用性检查"""
        
        # 检查每个所需的资源类型
        assigned_resources = {}
        
        for segment in task.segments:
            res_type = segment.resource_type
            
            # 找到该类型的可用资源
            available_resource = None
            
            for resource in scheduler.resources[res_type]:
                queue = scheduler.resource_queues.get(resource.unit_id)
                
                if queue is None:
                    continue
                
                # 关键修复：检查资源是否真正可用
                # 资源必须在当前时间或之前变为可用
                if queue.available_time <= current_time + 0.001:  # 微小容差
                    # 额外检查：确保没有其他任务正在使用这个资源
                    if not is_resource_busy(scheduler, resource.unit_id, current_time):
                        available_resource = resource
                        break
            
            if available_resource is None:
                # 如果任何一个所需资源不可用，返回None
                return None
            
            assigned_resources[res_type] = available_resource.unit_id
        
        return assigned_resources if assigned_resources else None
    
    # 替换方法
    scheduler.find_available_resources_for_task_with_segmentation = strict_find_available_resources
    print("  ✓ 资源可用性检查已修复")


def is_resource_busy(scheduler, resource_id, current_time):
    """检查资源是否正在被其他任务使用"""
    
    # 检查调度历史中是否有任务正在使用这个资源
    for schedule in scheduler.schedule_history:
        if (schedule.start_time <= current_time < schedule.end_time and 
            resource_id in schedule.assigned_resources.values()):
            return True
    
    # 检查活跃绑定
    for binding in scheduler.active_bindings:
        if (binding.start_time <= current_time < binding.end_time and 
            resource_id in binding.resource_ids):
            return True
    
    return False


def fix_priority_scheduling_logic(scheduler):
    """修复优先级调度逻辑"""
    
    def enhanced_priority_schedule(time_window: float = 150.0):
        """增强的优先级调度，确保高优先级任务优先"""
        
        # 清理状态
        scheduler.schedule_history = []
        scheduler.active_bindings = []
        
        # 重置资源队列
        for queue in scheduler.resource_queues.values():
            queue.available_time = 0.0
        
        # 重置任务状态
        for task in scheduler.tasks.values():
            task.schedule_info = None
            task.last_execution_time = -float('inf')
        
        current_time = 0.0
        scheduled_events = []
        task_execution_count = {task_id: 0 for task_id in scheduler.tasks}
        
        print(f"调度{time_window}ms...")
        
        # 主调度循环
        while current_time < time_window:
            
            # 找到所有就绪的任务
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
            
            if not ready_tasks:
                # 没有就绪任务，推进时间
                current_time += 1.0
                continue
            
            # 关键修复：严格按优先级排序
            # 优先级值越小，优先级越高
            ready_tasks.sort(key=lambda t: (t.priority.value, t.task_id))
            
            # 尝试调度任务（从高优先级开始）
            scheduled_in_this_round = False
            
            for task in ready_tasks:
                # 查找可用资源
                assigned_resources = scheduler.find_available_resources_for_task_with_segmentation(
                    task, current_time
                )
                
                if assigned_resources:
                    # 计算任务持续时间
                    task_duration = calculate_task_duration(task, assigned_resources, scheduler)
                    
                    # 创建调度信息
                    schedule_info = TaskScheduleInfo(
                        task_id=task.task_id,
                        start_time=current_time,
                        end_time=current_time + task_duration,
                        assigned_resources=assigned_resources,
                        actual_latency=task_duration,
                        runtime_type=task.runtime_type,
                        used_cuts=[],
                        segmentation_overhead=0.0,
                        sub_segment_schedule=[]
                    )
                    
                    # 更新任务状态
                    task.schedule_info = schedule_info
                    task.last_execution_time = current_time
                    task_execution_count[task.task_id] += 1
                    
                    # 关键修复：更新资源的可用时间
                    for res_type, res_id in assigned_resources.items():
                        queue = scheduler.resource_queues[res_id]
                        queue.available_time = current_time + task_duration
                    
                    # 记录调度
                    scheduler.schedule_history.append(schedule_info)
                    scheduled_events.append(schedule_info)
                    scheduled_in_this_round = True
                    
                    print(f"{current_time:6.1f}ms: [{task.priority.name:6}] {task.task_id} 开始")
                    
                    # 重要：只调度一个任务，然后重新评估
                    # 这确保高优先级任务总是优先
                    break
            
            if scheduled_in_this_round:
                # 小步推进，给其他任务机会
                current_time += 0.1
            else:
                # 没有任务能调度，找到下一个可能的时间点
                next_available_time = find_next_available_time(scheduler, ready_tasks, current_time)
                current_time = min(next_available_time, time_window)
        
        print(f"✅ 优先级调度完成: {len(scheduled_events)} 个事件")
        
        # 打印优先级分布
        priority_stats = defaultdict(int)
        for event in scheduled_events:
            task = scheduler.tasks[event.task_id]
            priority_stats[task.priority.name] += 1
        
        print("优先级分布:")
        for priority, count in priority_stats.items():
            print(f"  {priority}: {count} 个事件")
        
        return scheduled_events
    
    # 替换调度方法
    scheduler.priority_aware_schedule_with_segmentation = enhanced_priority_schedule
    print("  ✓ 优先级调度逻辑已修复")


def calculate_task_duration(task, assigned_resources, scheduler):
    """计算任务的执行持续时间"""
    
    task_duration = 0
    
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
                # 计算段的持续时间
                duration = segment.get_duration(resource.bandwidth)
                end_time = segment.start_time + duration
                task_duration = max(task_duration, end_time)
    
    return task_duration


def find_next_available_time(scheduler, ready_tasks, current_time):
    """找到下一个可能的调度时间"""
    
    next_time = current_time + 10.0  # 默认推进10ms
    
    # 检查资源何时可用
    for queue in scheduler.resource_queues.values():
        if queue.available_time > current_time:
            next_time = min(next_time, queue.available_time)
    
    # 检查任务何时就绪（FPS约束）
    for task in ready_tasks:
        if task.last_execution_time > -float('inf'):
            min_interval = 1000.0 / task.fps_requirement
            next_ready_time = task.last_execution_time + min_interval
            if next_ready_time > current_time:
                next_time = min(next_time, next_ready_time)
    
    return next_time


def add_strict_conflict_detection(scheduler):
    """添加严格的资源冲突检测"""
    
    def detect_conflicts_in_schedule():
        """检测调度中的资源冲突"""
        
        conflicts = []
        resource_timeline = defaultdict(list)
        
        # 构建资源时间线
        for schedule in scheduler.schedule_history:
            for res_type, res_id in schedule.assigned_resources.items():
                resource_timeline[res_id].append({
                    'start': schedule.start_time,
                    'end': schedule.end_time,
                    'task': schedule.task_id
                })
        
        # 检查每个资源的冲突
        for res_id, timeline in resource_timeline.items():
            # 按开始时间排序
            timeline.sort(key=lambda x: x['start'])
            
            # 检查重叠
            for i in range(len(timeline) - 1):
                curr = timeline[i]
                next_event = timeline[i + 1]
                
                if curr['end'] > next_event['start'] + 0.001:  # 允许微小误差
                    conflicts.append({
                        'resource': res_id,
                        'task1': curr['task'],
                        'task2': next_event['task'],
                        'overlap': curr['end'] - next_event['start'],
                        'time': next_event['start']
                    })
        
        return conflicts
    
    # 添加到调度器
    scheduler.detect_conflicts = detect_conflicts_in_schedule
    print("  ✓ 严格冲突检测已添加")


def validate_fixed_schedule(scheduler):
    """验证修复后的调度是否正确"""
    
    print("\n=== 调度验证 ===")
    
    # 检测冲突
    conflicts = scheduler.detect_conflicts()
    
    if conflicts:
        print(f"❌ 发现 {len(conflicts)} 个冲突:")
        for conflict in conflicts:
            print(f"  资源 {conflict['resource']}: {conflict['task1']} 与 {conflict['task2']} "
                  f"在 {conflict['time']:.1f}ms 重叠 {conflict['overlap']:.1f}ms")
        return False
    else:
        print("✅ 没有资源冲突")
        
        # 检查任务执行统计
        task_counts = defaultdict(int)
        for schedule in scheduler.schedule_history:
            task_counts[schedule.task_id] += 1
        
        print("\n任务执行统计:")
        for task_id in sorted(task_counts.keys()):
            count = task_counts[task_id]
            task = scheduler.tasks[task_id]
            print(f"  {task_id}: {count}次 (优先级: {task.priority.name})")
        
        return True


if __name__ == "__main__":
    print("完整资源冲突修复方案")
    print("使用方法:")
    print("  from complete_resource_fix import apply_complete_resource_fix")
    print("  apply_complete_resource_fix(scheduler)")
    print("  validate_fixed_schedule(scheduler)")
