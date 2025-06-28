#!/usr/bin/env python3
"""
高FPS感知调度器
特别优化高FPS任务（如T6的100FPS需求）
"""

from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from enums import ResourceType, TaskPriority


def apply_high_fps_aware_scheduling(scheduler):
    """应用高FPS感知调度补丁"""
    
    print("🔧 应用高FPS感知调度...")
    
    # 保存原始调度方法
    original_schedule = scheduler.priority_aware_schedule_with_segmentation
    
    def high_fps_aware_schedule(time_window: float = 200.0):
        """高FPS感知的调度方法"""
        
        # 识别高FPS任务（FPS >= 50）
        high_fps_tasks = {}
        normal_tasks = {}
        
        for task_id, task in scheduler.tasks.items():
            if task.fps_requirement >= 50:
                high_fps_tasks[task_id] = task
                print(f"  🚀 高FPS任务: {task_id} ({task.name}) - {task.fps_requirement} FPS")
            else:
                normal_tasks[task_id] = task
        
        # 重置调度状态
        reset_scheduler_state(scheduler)
        
        # 创建任务执行计数器
        task_execution_count = defaultdict(int)
        task_last_execution = defaultdict(lambda: -float('inf'))
        
        # 主调度循环
        current_time = 0.0
        
        while current_time < time_window:
            # 1. 首先尝试调度高FPS任务
            scheduled_any = False
            
            for task_id, task in high_fps_tasks.items():
                # 计算该任务的理想执行时间
                min_interval = 1000.0 / task.fps_requirement
                next_ideal_time = task_last_execution[task_id] + min_interval
                
                # 如果当前时间适合执行该任务
                if current_time >= next_ideal_time - 0.1:  # 允许0.1ms的提前
                    # 检查依赖
                    if check_dependencies_satisfied(scheduler, task, task_execution_count):
                        # 尝试分配资源
                        assigned_resources = find_resources_for_high_fps_task(
                            scheduler, task, current_time
                        )
                        
                        if assigned_resources:
                            # 执行调度
                            schedule_info = execute_task_scheduling(
                                scheduler, task, assigned_resources, current_time
                            )
                            
                            if schedule_info:
                                task_execution_count[task_id] += 1
                                task_last_execution[task_id] = current_time
                                scheduled_any = True
                                # 高FPS任务调度后立即继续，不break
            
            # 2. 然后调度普通任务
            for task_id, task in normal_tasks.items():
                # 检查执行间隔
                min_interval = 1000.0 / task.fps_requirement if task.fps_requirement > 0 else 100.0
                
                if current_time - task_last_execution[task_id] >= min_interval:
                    # 检查依赖
                    if check_dependencies_satisfied(scheduler, task, task_execution_count):
                        # 尝试分配资源
                        assigned_resources = scheduler.find_available_resources_for_task_with_segmentation(
                            task, current_time
                        )
                        
                        if assigned_resources:
                            # 执行调度
                            schedule_info = execute_task_scheduling(
                                scheduler, task, assigned_resources, current_time
                            )
                            
                            if schedule_info:
                                task_execution_count[task_id] += 1
                                task_last_execution[task_id] = current_time
                                scheduled_any = True
            
            # 3. 时间推进
            if not scheduled_any:
                # 找下一个可能的调度时间
                next_time = current_time + 0.1
                
                # 检查高FPS任务的下一个理想执行时间
                for task_id, task in high_fps_tasks.items():
                    min_interval = 1000.0 / task.fps_requirement
                    next_ideal = task_last_execution[task_id] + min_interval
                    if next_ideal > current_time:
                        next_time = min(next_time, next_ideal)
                
                current_time = min(next_time, time_window)
            else:
                # 有任务被调度，小步前进
                current_time += 0.05  # 更小的步进以捕获高FPS机会
        
        # 打印高FPS任务执行统计
        print_high_fps_statistics(scheduler, high_fps_tasks, task_execution_count, time_window)
        
        return scheduler.schedule_history
    
    # 替换调度方法
    scheduler.priority_aware_schedule_with_segmentation = high_fps_aware_schedule
    
    print("✅ 高FPS感知调度已应用")


def reset_scheduler_state(scheduler):
    """重置调度器状态"""
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
    
    # 清空调度历史
    scheduler.schedule_history.clear()
    if hasattr(scheduler, 'active_bindings'):
        scheduler.active_bindings.clear()


def check_dependencies_satisfied(scheduler, task, execution_count):
    """检查任务依赖是否满足"""
    if not hasattr(task, 'dependencies'):
        return True
    
    for dep_id in task.dependencies:
        if dep_id in scheduler.tasks:
            # 依赖任务必须至少执行过一次
            if execution_count[dep_id] == 0:
                return False
            
            # 对于高FPS任务，可能需要更灵活的依赖检查
            # 例如：允许在依赖任务的执行间隔内执行
    
    return True


def find_resources_for_high_fps_task(scheduler, task, current_time):
    """为高FPS任务寻找资源（更激进的策略）"""
    
    assigned_resources = {}
    
    # 获取任务需要的资源类型
    for segment in task.segments:
        res_type = segment.resource_type
        
        # 找最快可用的资源
        best_resource = None
        earliest_available = float('inf')
        
        for resource in scheduler.resources[res_type]:
            queue = scheduler.resource_queues[resource.unit_id]
            
            # 对高FPS任务，即使资源稍后可用也考虑
            if queue.available_time <= current_time + 0.5:  # 允许0.5ms的等待
                if queue.available_time < earliest_available:
                    earliest_available = queue.available_time
                    best_resource = resource
        
        if best_resource:
            assigned_resources[res_type] = best_resource.unit_id
        else:
            return None
    
    return assigned_resources


def execute_task_scheduling(scheduler, task, assigned_resources, current_time):
    """执行任务调度"""
    
    # 获取子段
    sub_segments = task.get_sub_segments_for_scheduling()
    
    actual_start = current_time
    actual_end = actual_start
    sub_segment_schedule = []
    
    # 调度每个子段
    for sub_seg in sub_segments:
        if sub_seg.resource_type in assigned_resources:
            resource_id = assigned_resources[sub_seg.resource_type]
            resource = next(r for r in scheduler.resources[sub_seg.resource_type] 
                          if r.unit_id == resource_id)
            
            # 计算执行时间
            sub_seg_start = actual_start + sub_seg.start_time
            sub_seg_duration = sub_seg.get_duration(resource.bandwidth)
            sub_seg_end = sub_seg_start + sub_seg_duration
            
            # 更新资源可用时间
            scheduler.resource_queues[resource_id].available_time = sub_seg_end
            
            # 记录子段调度
            sub_segment_schedule.append((sub_seg.sub_id, sub_seg_start, sub_seg_end))
            
            actual_end = max(actual_end, sub_seg_end)
    
    # 创建调度信息
    from models import TaskScheduleInfo
    schedule_info = TaskScheduleInfo(
        task_id=task.task_id,
        start_time=actual_start,
        end_time=actual_end,
        assigned_resources=assigned_resources,
        actual_latency=actual_end - current_time,
        runtime_type=task.runtime_type,
        sub_segment_schedule=sub_segment_schedule
    )
    
    # 记录调度
    scheduler.schedule_history.append(schedule_info)
    task.schedule_info = schedule_info
    task.last_execution_time = actual_start
    
    return schedule_info


def print_high_fps_statistics(scheduler, high_fps_tasks, execution_count, time_window):
    """打印高FPS任务执行统计"""
    
    print(f"\n📊 高FPS任务执行统计:")
    print(f"{'任务ID':<10} {'名称':<15} {'要求FPS':<10} {'期望次数':<10} {'实际次数':<10} {'达成率':<10}")
    print("-" * 75)
    
    for task_id, task in high_fps_tasks.items():
        expected = int((time_window / 1000.0) * task.fps_requirement)
        actual = execution_count[task_id]
        rate = (actual / expected * 100) if expected > 0 else 0
        
        status = "✅" if rate >= 95 else "❌"
        
        print(f"{task_id:<10} {task.name:<15} {task.fps_requirement:<10.0f} "
              f"{expected:<10} {actual:<10} {rate:<9.1f}% {status}")


if __name__ == "__main__":
    print("高FPS感知调度器")
    print("特性：")
    print("1. 优先调度高FPS任务")
    print("2. 更小的时间步进捕获执行机会")
    print("3. 灵活的资源分配策略")
    print("4. 专门的高FPS任务统计")
