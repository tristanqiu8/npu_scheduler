#!/usr/bin/env python3
"""
简单有效的调度紧凑化算法
真正消除调度中的空隙
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
import copy
from collections import defaultdict

from scheduler import MultiResourceScheduler
from enums import ResourceType


@dataclass
class TimeSlot:
    """时间槽"""
    start: float
    end: float
    
    @property
    def duration(self):
        return self.end - self.start


class SimpleCompactor:
    """简单的调度紧凑化器"""
    
    def __init__(self, scheduler: MultiResourceScheduler, time_window: float = 200.0):
        self.scheduler = scheduler
        self.time_window = time_window
        
    def compact(self) -> Tuple[List, float]:
        """
        执行简单的紧凑化
        返回: (紧凑化后的事件列表, 末尾空闲时间)
        """
        print("\n🔧 执行简单紧凑化...")
        
        # 获取原始事件
        original_events = copy.deepcopy(self.scheduler.schedule_history)
        if not original_events:
            return [], self.time_window
            
        # 按时间窗口分组
        windows = self._group_by_window(original_events)
        
        # 对每个窗口进行紧凑化
        compacted_events = []
        
        for window_idx in sorted(windows.keys()):
            window_events = windows[window_idx]
            window_start = window_idx * self.time_window
            
            # 对窗口内的事件进行紧凑化
            compacted_window = self._compact_window(window_events, window_start)
            compacted_events.extend(compacted_window)
        
        # 计算第一个窗口的末尾空闲时间
        first_window_events = [e for e in compacted_events if e.start_time < self.time_window]
        if first_window_events:
            last_end = max(e.end_time for e in first_window_events)
            idle_time = self.time_window - last_end
        else:
            idle_time = self.time_window
            
        print(f"✅ 紧凑化完成，末尾空闲时间: {idle_time:.1f}ms")
        
        return compacted_events, idle_time
    
    def _group_by_window(self, events: List) -> Dict[int, List]:
        """按时间窗口分组事件"""
        windows = defaultdict(list)
        for event in events:
            window_idx = int(event.start_time // self.time_window)
            windows[window_idx].append(event)
        return windows
    
    def _compact_window(self, events: List, window_start: float) -> List:
        """紧凑化一个时间窗口内的事件"""
        if not events:
            return []
        
        # 按任务分组
        task_groups = defaultdict(list)
        for event in events:
            task_groups[event.task_id].append(event)
        
        # 获取任务依赖关系
        dependencies = self._get_dependencies()
        
        # 计算任务执行顺序（拓扑排序）
        task_order = self._topological_sort(list(task_groups.keys()), dependencies)
        
        # 初始化资源可用时间
        resource_available = defaultdict(lambda: window_start)
        
        # 按顺序调度任务
        compacted = []
        task_end_times = {}
        
        for task_id in task_order:
            if task_id not in task_groups:
                continue
                
            task_events = sorted(task_groups[task_id], key=lambda e: e.start_time)
            
            # 计算任务的最早开始时间
            earliest_start = window_start
            
            # 考虑依赖关系
            if task_id in dependencies:
                for dep_id in dependencies[task_id]:
                    if dep_id in task_end_times:
                        earliest_start = max(earliest_start, task_end_times[dep_id])
            
            # 找到所有需要的资源都可用的最早时间
            for event in task_events:
                for res_type, res_id in event.assigned_resources.items():
                    earliest_start = max(earliest_start, resource_available[res_id])
            
            # 调度任务的所有事件
            task_start = earliest_start
            time_shift = task_start - task_events[0].start_time
            
            for event in task_events:
                # 创建新事件
                new_event = copy.deepcopy(event)
                new_event.start_time = event.start_time + time_shift
                new_event.end_time = event.end_time + time_shift
                
                # 检查是否超出窗口
                if new_event.start_time >= window_start + self.time_window:
                    # 如果超出窗口，保持原位置
                    compacted.append(copy.deepcopy(event))
                else:
                    compacted.append(new_event)
                    
                    # 更新资源可用时间
                    for res_type, res_id in new_event.assigned_resources.items():
                        resource_available[res_id] = max(resource_available[res_id], new_event.end_time)
                    
                    # 更新任务结束时间
                    task_end_times[task_id] = max(task_end_times.get(task_id, 0), new_event.end_time)
        
        return compacted
    
    def _get_dependencies(self) -> Dict[str, Set[str]]:
        """获取任务依赖关系"""
        dependencies = {}
        for task_id, task in self.scheduler.tasks.items():
            if task.dependencies:
                dependencies[task_id] = set(task.dependencies)
        return dependencies
    
    def _topological_sort(self, tasks: List[str], dependencies: Dict[str, Set[str]]) -> List[str]:
        """拓扑排序，考虑优先级"""
        # 计算入度
        in_degree = {task: 0 for task in tasks}
        for task in tasks:
            if task in dependencies:
                for dep in dependencies[task]:
                    if dep in tasks:
                        in_degree[task] += 1
        
        # 按优先级分组
        priority_groups = defaultdict(list)
        for task in tasks:
            priority = self.scheduler.tasks[task].priority.value
            priority_groups[priority].append(task)
        
        # 执行拓扑排序
        result = []
        available = []
        
        # 初始化可用任务（无依赖）
        for priority in sorted(priority_groups.keys()):
            for task in priority_groups[priority]:
                if in_degree[task] == 0:
                    available.append((priority, task))
        
        available.sort()  # 按优先级排序
        
        while available:
            _, task = available.pop(0)
            result.append(task)
            
            # 更新依赖此任务的其他任务
            for other_task in tasks:
                if other_task in dependencies and task in dependencies[other_task]:
                    in_degree[other_task] -= 1
                    if in_degree[other_task] == 0:
                        priority = self.scheduler.tasks[other_task].priority.value
                        available.append((priority, other_task))
                        available.sort()
        
        return result


def visualize_compaction(scheduler, original_events, compacted_events, idle_time, time_window=200.0):
    """生成紧凑化可视化"""
    from elegant_visualization import ElegantSchedulerVisualizer
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # 生成原始调度图（第一个窗口）
    original_first = [e for e in original_events if e.start_time < time_window]
    scheduler.schedule_history = original_first
    viz1 = ElegantSchedulerVisualizer(scheduler)
    viz1.plot_elegant_gantt()
    plt.title('Original Schedule - First Window')
    plt.xlim(0, time_window)
    plt.savefig('simple_compact_original.png', dpi=120)
    plt.close()
    
    # 生成紧凑化后的调度图（第一个窗口）
    compacted_first = [e for e in compacted_events if e.start_time < time_window]
    scheduler.schedule_history = compacted_first
    viz2 = ElegantSchedulerVisualizer(scheduler)
    viz2.plot_elegant_gantt()
    
    # 标记空闲区域
    if idle_time > 0:
        idle_start = time_window - idle_time
        plt.axvspan(idle_start, time_window, alpha=0.3, color='lightgreen')
        plt.axvline(x=idle_start, color='green', linestyle='--', linewidth=2)
        plt.text(idle_start + idle_time/2, plt.ylim()[1]*0.95,
                f'{idle_time:.1f}ms\nIDLE', 
                ha='center', va='top', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.title(f'Compacted Schedule - {idle_time:.1f}ms Idle at End')
    plt.xlim(0, time_window)
    plt.savefig('simple_compact_result.png', dpi=120)
    plt.close()
    
    # 生成Chrome trace
    scheduler.schedule_history = compacted_events
    viz3 = ElegantSchedulerVisualizer(scheduler)
    viz3.export_chrome_tracing('simple_compacted_trace.json')
    
    print("\n✅ 生成的文件:")
    print("  - simple_compact_original.png")
    print("  - simple_compact_result.png")
    print("  - simple_compacted_trace.json")


def test_simple_compactor():
    """测试简单紧凑化器"""
    # 这里需要一个已经优化过的scheduler实例
    # 作为示例，我们假设它已经存在
    print("🧪 测试简单紧凑化算法...")
    
    # 需要实际的scheduler来测试
    # scheduler = get_scheduler_instance()
    # 
    # compactor = SimpleCompactor(scheduler)
    # compacted_events, idle_time = compactor.compact()
    # 
    # visualize_compaction(scheduler, scheduler.schedule_history, 
    #                     compacted_events, idle_time)


if __name__ == "__main__":
    test_simple_compactor()
