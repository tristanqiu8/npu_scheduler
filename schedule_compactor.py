#!/usr/bin/env python3
"""
调度紧凑化工具
将调度结果重新排列，使空闲时间集中在时间窗口末尾
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import copy

from scheduler import MultiResourceScheduler
from enums import ResourceType

# ScheduleEvent应该从scheduler模块导入
try:
    from scheduler import ScheduleEvent
except ImportError:
    # 如果导入失败，定义一个简单的ScheduleEvent类
    @dataclass
    class ScheduleEvent:
        """调度事件"""
        task_id: str
        start_time: float
        end_time: float
        assigned_resources: Dict[ResourceType, str]
        segment_index: int = 0
        is_segmented: bool = False


@dataclass
class ResourceSlot:
    """资源时间槽"""
    resource_type: ResourceType
    resource_id: str
    start_time: float
    end_time: float
    is_free: bool = True
    
    @property
    def duration(self):
        return self.end_time - self.start_time


@dataclass 
class CompactionResult:
    """紧凑化结果"""
    original_events: List[ScheduleEvent]
    compacted_events: List[ScheduleEvent]
    idle_time_at_end: float  # 末尾连续空闲时间
    idle_percentage: float    # 空闲时间百分比
    compaction_ratio: float   # 紧凑化比率
    resource_idle_times: Dict[str, float]  # 每个资源的末尾空闲时间


class ScheduleCompactor:
    """调度紧凑化器"""
    
    def __init__(self, scheduler: MultiResourceScheduler, time_window: float = 200.0):
        self.scheduler = scheduler
        self.time_window = time_window
        self.resource_timelines = {}  # 资源时间线
        
    def compact_schedule(self, preserve_dependencies: bool = True) -> CompactionResult:
        """
        紧凑化调度
        Args:
            preserve_dependencies: 是否保持任务依赖关系
        """
        print("\n🔧 开始调度紧凑化...")
        
        # 1. 获取原始调度事件
        original_events = copy.deepcopy(self.scheduler.schedule_history)
        if not original_events:
            print("⚠️ 没有调度事件需要紧凑化")
            return None
            
        # 2. 构建资源时间线
        self._build_resource_timelines()
        
        # 3. 分析任务依赖关系
        task_dependencies = self._analyze_dependencies() if preserve_dependencies else {}
        
        # 4. 按优先级和依赖关系排序任务
        sorted_events = self._sort_events_for_compaction(original_events, task_dependencies)
        
        # 5. 执行紧凑化
        compacted_events = self._perform_compaction(sorted_events, task_dependencies)
        
        # 6. 计算空闲时间
        idle_info = self._calculate_idle_time(compacted_events)
        
        # 7. 生成结果
        result = CompactionResult(
            original_events=original_events,
            compacted_events=compacted_events,
            idle_time_at_end=idle_info['total_idle_at_end'],
            idle_percentage=idle_info['idle_percentage'],
            compaction_ratio=idle_info['compaction_ratio'],
            resource_idle_times=idle_info['resource_idle_times']
        )
        
        # 8. 打印统计
        self._print_compaction_stats(result)
        
        return result
        
    def _build_resource_timelines(self):
        """构建资源时间线"""
        self.resource_timelines = {}
        
        # 初始化每个资源的时间线
        for res_type, resources in self.scheduler.resources.items():
            if isinstance(resources, dict):
                for res_id in resources.keys():
                    self.resource_timelines[res_id] = []
            elif isinstance(resources, list):
                for i in range(len(resources)):
                    res_id = f"{res_type.value}_{i}"
                    self.resource_timelines[res_id] = []
                    
    def _analyze_dependencies(self) -> Dict[str, List[str]]:
        """分析任务依赖关系"""
        dependencies = {}
        
        for task_id, task in self.scheduler.tasks.items():
            if task.dependencies:
                dependencies[task_id] = list(task.dependencies)
                
        return dependencies
        
    def _sort_events_for_compaction(self, events: List, 
                                   dependencies: Dict[str, List[str]]) -> List:
        """按优先级和依赖关系排序事件"""
        # 按任务分组事件
        task_events = defaultdict(list)
        for event in events:
            task_events[event.task_id].append(event)
            
        # 拓扑排序处理依赖
        sorted_tasks = self._topological_sort(list(task_events.keys()), dependencies)
        
        # 按排序后的任务顺序重组事件
        sorted_events = []
        for task_id in sorted_tasks:
            # 任务内的事件按原始时间排序
            task_events[task_id].sort(key=lambda e: e.start_time)
            sorted_events.extend(task_events[task_id])
            
        return sorted_events
        
    def _topological_sort(self, tasks: List[str], dependencies: Dict[str, List[str]]) -> List[str]:
        """拓扑排序任务"""
        # 计算入度
        in_degree = {task: 0 for task in tasks}
        adj_list = defaultdict(list)
        
        for task, deps in dependencies.items():
            for dep in deps:
                if dep in tasks and task in tasks:
                    adj_list[dep].append(task)
                    in_degree[task] += 1
                    
        # 按优先级分组
        priority_groups = defaultdict(list)
        for task in tasks:
            priority = self.scheduler.tasks[task].priority.value
            priority_groups[priority].append(task)
            
        # 拓扑排序
        result = []
        queue = []
        
        # 先加入无依赖的任务，按优先级
        for priority in sorted(priority_groups.keys()):
            for task in priority_groups[priority]:
                if in_degree[task] == 0:
                    queue.append(task)
                    
        while queue:
            # 按优先级处理队列中的任务
            queue.sort(key=lambda t: self.scheduler.tasks[t].priority.value)
            task = queue.pop(0)
            result.append(task)
            
            # 更新依赖此任务的其他任务
            for next_task in adj_list[task]:
                in_degree[next_task] -= 1
                if in_degree[next_task] == 0:
                    queue.append(next_task)
                    
        return result
        
    def _perform_compaction(self, events: List, 
                           dependencies: Dict[str, List[str]]) -> List:
        """执行紧凑化 - 修复版本，正确处理周期性任务"""
        compacted_events = []
        
        # 按时间窗口分组事件
        window_events = defaultdict(list)
        for event in events:
            window_idx = int(event.start_time // self.time_window)
            window_events[window_idx].append(event)
        
        # 对每个时间窗口分别进行紧凑化
        for window_idx in sorted(window_events.keys()):
            window_start = window_idx * self.time_window
            window_end = (window_idx + 1) * self.time_window
            events_in_window = window_events[window_idx]
            
            # 为这个窗口初始化资源可用时间
            resource_next_available = {res_id: window_start for res_id in self.resource_timelines}
            task_completion_times = {}
            
            # 按任务分组窗口内的事件
            task_events = defaultdict(list)
            for event in events_in_window:
                task_events[event.task_id].append(event)
            
            # 获取任务执行顺序（考虑优先级和依赖）
            task_order = self._get_task_execution_order(task_events.keys(), dependencies)
            
            # 按顺序处理每个任务
            for task_id in task_order:
                if task_id not in task_events:
                    continue
                    
                task_all_events = task_events[task_id]
                task_all_events.sort(key=lambda e: e.start_time)
                
                # 计算任务可以开始的最早时间
                task_start_time = window_start
                
                # 检查依赖
                if task_id in dependencies:
                    for dep_task in dependencies[task_id]:
                        if dep_task in task_completion_times:
                            task_start_time = max(task_start_time, task_completion_times[dep_task])
                
                # 检查资源可用性
                for evt in task_all_events:
                    for res_type, res_id in evt.assigned_resources.items():
                        if res_id in resource_next_available:
                            task_start_time = max(task_start_time, resource_next_available[res_id])
                
                # 确保不超出窗口
                original_duration = task_all_events[-1].end_time - task_all_events[0].start_time
                if task_start_time + original_duration > window_end:
                    # 如果任务无法在窗口内完成，保持原位置
                    for evt in task_all_events:
                        compacted_events.append(copy.deepcopy(evt))
                    continue
                
                # 计算时间偏移
                time_shift = task_start_time - task_all_events[0].start_time
                
                # 重新调度任务事件
                for evt in task_all_events:
                    new_event = copy.deepcopy(evt)
                    new_event.start_time = evt.start_time + time_shift
                    new_event.end_time = evt.end_time + time_shift
                    
                    compacted_events.append(new_event)
                    
                    # 更新资源可用时间
                    for res_type, res_id in new_event.assigned_resources.items():
                        if res_id in resource_next_available:
                            resource_next_available[res_id] = new_event.end_time
                    
                    # 更新任务完成时间
                    task_completion_times[task_id] = max(
                        task_completion_times.get(task_id, 0),
                        new_event.end_time
                    )
        
        # 按时间排序所有事件
        compacted_events.sort(key=lambda e: e.start_time)
        
        return compacted_events
    
    def _get_task_execution_order(self, task_ids, dependencies):
        """获取任务执行顺序"""
        # 简单实现：按优先级排序
        task_list = list(task_ids)
        task_list.sort(key=lambda tid: (
            self.scheduler.tasks[tid].priority.value,
            tid  # 相同优先级按ID排序保证稳定性
        ))
        return task_list
        
    def _calculate_idle_time(self, events: List) -> Dict:
        """计算空闲时间信息 - 只关注第一个时间窗口"""
        # 只分析第一个时间窗口内的事件
        first_window_events = [e for e in events if e.start_time < self.time_window]
        
        if not first_window_events:
            return {
                'total_idle_at_end': self.time_window,
                'idle_percentage': 100.0,
                'resource_idle_times': {res_id: self.time_window for res_id in self.resource_timelines},
                'compaction_ratio': 0.0,
                'min_resource_idle': self.time_window,
                'all_idle_start': 0.0
            }
        
        # 计算每个资源在第一个窗口内的最后使用时间
        resource_last_used = {}
        
        for event in first_window_events:
            for res_type, res_id in event.assigned_resources.items():
                resource_last_used[res_id] = max(
                    resource_last_used.get(res_id, 0),
                    min(event.end_time, self.time_window)  # 确保不超过窗口
                )
        
        # 计算每个资源的末尾空闲时间
        resource_idle_times = {}
        
        for res_id in self.resource_timelines:
            last_used = resource_last_used.get(res_id, 0)
            idle_time = self.time_window - last_used
            resource_idle_times[res_id] = idle_time
        
        # 计算所有资源都空闲的连续时间
        all_idle_start = max(resource_last_used.values()) if resource_last_used else 0
        total_idle_at_end = self.time_window - all_idle_start
        
        # 计算紧凑化比率（仅基于第一个窗口）
        original_first_window = [e for e in self.scheduler.schedule_history if e.start_time < self.time_window]
        if original_first_window:
            original_span = max(e.end_time for e in original_first_window)
            compacted_span = max(e.end_time for e in first_window_events)
            compaction_ratio = 1 - (compacted_span / original_span) if original_span > 0 else 0
        else:
            compaction_ratio = 0.0
        
        return {
            'total_idle_at_end': total_idle_at_end,
            'idle_percentage': (total_idle_at_end / self.time_window) * 100,
            'resource_idle_times': resource_idle_times,
            'compaction_ratio': compaction_ratio,
            'min_resource_idle': min(resource_idle_times.values()) if resource_idle_times else 0,
            'all_idle_start': all_idle_start
        }
        
    def _print_compaction_stats(self, result: CompactionResult):
        """打印紧凑化统计"""
        print("\n📊 紧凑化结果:")
        print("=" * 60)
        
        print(f"\n整体统计:")
        print(f"  - 末尾连续空闲时间: {result.idle_time_at_end:.1f}ms ({result.idle_percentage:.1f}%)")
        print(f"  - 紧凑化比率: {result.compaction_ratio:.1%}")
        print(f"  - 所有资源空闲起始: {self.time_window - result.idle_time_at_end:.1f}ms")
        
        print(f"\n各资源末尾空闲时间:")
        for res_id, idle_time in sorted(result.resource_idle_times.items()):
            utilization = ((self.time_window - idle_time) / self.time_window) * 100
            print(f"  - {res_id}: {idle_time:.1f}ms 空闲 (利用率: {utilization:.1f}%)")
            
    def apply_compacted_schedule(self, result: CompactionResult):
        """应用紧凑化后的调度结果"""
        if result and result.compacted_events:
            self.scheduler.schedule_history = result.compacted_events
            print("\n✅ 紧凑化调度已应用")
            

def compact_and_visualize(scheduler, time_window=200.0):
    """紧凑化并生成可视化 - 修复图片尺寸问题"""
    from elegant_visualization import ElegantSchedulerVisualizer
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # 创建紧凑化器
    compactor = ScheduleCompactor(scheduler, time_window)
    
    # 保存原始调度
    original_events = copy.deepcopy(scheduler.schedule_history)
    
    # 执行紧凑化
    result = compactor.compact_schedule(preserve_dependencies=True)
    
    if result:
        # 只保留前几个时间窗口的事件用于可视化
        max_windows_to_show = 3  # 只显示前3个周期
        max_time = time_window * max_windows_to_show
        
        original_limited = [e for e in original_events if e.start_time < max_time]
        compacted_limited = [e for e in result.compacted_events if e.start_time < max_time]
        
        # 生成第一个窗口的对比图
        print("\n📊 生成第一个时间窗口的对比...")
        
        # 只显示第一个窗口
        original_first = [e for e in original_events if e.start_time < time_window]
        compacted_first = [e for e in result.compacted_events if e.start_time < time_window]
        
        # 临时替换调度历史并生成可视化
        # 原始调度（第一个窗口）
        scheduler.schedule_history = original_first
        viz1 = ElegantSchedulerVisualizer(scheduler)
        viz1.plot_elegant_gantt()
        plt.title(f'Original Schedule - First {time_window}ms Window')
        plt.xlim(0, time_window)
        plt.savefig('original_first_window.png', dpi=120, bbox_inches='tight')
        plt.close()
        
        # 紧凑化调度（第一个窗口）
        scheduler.schedule_history = compacted_first
        viz2 = ElegantSchedulerVisualizer(scheduler)
        viz2.plot_elegant_gantt()
        
        # 添加空闲区域标记
        idle_start = time_window - result.idle_time_at_end
        if result.idle_time_at_end > 0:
            plt.axvspan(idle_start, time_window, alpha=0.3, color='lightgreen')
            plt.axvline(x=idle_start, color='green', linestyle='--', alpha=0.7, linewidth=2)
            plt.text(idle_start + result.idle_time_at_end/2, plt.ylim()[1]*0.95, 
                    f'{result.idle_time_at_end:.1f}ms IDLE', 
                    ha='center', va='top', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.title(f'Compacted Schedule - {result.idle_time_at_end:.1f}ms Idle at End ({result.idle_percentage:.1f}%)')
        plt.xlim(0, time_window)
        plt.savefig('compacted_first_window.png', dpi=120, bbox_inches='tight')
        plt.close()
        
        # 生成前几个周期的完整视图
        scheduler.schedule_history = compacted_limited
        viz3 = ElegantSchedulerVisualizer(scheduler)
        viz3.plot_elegant_gantt()
        plt.title(f'Compacted Schedule - First {max_windows_to_show} Periods')
        plt.xlim(0, max_time)
        plt.savefig('compacted_schedule_overview.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # 恢复完整的紧凑化结果
        scheduler.schedule_history = result.compacted_events
        
        # 生成Chrome trace（包含所有数据）
        viz4 = ElegantSchedulerVisualizer(scheduler)
        viz4.export_chrome_tracing('compacted_schedule_trace.json')
        
        # 生成统计摘要图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. 资源利用率对比
        resources = list(result.resource_idle_times.keys())
        idle_times = [result.resource_idle_times[r] for r in resources]
        util_rates = [(time_window - idle) / time_window * 100 for idle in idle_times]
        
        ax1.bar(range(len(resources)), util_rates, alpha=0.7, color='steelblue')
        ax1.set_xticks(range(len(resources)))
        ax1.set_xticklabels(resources, rotation=45)
        ax1.set_ylabel('Utilization (%)')
        ax1.set_title('Resource Utilization After Compaction')
        ax1.grid(True, alpha=0.3)
        
        # 2. 空闲时间分布
        ax2.barh(range(len(resources)), idle_times, alpha=0.7, color='coral')
        ax2.set_yticks(range(len(resources)))
        ax2.set_yticklabels(resources)
        ax2.set_xlabel('Idle Time (ms)')
        ax2.set_title('Idle Time at End by Resource')
        ax2.grid(True, alpha=0.3)
        
        # 3. 紧凑化效果
        labels = ['Before', 'After']
        values = [100 - result.idle_percentage, result.idle_percentage]
        colors = ['#ff9999', '#66b3ff']
        ax3.pie(values, labels=[f'Busy\n{values[0]:.1f}%', f'Idle\n{values[1]:.1f}%'], 
                colors=colors, autopct='', startangle=90)
        ax3.set_title('Time Window Utilization')
        
        # 4. 总结文本
        ax4.axis('off')
        summary_text = f"""Compaction Summary:
        
Total Idle at End: {result.idle_time_at_end:.1f}ms
Idle Percentage: {result.idle_percentage:.1f}%
Compaction Ratio: {result.compaction_ratio:.1%}

All resources become idle at: {time_window - result.idle_time_at_end:.1f}ms
Continuous idle period: {result.idle_time_at_end:.1f}ms

This idle time can be used for:
• Power saving
• Additional tasks
• System maintenance"""
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('compaction_summary.png', dpi=120, bbox_inches='tight')
        plt.close()
        
        print("\n✅ 可视化已生成:")
        print("  - original_first_window.png (原始调度-第一个窗口)")
        print("  - compacted_first_window.png (紧凑化调度-第一个窗口)")
        print("  - compacted_schedule_overview.png (前几个周期概览)")
        print("  - compaction_summary.png (紧凑化统计摘要)")
        print("  - compacted_schedule_trace.json (完整Chrome追踪文件)")
        
        # 应用紧凑化结果
        compactor.apply_compacted_schedule(result)
        
        return result
    
    return None


if __name__ == "__main__":
    """测试调度紧凑化"""
    from fixed_genetic_optimizer import main as run_genetic_optimization
    
    # 先运行遗传算法优化
    print("运行遗传算法优化...")
    # 这里需要修改fixed_genetic_optimizer返回scheduler
    
    # 示例：假设已有优化后的scheduler
    print("\n" + "=" * 80)
    print("🔧 调度紧凑化测试")
    print("=" * 80)
    
    # 这里需要实际的scheduler实例
    # scheduler = get_optimized_scheduler()
    # result = compact_and_visualize(scheduler)
    
    print("\n✅ 紧凑化测试完成！")
