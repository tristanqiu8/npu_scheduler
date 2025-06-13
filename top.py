from enum import Enum
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
import heapq
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class ResourceType(Enum):
    """资源类型枚举"""
    DSP = "DSP"
    NPU = "NPU"

@dataclass
class ResourceSegment:
    """资源占用片段"""
    resource_type: ResourceType
    duration_table: Dict[float, float]  # {BW: duration} 查找表
    start_time: float  # 相对于任务开始的时间
    
    def get_duration(self, bw: float) -> float:
        """根据带宽获取执行时间"""
        if bw in self.duration_table:
            return self.duration_table[bw]
        closest_bw = min(self.duration_table.keys(), key=lambda x: abs(x - bw))
        return self.duration_table[closest_bw]

@dataclass
class ResourceUnit:
    """资源单元（NPU或DSP实例）"""
    unit_id: str
    resource_type: ResourceType
    bandwidth: float  # 该单元的带宽能力
    
    def __hash__(self):
        return hash(self.unit_id)

@dataclass
class TaskScheduleInfo:
    """任务调度信息"""
    task_id: str
    start_time: float
    end_time: float
    assigned_resources: Dict[ResourceType, str]  # {资源类型: 分配的资源ID}
    actual_latency: float  # 从调度开始到完成的实际延时

class NNTask:
    """神经网络任务类"""
    
    def __init__(self, task_id: str, name: str = ""):
        self.task_id = task_id
        self.name = name or f"Task_{task_id}"
        self.segments: List[ResourceSegment] = []
        self.dependencies: Set[str] = set()
        self.fps_requirement: float = 30.0
        self.latency_requirement: float = 100.0
        
        # 调度相关信息
        self.schedule_info: Optional[TaskScheduleInfo] = None
        self.last_execution_time: float = -float('inf')  # 上次执行时间
    
    def set_npu_only(self, duration_table: Dict[float, float]):
        """设置为仅使用NPU的任务"""
        self.segments = [ResourceSegment(ResourceType.NPU, duration_table, 0)]
        
    def set_dsp_npu_sequence(self, segments: List[Tuple[ResourceType, Dict[float, float], float]]):
        """设置为DSP+NPU混合执行的任务"""
        self.segments = []
        for resource_type, duration_table, start_time in segments:
            self.segments.append(ResourceSegment(resource_type, duration_table, start_time))
    
    def add_dependency(self, task_id: str):
        self.dependencies.add(task_id)
    
    def add_dependencies(self, task_ids: List[str]):
        self.dependencies.update(task_ids)
    
    def set_performance_requirements(self, fps: float, latency: float):
        self.fps_requirement = fps
        self.latency_requirement = latency
    
    def get_total_duration(self, resource_bw_map: Dict[ResourceType, float]) -> float:
        """根据分配的资源带宽获取总执行时间"""
        if not self.segments:
            return 0
        max_end_time = 0
        for seg in self.segments:
            bw = resource_bw_map.get(seg.resource_type, 1.0)
            end_time = seg.start_time + seg.get_duration(bw)
            max_end_time = max(max_end_time, end_time)
        return max_end_time
    
    @property
    def min_interval_ms(self) -> float:
        """根据FPS需求计算的最小调度间隔"""
        return 1000.0 / self.fps_requirement if self.fps_requirement > 0 else float('inf')
    
    @property
    def uses_dsp(self) -> bool:
        return any(seg.resource_type == ResourceType.DSP for seg in self.segments)
    
    @property
    def uses_npu(self) -> bool:
        return any(seg.resource_type == ResourceType.NPU for seg in self.segments)
    
    def __repr__(self):
        sched_str = f", scheduled@{self.schedule_info.start_time:.1f}ms" if self.schedule_info else ""
        return f"Task{self.task_id}({self.name}, fps={self.fps_requirement}{sched_str})"


class MultiResourceScheduler:
    """多资源调度器，支持多NPU和多DSP"""
    
    def __init__(self):
        self.tasks: Dict[str, NNTask] = {}
        self.resources: Dict[ResourceType, List[ResourceUnit]] = {
            ResourceType.NPU: [],
            ResourceType.DSP: []
        }
        self.resource_availability: Dict[str, float] = {}  # 资源ID -> 最早可用时间
        self.schedule_history: List[TaskScheduleInfo] = []
        
    def add_npu(self, npu_id: str, bandwidth: float):
        """添加NPU资源"""
        npu = ResourceUnit(npu_id, ResourceType.NPU, bandwidth)
        self.resources[ResourceType.NPU].append(npu)
        self.resource_availability[npu_id] = 0.0
        
    def add_dsp(self, dsp_id: str, bandwidth: float):
        """添加DSP资源"""
        dsp = ResourceUnit(dsp_id, ResourceType.DSP, bandwidth)
        self.resources[ResourceType.DSP].append(dsp)
        self.resource_availability[dsp_id] = 0.0
    
    def add_task(self, task: NNTask):
        """添加任务"""
        self.tasks[task.task_id] = task
    
    def get_earliest_available_resource(self, resource_type: ResourceType, 
                                      start_after: float = 0) -> Tuple[str, float, float]:
        """获取最早可用的资源
        
        Returns:
            (资源ID, 可用时间, 带宽)
        """
        resources = self.resources[resource_type]
        if not resources:
            raise ValueError(f"No {resource_type.value} resources available")
        
        earliest_time = float('inf')
        selected_resource = None
        selected_bw = 0
        
        for resource in resources:
            available_time = max(self.resource_availability[resource.unit_id], start_after)
            if available_time < earliest_time:
                earliest_time = available_time
                selected_resource = resource.unit_id
                selected_bw = resource.bandwidth
        
        return selected_resource, earliest_time, selected_bw
    
    def schedule_task(self, task: NNTask, current_time: float) -> Optional[TaskScheduleInfo]:
        """调度单个任务
        
        Args:
            task: 要调度的任务
            current_time: 当前时间
            
        Returns:
            调度信息，如果无法调度则返回None
        """
        # 检查依赖是否满足
        earliest_start = current_time
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if dep_task and dep_task.schedule_info:
                earliest_start = max(earliest_start, dep_task.schedule_info.end_time)
            elif dep_task and not dep_task.schedule_info:
                return None  # 依赖任务还未调度
        
        # 检查FPS约束
        min_interval = task.min_interval_ms
        if task.last_execution_time + min_interval > earliest_start:
            earliest_start = task.last_execution_time + min_interval
        
        # 为每个资源段分配资源
        assigned_resources = {}
        segment_schedules = []
        task_start_time = earliest_start
        
        for seg in task.segments:
            # 获取该类型资源的最早可用时间
            resource_id, available_time, bw = self.get_earliest_available_resource(
                seg.resource_type, 
                task_start_time + seg.start_time
            )
            
            seg_start = max(available_time, task_start_time + seg.start_time)
            seg_duration = seg.get_duration(bw)
            seg_end = seg_start + seg_duration
            
            assigned_resources[seg.resource_type] = resource_id
            segment_schedules.append((resource_id, seg_start, seg_end))
        
        # 计算任务实际开始和结束时间
        actual_start = min(s[1] for s in segment_schedules)
        actual_end = max(s[2] for s in segment_schedules)
        
        # 更新资源可用时间
        for resource_id, seg_start, seg_end in segment_schedules:
            self.resource_availability[resource_id] = seg_end
        
        # 创建调度信息
        schedule_info = TaskScheduleInfo(
            task_id=task.task_id,
            start_time=actual_start,
            end_time=actual_end,
            assigned_resources=assigned_resources,
            actual_latency=actual_end - current_time
        )
        
        task.schedule_info = schedule_info
        task.last_execution_time = actual_start
        self.schedule_history.append(schedule_info)
        
        return schedule_info
    
    def simple_schedule(self, time_window: float = 1000.0) -> List[TaskScheduleInfo]:
        """简单调度算法：按优先级和依赖关系调度
        
        Args:
            time_window: 调度时间窗口（ms）
            
        Returns:
            调度结果列表
        """
        # 重置调度状态
        for resource_id in self.resource_availability:
            self.resource_availability[resource_id] = 0.0
        for task in self.tasks.values():
            task.schedule_info = None
            task.last_execution_time = -float('inf')
        self.schedule_history.clear()
        
        current_time = 0.0
        scheduled_count = 0
        
        # 创建任务优先级队列（基于FPS需求）
        task_queue = []
        for task in self.tasks.values():
            priority = -task.fps_requirement  # 负数使得FPS高的优先
            heapq.heappush(task_queue, (priority, task.task_id))
        
        # 模拟调度过程
        while current_time < time_window and task_queue:
            # 尝试调度所有待调度任务
            temp_queue = []
            any_scheduled = False
            
            while task_queue:
                priority, task_id = heapq.heappop(task_queue)
                task = self.tasks[task_id]
                
                # 检查是否需要再次调度（基于FPS）
                if task.last_execution_time + task.min_interval_ms <= current_time:
                    schedule_info = self.schedule_task(task, current_time)
                    if schedule_info:
                        scheduled_count += 1
                        any_scheduled = True
                        # 任务可能需要再次调度
                        heapq.heappush(temp_queue, (priority, task_id))
                    else:
                        # 无法调度（依赖未满足），稍后重试
                        heapq.heappush(temp_queue, (priority, task_id))
                else:
                    # 还不到调度时间
                    heapq.heappush(temp_queue, (priority, task_id))
            
            task_queue = temp_queue
            
            # 推进时间
            if not any_scheduled:
                # 找到下一个事件时间
                next_time = current_time + 1.0
                # 检查资源释放时间
                for available_time in self.resource_availability.values():
                    if available_time > current_time:
                        next_time = min(next_time, available_time)
                # 检查任务可调度时间
                for task in self.tasks.values():
                    next_schedule_time = task.last_execution_time + task.min_interval_ms
                    if next_schedule_time > current_time:
                        next_time = min(next_time, next_schedule_time)
                
                current_time = next_time
            
        return self.schedule_history
    
    def get_resource_utilization(self, time_window: float) -> Dict[str, float]:
        """计算资源利用率"""
        utilization = {}
        
        for resource_type, resources in self.resources.items():
            for resource in resources:
                busy_time = 0.0
                for schedule in self.schedule_history:
                    if resource.unit_id in schedule.assigned_resources.values():
                        # 简化计算：假设资源在整个任务期间都被占用
                        busy_time += (schedule.end_time - schedule.start_time)
                
                utilization[resource.unit_id] = min(busy_time / time_window * 100, 100)
        
        return utilization
    
    def plot_task_overview(self, selected_bw: float = 4.0):
        """Plot task overview showing resource requirements and performance needs"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 准备数据
        task_ids = []
        task_names = []
        fps_requirements = []
        latency_requirements = []
        task_durations = []
        task_types = []  # NPU-only or DSP+NPU
        dependencies_str = []
        
        for task_id, task in self.tasks.items():
            task_ids.append(task_id)
            task_names.append(task.name)
            fps_requirements.append(task.fps_requirement)
            latency_requirements.append(task.latency_requirement)
            
            # 计算在选定带宽下的执行时间
            resource_bw_map = {ResourceType.NPU: selected_bw, ResourceType.DSP: selected_bw}
            duration = task.get_total_duration(resource_bw_map)
            task_durations.append(duration)
            
            # 确定任务类型
            if task.uses_dsp and task.uses_npu:
                task_types.append('DSP+NPU')
            elif task.uses_npu:
                task_types.append('NPU-only')
            else:
                task_types.append('DSP-only')
            
            # 依赖关系
            dep_str = ','.join(task.dependencies) if task.dependencies else 'None'
            dependencies_str.append(dep_str)
        
        # 图1：任务性能需求
        x = np.arange(len(task_ids))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, fps_requirements, width, label='FPS Requirement', color='skyblue')
        bars2 = ax1.bar(x + width/2, latency_requirements, width, label='Latency Requirement (ms)', color='lightcoral')
        
        ax1.set_xlabel('Task')
        ax1.set_ylabel('Value')
        ax1.set_title(f'Task Performance Requirements Overview (BW={selected_bw})')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{tid}\n{name}' for tid, name in zip(task_ids, task_names)], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 在柱子上添加数值
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=8)
        
        # 图2：任务执行时间和类型
        colors = {'NPU-only': 'green', 'DSP+NPU': 'orange', 'DSP-only': 'blue'}
        bar_colors = [colors.get(t, 'gray') for t in task_types]
        
        bars3 = ax2.bar(x, task_durations, color=bar_colors)
        
        ax2.set_xlabel('Task')
        ax2.set_ylabel('Execution Time (ms)')
        ax2.set_title(f'Task Execution Time and Resource Type (BW={selected_bw})')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{tid}\n{dep}' for tid, dep in zip(task_ids, dependencies_str)], 
                           rotation=45, ha='right')
        
        # 添加图例
        legend_elements = [patches.Patch(color=color, label=task_type) 
                          for task_type, color in colors.items()]
        ax2.legend(handles=legend_elements, loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 在柱子上添加数值
        for bar, duration in zip(bars3, task_durations):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{duration:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def plot_pipeline_schedule(self, time_window: float = None, show_first_n: int = None):
        """Plot pipeline schedule Gantt chart"""
        if not self.schedule_history:
            print("No schedule history, please run scheduling algorithm first")
            return
        
        # 确定时间窗口
        if time_window is None:
            time_window = max(s.end_time for s in self.schedule_history) * 1.1
        
        # 准备资源列表
        all_resources = []
        resource_types = []
        for res_type in [ResourceType.NPU, ResourceType.DSP]:
            for resource in self.resources[res_type]:
                all_resources.append(resource.unit_id)
                resource_types.append(res_type.value)
        
        # 创建资源索引映射
        resource_to_y = {res_id: i for i, res_id in enumerate(all_resources)}
        
        # 准备颜色映射（每个任务一个颜色）
        task_colors = {}
        color_palette = plt.cm.Set3(np.linspace(0, 1, len(self.tasks)))
        for i, task_id in enumerate(self.tasks.keys()):
            task_colors[task_id] = color_palette[i]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(14, max(6, len(all_resources) * 0.8)))
        
        # 绘制调度块
        schedules_to_plot = self.schedule_history[:show_first_n] if show_first_n else self.schedule_history
        
        for schedule in schedules_to_plot:
            task = self.tasks[schedule.task_id]
            
            # 对于每个任务段，绘制相应的资源占用
            for seg in task.segments:
                if seg.resource_type in schedule.assigned_resources:
                    resource_id = schedule.assigned_resources[seg.resource_type]
                    if resource_id in resource_to_y:
                        y_pos = resource_to_y[resource_id]
                        
                        # 计算该段的实际执行时间
                        resource_unit = next((r for r in self.resources[seg.resource_type] 
                                            if r.unit_id == resource_id), None)
                        if resource_unit:
                            duration = seg.get_duration(resource_unit.bandwidth)
                            start_time = schedule.start_time + seg.start_time
                            
                            # 绘制矩形
                            rect = patches.Rectangle(
                                (start_time, y_pos - 0.4), duration, 0.8,
                                linewidth=1, edgecolor='black',
                                facecolor=task_colors[schedule.task_id],
                                alpha=0.8
                            )
                            ax.add_patch(rect)
                            
                            # 添加任务标签
                            if duration > 5:  # 只在足够宽的块上添加标签
                                ax.text(start_time + duration/2, y_pos,
                                       f'{task.task_id}', 
                                       ha='center', va='center', fontsize=8,
                                       weight='bold')
        
        # 设置坐标轴
        ax.set_ylim(-0.5, len(all_resources) - 0.5)
        ax.set_xlim(0, time_window)
        ax.set_yticks(range(len(all_resources)))
        ax.set_yticklabels([f'{res_id}\n({res_type})' for res_id, res_type 
                           in zip(all_resources, resource_types)])
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Resource')
        ax.set_title('Task Scheduling Gantt Chart')
        ax.grid(True, axis='x', alpha=0.3)
        
        # 添加资源类型分隔线
        current_type = resource_types[0]
        for i, res_type in enumerate(resource_types[1:], 1):
            if res_type != current_type:
                ax.axhline(y=i-0.5, color='red', linestyle='--', linewidth=2)
                current_type = res_type
        
        # 添加图例
        legend_elements = []
        for task_id, task in self.tasks.items():
            legend_elements.append(
                patches.Patch(color=task_colors[task_id], 
                            label=f'{task_id}: {task.name}')
            )
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Add resource utilization info
        utilization = self.get_resource_utilization(time_window)
        util_text = "Resource Utilization:\n"
        for res_id, util in utilization.items():
            util_text += f"{res_id}: {util:.1f}%\n"
        
        ax.text(1.02, 0.02, util_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 创建调度器
    scheduler = MultiResourceScheduler()
    
    # 添加多个NPU资源（不同带宽）
    scheduler.add_npu("NPU_0", bandwidth=8.0)  # 高性能NPU
    scheduler.add_npu("NPU_1", bandwidth=4.0)  # 中等性能NPU
    scheduler.add_npu("NPU_2", bandwidth=2.0)  # 低性能NPU
    
    # 添加DSP资源
    scheduler.add_dsp("DSP_0", bandwidth=4.0)
    scheduler.add_dsp("DSP_1", bandwidth=4.0)
    
    # 创建多个任务
    # 高优先级任务（高FPS需求）
    task1 = NNTask("T1", "Detection")
    task1.set_npu_only({2.0: 30, 4.0: 20, 8.0: 12})
    task1.set_performance_requirements(fps=30, latency=40)
    scheduler.add_task(task1)
    
    # 中等优先级任务
    task2 = NNTask("T2", "Tracking")
    task2.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 10, 8.0: 8}, 0),
        (ResourceType.NPU, {2.0: 25, 4.0: 15, 8.0: 10}, 10),
    ])
    task2.set_performance_requirements(fps=15, latency=60)
    scheduler.add_task(task2)
    
    # 复杂任务（有依赖）
    task3 = NNTask("T3", "Analysis")
    task3.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 8}, 0),
        (ResourceType.NPU, {2.0: 20, 4.0: 12, 8.0: 8}, 8),
        (ResourceType.DSP, {4.0: 5}, 20),
    ])
    task3.add_dependency("T1")  # 依赖T1
    task3.set_performance_requirements(fps=10, latency=80)
    scheduler.add_task(task3)
    
    # 低优先级批处理任务
    task4 = NNTask("T4", "Background")
    task4.set_npu_only({2.0: 40, 4.0: 25, 8.0: 15})
    task4.set_performance_requirements(fps=5, latency=200)
    scheduler.add_task(task4)
    
    # 更多任务以展示多任务调度
    for i in range(5, 10):
        task = NNTask(f"T{i}", f"Task_{i}")
        task.set_npu_only({2.0: 20+i*5, 4.0: 15+i*3, 8.0: 10+i*2})
        task.set_performance_requirements(fps=10+i, latency=100)
        scheduler.add_task(task)
    
    # Execute scheduling
    print("Starting scheduling...")
    schedule_results = scheduler.simple_schedule(time_window=500.0)
    
    # Print results
    # scheduler.print_schedule_summary()
    
    # Plot task overview
    print("\nPlotting task overview...")
    scheduler.plot_task_overview(selected_bw=4.0)
    
    # Plot scheduling Gantt chart
    print("\nPlotting scheduling Gantt chart...")
    scheduler.plot_pipeline_schedule(time_window=200.0)  # Show first 200ms
    
    # Print first 10 scheduling events
    print("\nFirst 10 scheduling events:")
    for i, schedule in enumerate(schedule_results[:10]):
        task = scheduler.tasks[schedule.task_id]
        print(f"{i+1}. {task.name} @ {schedule.start_time:.1f}-{schedule.end_time:.1f}ms, "
              f"Resources used: {list(schedule.assigned_resources.values())}")