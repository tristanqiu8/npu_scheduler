from enum import Enum
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
import heapq
from collections import defaultdict

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
    
    def print_schedule_summary(self):
        """打印调度摘要"""
        print("=== 调度摘要 ===")
        print(f"NPU资源数: {len(self.resources[ResourceType.NPU])}")
        print(f"DSP资源数: {len(self.resources[ResourceType.DSP])}")
        print(f"任务总数: {len(self.tasks)}")
        print(f"已调度次数: {len(self.schedule_history)}")
        
        # 统计每个任务的调度情况
        task_schedule_count = defaultdict(int)
        task_latencies = defaultdict(list)
        
        for schedule in self.schedule_history:
            task_schedule_count[schedule.task_id] += 1
            task_latencies[schedule.task_id].append(schedule.actual_latency)
        
        print("\n任务调度详情:")
        for task_id, task in self.tasks.items():
            count = task_schedule_count[task_id]
            avg_latency = sum(task_latencies[task_id]) / len(task_latencies[task_id]) if task_latencies[task_id] else 0
            achieved_fps = count / (self.schedule_history[-1].end_time / 1000) if self.schedule_history else 0
            
            print(f"  {task}: ")
            print(f"    调度次数: {count}")
            print(f"    平均延时: {avg_latency:.1f}ms (需求: {task.latency_requirement}ms)")
            print(f"    实现FPS: {achieved_fps:.1f} (需求: {task.fps_requirement})")
        
        # 资源利用率
        if self.schedule_history:
            time_window = self.schedule_history[-1].end_time
            utilization = self.get_resource_utilization(time_window)
            print("\n资源利用率:")
            for resource_id, util in utilization.items():
                print(f"  {resource_id}: {util:.1f}%")


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
    
    # 执行调度
    print("开始调度...")
    schedule_results = scheduler.simple_schedule(time_window=500.0)
    
    # 打印结果
    scheduler.print_schedule_summary()
    
    # 打印前10个调度事件
    print("\n前10个调度事件:")
    for i, schedule in enumerate(schedule_results[:10]):
        task = scheduler.tasks[schedule.task_id]
        print(f"{i+1}. {task.name} @ {schedule.start_time:.1f}-{schedule.end_time:.1f}ms, "
              f"使用资源: {list(schedule.assigned_resources.values())}")