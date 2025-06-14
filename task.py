from typing import List, Dict, Set, Optional, Tuple
from enums import ResourceType, TaskPriority
from models import ResourceSegment, TaskScheduleInfo

class NNTask:
    """Neural network task class"""
    
    def __init__(self, task_id: str, name: str = "", priority: TaskPriority = TaskPriority.NORMAL):
        self.task_id = task_id
        self.name = name or f"Task_{task_id}"
        self.priority = priority  # Task priority level
        self.segments: List[ResourceSegment] = []
        self.dependencies: Set[str] = set()
        self.fps_requirement: float = 30.0
        self.latency_requirement: float = 100.0
        
        # Scheduling related info
        self.schedule_info: Optional[TaskScheduleInfo] = None
        self.last_execution_time: float = -float('inf')
        self.ready_time: float = 0  # Time when task becomes ready (dependencies satisfied)
    
    def set_npu_only(self, duration_table: Dict[float, float]):
        """Set as NPU-only task"""
        self.segments = [ResourceSegment(ResourceType.NPU, duration_table, 0)]
        
    def set_dsp_npu_sequence(self, segments: List[Tuple[ResourceType, Dict[float, float], float]]):
        """Set as DSP+NPU mixed execution task"""
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
        """Get total execution time based on assigned resource bandwidths"""
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
        """Minimum scheduling interval based on FPS requirement"""
        return 1000.0 / self.fps_requirement if self.fps_requirement > 0 else float('inf')
    
    @property
    def uses_dsp(self) -> bool:
        return any(seg.resource_type == ResourceType.DSP for seg in self.segments)
    
    @property
    def uses_npu(self) -> bool:
        return any(seg.resource_type == ResourceType.NPU for seg in self.segments)
    
    def __repr__(self):
        sched_str = f", scheduled@{self.schedule_info.start_time:.1f}ms" if self.schedule_info else ""
        return f"Task{self.task_id}({self.name}, priority={self.priority.name}, fps={self.fps_requirement}{sched_str})"
