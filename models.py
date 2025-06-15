from typing import Dict, Optional, Set
from dataclasses import dataclass
from enums import ResourceType, RuntimeType

@dataclass
class ResourceSegment:
    """Resource usage segment"""
    resource_type: ResourceType
    duration_table: Dict[float, float]  # {BW: duration} lookup table
    start_time: float  # Start time relative to task beginning
    
    def get_duration(self, bw: float) -> float:
        """Get execution time based on bandwidth"""
        if bw in self.duration_table:
            return self.duration_table[bw]
        closest_bw = min(self.duration_table.keys(), key=lambda x: abs(x - bw))
        return self.duration_table[closest_bw]

@dataclass
class ResourceUnit:
    """Resource unit (NPU or DSP instance)"""
    unit_id: str
    resource_type: ResourceType
    bandwidth: float  # Bandwidth capability of this unit
    
    def __hash__(self):
        return hash(self.unit_id)

@dataclass
class TaskScheduleInfo:
    """Task scheduling information"""
    task_id: str
    start_time: float
    end_time: float
    assigned_resources: Dict[ResourceType, str]  # {resource type: assigned resource ID}
    actual_latency: float  # Actual latency from scheduling to completion
    runtime_type: RuntimeType  # Runtime configuration used

@dataclass
class ResourceBinding:
    """Resource binding information for DSP_Runtime tasks"""
    task_id: str
    bound_resources: Set[str]  # Set of resource IDs that are bound together
    binding_start: float
    binding_end: float