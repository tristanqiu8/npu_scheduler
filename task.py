from typing import List, Dict, Set, Optional, Tuple
from enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from models import ResourceSegment, TaskScheduleInfo, SubSegment, SegmentationDecision

class NNTask:
    """Enhanced neural network task class with network segmentation support"""
    
    def __init__(self, task_id: str, name: str = "", priority: TaskPriority = TaskPriority.NORMAL, 
                 runtime_type: RuntimeType = RuntimeType.ACPU_RUNTIME,
                 segmentation_strategy: SegmentationStrategy = SegmentationStrategy.ADAPTIVE_SEGMENTATION):
        self.task_id = task_id
        self.name = name or f"Task_{task_id}"
        self.priority = priority  # Task priority level
        self.runtime_type = runtime_type  # Runtime configuration
        self.segmentation_strategy = segmentation_strategy  # How to handle network cutting
        
        self.segments: List[ResourceSegment] = []
        self.dependencies: Set[str] = set()
        self.fps_requirement: float = 30.0
        self.latency_requirement: float = 100.0
        
        # Scheduling related info
        self.schedule_info: Optional[TaskScheduleInfo] = None
        self.last_execution_time: float = -float('inf')
        self.ready_time: float = 0  # Time when task becomes ready (dependencies satisfied)
        
        # Network segmentation related
        self.segmentation_decisions: List[SegmentationDecision] = []
        self.current_segmentation: Dict[str, List[str]] = {}  # {segment_id: [enabled_cut_ids]}
        self.total_segmentation_overhead: float = 0.0
    
    def set_npu_only(self, duration_table: Dict[float, float], segment_id: str = "npu_seg"):
        """Set as NPU-only task with optional cutting support"""
        segment = ResourceSegment(ResourceType.NPU, duration_table, 0, segment_id)
        self.segments = [segment]
        
    def set_dsp_npu_sequence(self, segments: List[Tuple[ResourceType, Dict[float, float], float, str]]):
        """Set as DSP+NPU mixed execution task with segment IDs"""
        self.segments = []
        for i, (resource_type, duration_table, start_time, segment_id) in enumerate(segments):
            if not segment_id:
                segment_id = f"{resource_type.value.lower()}_seg_{i}"
            segment = ResourceSegment(resource_type, duration_table, start_time, segment_id)
            self.segments.append(segment)
    
    def add_cut_points_to_segment(self, segment_id: str, cut_points: List[Tuple[str, float, float]]):
        """Add cut points to a specific segment
        Args:
            segment_id: ID of the segment to add cuts to
            cut_points: List of (op_id, position, overhead_ms) tuples
        """
        segment = self.get_segment_by_id(segment_id)
        if segment:
            for op_id, position, overhead_ms in cut_points:
                segment.add_cut_point(op_id, position, overhead_ms)
        else:
            raise ValueError(f"Segment {segment_id} not found in task {self.task_id}")
    
    def get_segment_by_id(self, segment_id: str) -> Optional[ResourceSegment]:
        """Get segment by its ID"""
        return next((seg for seg in self.segments if seg.segment_id == segment_id), None)
    
    def get_all_available_cuts(self) -> Dict[str, List[str]]:
        """Get all available cut points for all segments"""
        available_cuts = {}
        for segment in self.segments:
            available_cuts[segment.segment_id] = segment.get_available_cuts()
        return available_cuts
    
    def apply_segmentation_decision(self, decisions: Dict[str, List[str]]) -> float:
        """Apply segmentation decisions and return total overhead
        Args:
            decisions: {segment_id: [enabled_cut_ids]}
        Returns:
            Total segmentation overhead in ms
        """
        total_overhead = 0.0
        self.current_segmentation = decisions.copy()
        
        for segment in self.segments:
            enabled_cuts = decisions.get(segment.segment_id, [])
            segment.apply_segmentation(enabled_cuts)
            total_overhead += segment.segmentation_overhead
        
        self.total_segmentation_overhead = total_overhead
        return total_overhead
    
    def get_optimal_segmentation(self, available_resources: Dict[ResourceType, List[float]], 
                               current_time: float = 0.0) -> Dict[str, List[str]]:
        """Determine optimal segmentation based on available resources and strategy"""
        if self.segmentation_strategy == SegmentationStrategy.NO_SEGMENTATION:
            return {seg.segment_id: [] for seg in self.segments}
        
        elif self.segmentation_strategy == SegmentationStrategy.FORCED_SEGMENTATION:
            return {seg.segment_id: seg.get_available_cuts() for seg in self.segments}
        
        elif self.segmentation_strategy == SegmentationStrategy.ADAPTIVE_SEGMENTATION:
            return self._calculate_adaptive_segmentation(available_resources, current_time)
        
        else:  # CUSTOM_SEGMENTATION or others
            return self.current_segmentation
    
    def _calculate_adaptive_segmentation(self, available_resources: Dict[ResourceType, List[float]], 
                                       current_time: float) -> Dict[str, List[str]]:
        """Calculate adaptive segmentation based on resource availability and task requirements"""
        decisions = {}
        
        for segment in self.segments:
            segment_decisions = []
            available_cuts = segment.get_available_cuts()
            
            if not available_cuts:
                decisions[segment.segment_id] = []
                continue
            
            # Get available bandwidths for this resource type
            available_bw = available_resources.get(segment.resource_type, [])
            if not available_bw:
                decisions[segment.segment_id] = []
                continue
            
            # Calculate benefit of segmentation
            base_duration = segment.get_duration(max(available_bw))
            
            # Simple heuristic: use cuts if they can potentially improve parallelism
            # and the overhead is acceptable relative to task latency requirement
            max_acceptable_overhead = self.latency_requirement * 0.1  # 10% of latency requirement
            
            # Calculate potential cuts that don't exceed overhead threshold
            potential_cuts = []
            total_overhead = 0.0
            
            for cut_point in segment.cut_points:
                if total_overhead + cut_point.overhead_ms <= max_acceptable_overhead:
                    potential_cuts.append(cut_point.op_id)
                    total_overhead += cut_point.overhead_ms
                else:
                    break  # Stop adding cuts if overhead becomes too high
            
            # Additional heuristic: prefer cutting for high-priority tasks in resource-constrained scenarios
            if self.priority.value <= 1:  # CRITICAL or HIGH priority
                # More aggressive cutting for high-priority tasks
                segment_decisions = potential_cuts
            elif len(available_bw) > 1:
                # Only cut if multiple bandwidths available (can benefit from parallelism)
                segment_decisions = potential_cuts[:len(potential_cuts)//2]  # Use half of potential cuts
            else:
                # Conservative cutting for low priority or limited resources
                segment_decisions = potential_cuts[:1] if potential_cuts else []
            
            decisions[segment.segment_id] = segment_decisions
        
        return decisions
    
    def get_total_duration_with_segmentation(self, resource_bw_map: Dict[ResourceType, float], 
                                           segmentation_decisions: Optional[Dict[str, List[str]]] = None) -> float:
        """Get total execution time including segmentation overhead"""
        if segmentation_decisions is None:
            segmentation_decisions = self.current_segmentation
        
        max_end_time = 0.0
        for segment in self.segments:
            enabled_cuts = segmentation_decisions.get(segment.segment_id, [])
            bw = resource_bw_map.get(segment.resource_type, 1.0)
            duration = segment.get_total_duration_with_cuts(bw, enabled_cuts)
            end_time = segment.start_time + duration
            max_end_time = max(max_end_time, end_time)
        
        return max_end_time
    
    def get_sub_segments_for_scheduling(self) -> List[SubSegment]:
        """Get all sub-segments created by current segmentation for scheduling"""
        all_sub_segments = []
        for segment in self.segments:
            if segment.is_segmented:
                all_sub_segments.extend(segment.sub_segments)
            else:
                # Create a single sub-segment for non-segmented segments
                sub_seg = SubSegment(
                    sub_id=f"{segment.segment_id}_0",
                    resource_type=segment.resource_type,
                    duration_table=segment.duration_table.copy(),
                    start_time=segment.start_time,
                    original_segment_id=segment.segment_id
                )
                all_sub_segments.append(sub_seg)
        
        return all_sub_segments
    
    def add_dependency(self, task_id: str):
        self.dependencies.add(task_id)
    
    def add_dependencies(self, task_ids: List[str]):
        self.dependencies.update(task_ids)
    
    def set_performance_requirements(self, fps: float, latency: float):
        self.fps_requirement = fps
        self.latency_requirement = latency
    
    def set_runtime_type(self, runtime_type: RuntimeType):
        """Set runtime configuration type"""
        self.runtime_type = runtime_type
    
    def set_segmentation_strategy(self, strategy: SegmentationStrategy):
        """Set network segmentation strategy"""
        self.segmentation_strategy = strategy
    
    def get_total_duration(self, resource_bw_map: Dict[ResourceType, float]) -> float:
        """Get total execution time based on assigned resource bandwidths (legacy method)"""
        return self.get_total_duration_with_segmentation(resource_bw_map)
    
    def requires_resource_binding(self) -> bool:
        """Check if task requires resource binding (DSP_Runtime with multiple resource types)"""
        return (self.runtime_type == RuntimeType.DSP_RUNTIME and 
                len(set(seg.resource_type for seg in self.segments)) > 1)
    
    def get_segmentation_summary(self) -> str:
        """Get a summary of current segmentation decisions"""
        if not self.current_segmentation:
            return "No segmentation applied"
        
        summary_parts = []
        for segment_id, cuts in self.current_segmentation.items():
            if cuts:
                summary_parts.append(f"{segment_id}: {len(cuts)} cuts ({', '.join(cuts)})")
            else:
                summary_parts.append(f"{segment_id}: no cuts")
        
        overhead_info = f"Total overhead: {self.total_segmentation_overhead:.2f}ms"
        return f"{'; '.join(summary_parts)}. {overhead_info}"
    
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
    
    @property
    def is_segmented(self) -> bool:
        """Check if any segment is currently segmented"""
        return any(seg.is_segmented for seg in self.segments)
    
    def __repr__(self):
        sched_str = f", scheduled@{self.schedule_info.start_time:.1f}ms" if self.schedule_info else ""
        seg_str = f", segmented" if self.is_segmented else ""
        return (f"Task{self.task_id}({self.name}, {self.runtime_type.value}, "
                f"priority={self.priority.name}, fps={self.fps_requirement}"
                f"{sched_str}{seg_str})")