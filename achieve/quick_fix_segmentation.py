#!/usr/bin/env python3
"""
Quick fix for segmentation with buffer and scheduling cost - Python 3.12 compatible
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from enums import ResourceType, TaskPriority, RuntimeType
from models import TaskScheduleInfo, ResourceBinding, SubSegment

# Configuration constants
SEGMENT_BUFFER_MS = 0.1      # Buffer between segments to avoid conflicts
SCHEDULING_COST_MS = 0.12    # Fixed cost for each segment scheduling


class SimpleSegmentationConfig:
    """Simple configuration class for segmentation"""
    
    def __init__(self, buffer_ms: float = 0.1, cost_ms: float = 0.12):
        self.segment_buffer_ms = buffer_ms
        self.scheduling_cost_ms = cost_ms
        self.enable_dynamic_buffer = True
        
        # Priority-based buffer multipliers
        self.priority_multipliers = {
            TaskPriority.CRITICAL: 0.5,
            TaskPriority.HIGH: 1.0,
            TaskPriority.NORMAL: 1.5,
            TaskPriority.LOW: 2.0
        }
    
    def get_buffer_for_task(self, task) -> float:
        """Get buffer based on task priority"""
        base_buffer = self.segment_buffer_ms
        if self.enable_dynamic_buffer:
            multiplier = self.priority_multipliers.get(task.priority, 1.0)
            base_buffer *= multiplier
        return base_buffer
    
    def get_cost_for_task(self, task, num_segments: int = 1) -> float:
        """Get scheduling cost"""
        return self.scheduling_cost_ms * num_segments


def apply_quick_segmentation_fix(scheduler, buffer_ms: float = 0.1, cost_ms: float = 0.12):
    """
    Quick fix for segmentation - adds buffer and scheduling cost
    """
    
    config = SimpleSegmentationConfig(buffer_ms, cost_ms)
    scheduler._quick_seg_config = config
    
    # Store original methods
    original_find_resources = getattr(scheduler, 'find_available_resources_for_task_with_segmentation', None)
    original_schedule = getattr(scheduler, 'priority_aware_schedule_with_segmentation', None)
    
    def enhanced_find_resources(task, current_time):
        """Enhanced resource finding with timing consideration"""
        if original_find_resources:
            return original_find_resources(task, current_time)
        else:
            # Simple fallback
            return find_basic_resources(scheduler, task, current_time)
    
    def enhanced_scheduling(time_window: float = 1000.0) -> List[TaskScheduleInfo]:
        """Enhanced scheduling with buffer and cost"""
        
        if original_schedule:
            # Call original scheduling
            results = original_schedule(time_window)
        else:
            # Fallback scheduling
            results = run_basic_scheduling(scheduler, time_window)
        
        # Post-process to add buffer and costs
        enhanced_results = []
        
        for schedule in results:
            task = scheduler.tasks[schedule.task_id]
            
            # Add buffer between segments if segmented
            if hasattr(schedule, 'sub_segment_schedule') and schedule.sub_segment_schedule:
                # Apply buffer and cost to sub-segments
                schedule = apply_buffer_and_cost_to_schedule(schedule, task, config)
            
            enhanced_results.append(schedule)
        
        return enhanced_results
    
    # Replace methods
    scheduler.find_available_resources_for_task_with_segmentation = enhanced_find_resources
    scheduler.priority_aware_schedule_with_segmentation = enhanced_scheduling
    
    print(f"✅ Quick segmentation fix applied:")
    print(f"  Buffer: {buffer_ms}ms")
    print(f"  Scheduling cost: {cost_ms}ms per segment")
    
    return config


def find_basic_resources(scheduler, task, current_time):
    """Basic resource finding fallback"""
    assigned_resources = {}
    
    for segment in task.segments:
        res_type = segment.resource_type
        
        # Find available resource of this type
        for resource in scheduler.resources[res_type]:
            queue = scheduler.resource_queues.get(resource.unit_id)
            if queue and queue.available_time <= current_time:
                assigned_resources[res_type] = resource.unit_id
                break
    
    return assigned_resources if len(assigned_resources) == len(task.segments) else None


def run_basic_scheduling(scheduler, time_window: float) -> List[TaskScheduleInfo]:
    """Basic scheduling fallback"""
    results = []
    current_time = 0.0
    
    # Reset resource queues
    for queue in scheduler.resource_queues.values():
        queue.available_time = 0.0
    
    # Reset task states
    for task in scheduler.tasks.values():
        task.last_execution_time = -float('inf')
    
    # Simple scheduling loop
    task_execution_counts = defaultdict(int)
    
    while current_time < time_window:
        scheduled_any = False
        
        # Get tasks sorted by priority
        ready_tasks = []
        for task in scheduler.tasks.values():
            if task.last_execution_time + task.min_interval_ms <= current_time:
                ready_tasks.append(task)
        
        ready_tasks.sort(key=lambda t: t.priority.value)
        
        for task in ready_tasks:
            # Find resources
            assigned_resources = find_basic_resources(scheduler, task, current_time)
            
            if assigned_resources:
                # Calculate duration
                total_duration = 0.0
                for segment in task.segments:
                    if segment.resource_type in assigned_resources:
                        resource_id = assigned_resources[segment.resource_type]
                        resource = next(r for r in scheduler.resources[segment.resource_type] 
                                      if r.unit_id == resource_id)
                        duration = segment.get_duration(resource.bandwidth)
                        total_duration = max(total_duration, segment.start_time + duration)
                
                # Create schedule
                schedule = TaskScheduleInfo(
                    task_id=task.task_id,
                    start_time=current_time,
                    end_time=current_time + total_duration,
                    assigned_resources=assigned_resources,
                    actual_latency=total_duration,
                    runtime_type=task.runtime_type,
                    used_cuts={},
                    segmentation_overhead=0.0,
                    sub_segment_schedule=[]
                )
                
                # Update resource availability
                for segment in task.segments:
                    if segment.resource_type in assigned_resources:
                        resource_id = assigned_resources[segment.resource_type]
                        resource = next(r for r in scheduler.resources[segment.resource_type] 
                                      if r.unit_id == resource_id)
                        duration = segment.get_duration(resource.bandwidth)
                        end_time = current_time + segment.start_time + duration
                        scheduler.resource_queues[resource_id].available_time = end_time
                
                # Record schedule
                results.append(schedule)
                task.last_execution_time = current_time
                task_execution_counts[task.task_id] += 1
                scheduled_any = True
                break
        
        # Advance time
        if scheduled_any:
            current_time += 0.1  # Small advance
        else:
            current_time += 1.0  # Larger advance
    
    return results


def apply_buffer_and_cost_to_schedule(schedule, task, config):
    """Apply buffer and scheduling cost to a schedule"""
    
    if not schedule.sub_segment_schedule:
        return schedule
    
    # Get buffer for this task
    buffer = config.get_buffer_for_task(task)
    cost_per_segment = config.scheduling_cost_ms
    
    # Apply buffer between segments
    enhanced_sub_schedule = []
    prev_end = None
    
    for i, (sub_id, start, end) in enumerate(schedule.sub_segment_schedule):
        # Add scheduling cost to duration
        duration = end - start
        enhanced_duration = duration + cost_per_segment
        
        # Apply buffer from previous segment
        if prev_end is not None:
            enhanced_start = max(start, prev_end + buffer)
        else:
            enhanced_start = start
        
        enhanced_end = enhanced_start + enhanced_duration
        enhanced_sub_schedule.append((sub_id, enhanced_start, enhanced_end))
        prev_end = enhanced_end
    
    # Update schedule
    schedule.sub_segment_schedule = enhanced_sub_schedule
    if enhanced_sub_schedule:
        schedule.end_time = enhanced_sub_schedule[-1][2]  # Last segment end time
        schedule.actual_latency = schedule.end_time - schedule.start_time
    
    # Add total scheduling overhead
    num_segments = len(enhanced_sub_schedule)
    schedule.segmentation_overhead += config.get_cost_for_task(task, num_segments)
    
    return schedule


def test_quick_fix():
    """Test the quick fix"""
    
    print("=== Testing Quick Segmentation Fix ===")
    print(f"Python version: {__import__('sys').version}")
    
    try:
        from task import NNTask
        from scheduler import MultiResourceScheduler
        from enums import SegmentationStrategy
        
        # Create scheduler
        scheduler = MultiResourceScheduler(enable_segmentation=True)
        
        # Apply quick fix
        config = apply_quick_segmentation_fix(scheduler, buffer_ms=0.1, cost_ms=0.12)
        
        # Add resources
        scheduler.add_npu("NPU_0", bandwidth=8.0)
        scheduler.add_npu("NPU_1", bandwidth=4.0)
        scheduler.add_dsp("DSP_0", bandwidth=4.0)
        
        print(f"\nAdded {len(scheduler.resources[ResourceType.NPU])} NPUs and {len(scheduler.resources[ResourceType.DSP])} DSPs")
        
        # Create test tasks
        task1 = NNTask("T1", "HighPriorityTask", 
                      priority=TaskPriority.HIGH,
                      runtime_type=RuntimeType.ACPU_RUNTIME,
                      segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION)
        task1.set_npu_only({2.0: 30, 4.0: 20, 8.0: 12}, "task1_seg")
        
        if hasattr(task1, 'add_cut_points_to_segment'):
            task1.add_cut_points_to_segment("task1_seg", [("cut1", 0.5, 0.15)])
        
        task1.set_performance_requirements(fps=20, latency=50)
        scheduler.add_task(task1)
        
        task2 = NNTask("T2", "NormalPriorityTask", 
                      priority=TaskPriority.NORMAL,
                      runtime_type=RuntimeType.ACPU_RUNTIME,
                      segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
        task2.set_npu_only({2.0: 25, 4.0: 15, 8.0: 10}, "task2_seg")
        task2.set_performance_requirements(fps=15, latency=70)
        scheduler.add_task(task2)
        
        print(f"Created {len(scheduler.tasks)} test tasks")
        
        # Run scheduling
        print("\nRunning enhanced scheduling...")
        results = scheduler.priority_aware_schedule_with_segmentation(time_window=100.0)
        
        print(f"Scheduled {len(results)} events")
        
        # Analyze results
        print("\nResults analysis:")
        for i, schedule in enumerate(results[:3]):
            task = scheduler.tasks[schedule.task_id]
            buffer = config.get_buffer_for_task(task)
            
            print(f"  Event {i+1}: {task.name}")
            print(f"    Time: {schedule.start_time:.2f} - {schedule.end_time:.2f}ms")
            print(f"    Buffer for this priority: {buffer:.2f}ms")
            
            if hasattr(schedule, 'sub_segment_schedule') and schedule.sub_segment_schedule:
                print(f"    Sub-segments: {len(schedule.sub_segment_schedule)}")
                for j, (sub_id, start, end) in enumerate(schedule.sub_segment_schedule):
                    print(f"      {j+1}: {start:.2f} - {end:.2f}ms")
            
            if hasattr(schedule, 'segmentation_overhead'):
                print(f"    Total overhead: {schedule.segmentation_overhead:.2f}ms")
        
        # Validate with basic check
        print("\nValidation:")
        conflicts = check_basic_conflicts(results, scheduler)
        
        if not conflicts:
            print("  ✅ No obvious resource conflicts detected")
        else:
            print(f"  ⚠️ Found {len(conflicts)} potential conflicts")
            for conflict in conflicts[:3]:
                print(f"    - {conflict}")
        
        print("\n✅ Quick fix test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_basic_conflicts(results, scheduler):
    """Basic conflict detection"""
    conflicts = []
    
    # Group by resource
    resource_usage = defaultdict(list)
    
    for schedule in results:
        for res_type, res_id in schedule.assigned_resources.items():
            resource_usage[res_id].append((schedule.start_time, schedule.end_time, schedule.task_id))
    
    # Check for overlaps
    for res_id, usages in resource_usage.items():
        usages.sort()  # Sort by start time
        
        for i in range(len(usages) - 1):
            curr_start, curr_end, curr_task = usages[i]
            next_start, next_end, next_task = usages[i + 1]
            
            if curr_end > next_start + 0.001:  # Small tolerance
                overlap = curr_end - next_start
                conflicts.append(f"{res_id}: {curr_task} overlaps {next_task} by {overlap:.3f}ms")
    
    return conflicts


if __name__ == "__main__":
    success = test_quick_fix()
    
    if success:
        print("\n🎉 Quick segmentation fix is ready for Python 3.12!")
        print("\nUsage:")
        print("  from quick_fix_segmentation import apply_quick_segmentation_fix")
        print("  apply_quick_segmentation_fix(scheduler, buffer_ms=0.1, cost_ms=0.12)")
    else:
        print("\n❌ Please check the error messages above")
