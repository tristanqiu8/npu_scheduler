#!/usr/bin/env python3
"""
Simplified segmentation fix for the scheduler
This version focuses on making segmentation work with minimal changes
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from enums import ResourceType, TaskPriority, RuntimeType
from models import TaskScheduleInfo, ResourceBinding, SubSegment


def patch_scheduler_segmentation_v2(scheduler):
    """
    Apply a simplified patch to fix segmentation scheduling
    """
    
    # Save original methods
    original_find_available = scheduler.find_available_resources_for_task_with_segmentation
    
    def find_available_resources_fixed(task, current_time):
        """Fixed version that properly handles segmentation"""
        
        # Make segmentation decision
        if scheduler.enable_segmentation:
            segmentation_decision = scheduler.make_segmentation_decision(task, current_time)
            task.apply_segmentation_decision(segmentation_decision)
        
        # For segmented tasks, we need special handling
        if task.is_segmented and scheduler.enable_segmentation:
            return handle_segmented_task_scheduling(scheduler, task, current_time)
        else:
            # Use original logic for non-segmented tasks
            return original_find_available(task, current_time)
    
    # Replace the method
    scheduler.find_available_resources_for_task_with_segmentation = find_available_resources_fixed
    
    print("✅ Segmentation patch V2 applied")


def handle_segmented_task_scheduling(scheduler, task, current_time):
    """
    Handle scheduling of segmented tasks
    """
    
    # Get sub-segments
    sub_segments = task.get_sub_segments_for_scheduling()
    if not sub_segments:
        return None
    
    # Group by resource type
    segments_by_type = defaultdict(list)
    for seg in sub_segments:
        segments_by_type[seg.resource_type].append(seg)
    
    # Try to find resources
    if task.runtime_type == RuntimeType.DSP_RUNTIME and len(segments_by_type) > 1:
        # Need bound resources
        return find_bound_resources_for_segmented_task(scheduler, task, segments_by_type, current_time)
    else:
        # Can use pipelined resources
        return find_pipelined_resources_for_segmented_task(scheduler, task, segments_by_type, current_time)


def find_bound_resources_for_segmented_task(scheduler, task, segments_by_type, current_time):
    """
    Find resources that can be bound together for segmented DSP_Runtime tasks
    """
    
    # Find a valid combination of resources
    best_combo = None
    min_end_time = float('inf')
    
    # Try all combinations (simplified - just try first available of each type)
    combo = {}
    
    for res_type, segments in segments_by_type.items():
        found = False
        for resource in scheduler.resources[res_type]:
            queue = scheduler.resource_queues[resource.unit_id]
            
            # Check if available
            if (queue.available_time <= current_time and 
                not queue.is_bound_to_other_task(task.task_id, current_time) and
                not queue.has_higher_priority_tasks(task.priority, current_time, task.task_id)):
                
                combo[res_type] = resource.unit_id
                found = True
                break
        
        if not found:
            return None  # Can't find resources
    
    # Calculate binding duration
    max_end_time = current_time
    for res_type, segments in segments_by_type.items():
        resource_id = combo[res_type]
        resource = next(r for r in scheduler.resources[res_type] if r.unit_id == resource_id)
        
        for seg in segments:
            duration = seg.get_duration(resource.bandwidth)
            seg_end = current_time + seg.start_time + duration
            max_end_time = max(max_end_time, seg_end)
    
    # Bind resources
    for resource_id in combo.values():
        scheduler.resource_queues[resource_id].bind_resource(task.task_id, max_end_time)
    
    # Record binding
    binding = ResourceBinding(
        task_id=task.task_id,
        bound_resources=set(combo.values()),
        binding_start=current_time,
        binding_end=max_end_time
    )
    scheduler.active_bindings.append(binding)
    
    return combo


def find_pipelined_resources_for_segmented_task(scheduler, task, segments_by_type, current_time):
    """
    Find resources for pipelined execution of segmented tasks
    """
    
    assigned_resources = {}
    
    for res_type, segments in segments_by_type.items():
        # Find best resource for this type
        best_resource = None
        best_score = float('inf')
        
        for resource in scheduler.resources[res_type]:
            queue = scheduler.resource_queues[resource.unit_id]
            
            # Calculate score (prefer available resources with higher bandwidth)
            if queue.available_time <= current_time:
                if not queue.has_higher_priority_tasks(task.priority, current_time, task.task_id):
                    score = queue.available_time - resource.bandwidth * 0.1  # Favor high bandwidth
                    if score < best_score:
                        best_score = score
                        best_resource = resource
        
        if best_resource:
            assigned_resources[res_type] = best_resource.unit_id
        else:
            return None  # Can't find suitable resource
    
    return assigned_resources


def create_sub_segment_schedule(scheduler, task, assigned_resources, current_time):
    """
    Create the sub-segment execution schedule
    """
    
    sub_segment_schedule = []
    
    # Get sub-segments
    sub_segments = task.get_sub_segments_for_scheduling()
    
    for sub_seg in sub_segments:
        if sub_seg.resource_type in assigned_resources:
            resource_id = assigned_resources[sub_seg.resource_type]
            resource = next(r for r in scheduler.resources[sub_seg.resource_type] 
                          if r.unit_id == resource_id)
            
            duration = sub_seg.get_duration(resource.bandwidth)
            start_time = current_time + sub_seg.start_time
            end_time = start_time + duration
            
            sub_segment_schedule.append((sub_seg.sub_id, start_time, end_time))
            
            # Update resource availability for pipelined tasks
            if task.runtime_type == RuntimeType.ACPU_RUNTIME:
                scheduler.resource_queues[resource_id].available_time = end_time
    
    return sub_segment_schedule


# Also need to patch the main scheduling loop
def patch_scheduling_loop(scheduler):
    """
    Patch the main scheduling loop to handle segmented tasks properly
    """
    
    original_schedule = scheduler.priority_aware_schedule_with_segmentation
    
    def enhanced_schedule_with_segmentation(time_window: float = 1000.0):
        """Enhanced scheduling that properly creates sub-segment schedules"""
        
        # First, call the original scheduling
        results = original_schedule(time_window)
        
        # Post-process to add sub-segment schedules
        enhanced_results = []
        
        for schedule in results:
            task = scheduler.tasks[schedule.task_id]
            
            # If task is segmented but schedule doesn't have sub-segments, create them
            if task.is_segmented and not schedule.sub_segment_schedule:
                sub_segment_schedule = create_sub_segment_schedule(
                    scheduler, task, schedule.assigned_resources, schedule.start_time
                )
                
                # Update schedule info
                schedule.sub_segment_schedule = sub_segment_schedule
                
                # Update end time based on sub-segments
                if sub_segment_schedule:
                    schedule.end_time = max(end for _, _, end in sub_segment_schedule)
                    schedule.actual_latency = schedule.end_time - schedule.start_time
            
            enhanced_results.append(schedule)
        
        return enhanced_results
    
    scheduler.priority_aware_schedule_with_segmentation = enhanced_schedule_with_segmentation


def apply_complete_segmentation_fix(scheduler):
    """
    Apply the complete segmentation fix
    """
    patch_scheduler_segmentation_v2(scheduler)
    patch_scheduling_loop(scheduler)
    print("✅ Complete segmentation fix applied")


# Test function
def test_segmentation_fix_v2():
    """
    Test the simplified segmentation fix
    """
    from main import create_sample_tasks_with_segmentation, setup_scheduler_with_segmentation
    from schedule_validator import validate_schedule
    
    print("=== Testing Segmentation Fix V2 ===\n")
    
    # Create scheduler
    scheduler = setup_scheduler_with_segmentation(enable_segmentation=True)
    
    # Apply fix
    apply_complete_segmentation_fix(scheduler)
    
    # Add simple test tasks
    print("Creating test tasks...")
    
    # Task 1: Simple segmented task
    task1 = NNTask("T1", "TestSegmented", 
                   priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION)
    task1.set_npu_only({2.0: 30, 4.0: 20, 8.0: 12}, "test_seg")
    task1.add_cut_points_to_segment("test_seg", [
        ("cut1", 0.5, 0.15)
    ])
    task1.set_performance_requirements(fps=20, latency=50)
    scheduler.add_task(task1)
    
    # Task 2: Non-segmented task
    task2 = NNTask("T2", "TestNormal", 
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task2.set_npu_only({2.0: 25, 4.0: 15, 8.0: 10}, "normal_seg")
    task2.set_performance_requirements(fps=15, latency=70)
    scheduler.add_task(task2)
    
    print(f"Added {len(scheduler.tasks)} tasks")
    
    # Run scheduling
    print("\nRunning scheduling...")
    results = scheduler.priority_aware_schedule_with_segmentation(time_window=200.0)
    
    print(f"Scheduled {len(results)} events")
    
    # Analyze results
    segmented_count = 0
    for schedule in results:
        if schedule.sub_segment_schedule and len(schedule.sub_segment_schedule) > 1:
            segmented_count += 1
            print(f"  {schedule.task_id}: {len(schedule.sub_segment_schedule)} segments")
    
    print(f"Segmented executions: {segmented_count}")
    
    # Validate
    is_valid, errors = validate_schedule(scheduler)
    
    if is_valid:
        print("\n✅ Segmentation fix V2 works correctly!")
    else:
        print(f"\n❌ Found {len(errors)} errors")
        for error in errors[:3]:
            print(f"  - {error}")
    
    return is_valid


if __name__ == "__main__":
    # Need to import NNTask for testing
    from task import NNTask
    from enums import SegmentationStrategy
    
    test_segmentation_fix_v2()
