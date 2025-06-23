#!/usr/bin/env python3
"""
Final segmentation fix that addresses timing precision issues
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from enums import ResourceType, TaskPriority, RuntimeType
from models import TaskScheduleInfo, ResourceBinding, SubSegment
from scheduler_segmentation_fix_v2 import create_sub_segment_schedule

# Timing precision buffer to avoid floating-point conflicts
TIMING_BUFFER = 0.01  # 0.01ms buffer between segments


def apply_final_segmentation_fix(scheduler):
    """
    Apply the final comprehensive segmentation fix
    """
    
    # First apply the basic fix
    from scheduler_segmentation_fix_v2 import patch_scheduler_segmentation_v2
    patch_scheduler_segmentation_v2(scheduler)
    
    # Then apply timing precision fixes
    original_create_schedule = create_sub_segment_schedule
    
    def create_sub_segment_schedule_with_buffer(scheduler, task, assigned_resources, current_time):
        """Enhanced version with timing buffers"""
        
        sub_segment_schedule = []
        sub_segments = task.get_sub_segments_for_scheduling()
        
        # Track the actual end time of previous segment
        prev_actual_end = current_time
        
        for i, sub_seg in enumerate(sub_segments):
            if sub_seg.resource_type in assigned_resources:
                resource_id = assigned_resources[sub_seg.resource_type]
                resource = next(r for r in scheduler.resources[sub_seg.resource_type] 
                              if r.unit_id == resource_id)
                
                duration = sub_seg.get_duration(resource.bandwidth)
                
                # Calculate start time with buffer
                ideal_start = current_time + sub_seg.start_time
                actual_start = max(ideal_start, prev_actual_end + TIMING_BUFFER)
                
                # Ensure resource is actually available
                queue = scheduler.resource_queues[resource_id]
                if queue.available_time > actual_start:
                    actual_start = queue.available_time + TIMING_BUFFER
                
                actual_end = actual_start + duration
                
                sub_segment_schedule.append((sub_seg.sub_id, actual_start, actual_end))
                
                # Update resource availability with buffer
                if task.runtime_type == RuntimeType.ACPU_RUNTIME:
                    queue.available_time = actual_end + TIMING_BUFFER
                
                prev_actual_end = actual_end
        
        return sub_segment_schedule
    
    # Replace the function in the module
    import scheduler_segmentation_fix_v2
    scheduler_segmentation_fix_v2.create_sub_segment_schedule = create_sub_segment_schedule_with_buffer
    
    # Also patch the scheduling loop to add buffer
    patch_scheduling_with_precision_fix(scheduler)
    
    print("✅ Final segmentation fix with timing precision applied")


def patch_scheduling_with_precision_fix(scheduler):
    """
    Patch the main scheduling loop to handle timing precision
    """
    
    original_schedule = scheduler.priority_aware_schedule_with_segmentation
    
    def schedule_with_precision_fix(time_window: float = 1000.0):
        """Enhanced scheduling with timing precision fixes"""
        
        # Reset with proper initialization
        for queue in scheduler.resource_queues.values():
            queue.available_time = 0.0
            queue.release_binding()
            if hasattr(queue, 'sub_segment_reservations'):
                queue.sub_segment_reservations.clear()
            # Clear priority queues
            from enums import TaskPriority
            for p in TaskPriority:
                queue.queues[p].clear()
        
        for task in scheduler.tasks.values():
            task.schedule_info = None
            task.last_execution_time = -float('inf')
            task.ready_time = 0
            task.current_segmentation = {}
            task.total_segmentation_overhead = 0.0
        
        scheduler.schedule_history.clear()
        scheduler.active_bindings.clear()
        if hasattr(scheduler, 'segmentation_decisions_history'):
            scheduler.segmentation_decisions_history.clear()
        
        # Track execution counts
        task_execution_count = defaultdict(int)
        current_time = 0.0
        
        while current_time < time_window:
            # Clean expired bindings
            scheduler.cleanup_expired_bindings(current_time)
            
            # Find ready tasks
            ready_tasks = []
            
            for task in scheduler.tasks.values():
                # Check minimum interval with buffer
                if task.last_execution_time + task.min_interval_ms - TIMING_BUFFER > current_time:
                    continue
                
                # Check dependencies
                deps_satisfied = True
                max_dep_end = 0.0
                
                for dep_id in task.dependencies:
                    if dep_id in scheduler.tasks:
                        if task_execution_count[dep_id] <= task_execution_count[task.task_id]:
                            deps_satisfied = False
                            break
                        
                        # Find latest execution of dependency
                        dep_schedules = [s for s in scheduler.schedule_history if s.task_id == dep_id]
                        if dep_schedules:
                            max_dep_end = max(s.end_time for s in dep_schedules) + TIMING_BUFFER
                
                if deps_satisfied:
                    task.ready_time = max(current_time, max_dep_end)
                    if task.ready_time <= current_time + TIMING_BUFFER:
                        ready_tasks.append(task)
            
            # Sort by priority
            ready_tasks.sort(key=lambda t: (t.priority.value, t.task_id))
            
            scheduled_any = False
            
            for task in ready_tasks:
                # Find resources
                assigned_resources = scheduler.find_available_resources_for_task_with_segmentation(
                    task, current_time
                )
                
                if assigned_resources:
                    # Create schedule with proper timing
                    actual_start = current_time
                    actual_end = current_time
                    sub_segment_schedule = []
                    
                    if hasattr(scheduler, '_current_schedule_plan') and scheduler._current_schedule_plan:
                        # Use the plan from segmentation scheduler
                        plan = scheduler._current_schedule_plan
                        for timing in plan.get('segment_timings', []):
                            sub_id = timing['sub_segment'].sub_id
                            start = timing['start_time']
                            end = timing['end_time']
                            sub_segment_schedule.append((sub_id, start, end))
                            actual_end = max(actual_end, end)
                    else:
                        # Create schedule from task info
                        from scheduler_segmentation_fix_v2 import create_sub_segment_schedule
                        sub_segment_schedule = create_sub_segment_schedule(
                            scheduler, task, assigned_resources, actual_start
                        )
                        
                        if sub_segment_schedule:
                            actual_end = max(end for _, _, end in sub_segment_schedule)
                        else:
                            # Non-segmented task
                            duration = task.get_total_duration({
                                res_type: next(r.bandwidth for r in scheduler.resources[res_type] 
                                             if r.unit_id == res_id)
                                for res_type, res_id in assigned_resources.items()
                            })
                            actual_end = actual_start + duration
                    
                    # Create schedule info
                    schedule_info = TaskScheduleInfo(
                        task_id=task.task_id,
                        start_time=actual_start,
                        end_time=actual_end,
                        assigned_resources=assigned_resources,
                        actual_latency=actual_end - current_time,
                        runtime_type=task.runtime_type,
                        used_cuts=task.current_segmentation.copy(),
                        segmentation_overhead=task.total_segmentation_overhead,
                        sub_segment_schedule=sub_segment_schedule
                    )
                    
                    # Update task state
                    task.schedule_info = schedule_info
                    task.last_execution_time = actual_start
                    scheduler.schedule_history.append(schedule_info)
                    task_execution_count[task.task_id] += 1
                    scheduled_any = True
                    
                    # Clear the plan
                    if hasattr(scheduler, '_current_schedule_plan'):
                        scheduler._current_schedule_plan = None
            
            # Advance time
            if not scheduled_any:
                # Find next event
                next_time = current_time + 1.0
                
                # Check resource availability
                for queue in scheduler.resource_queues.values():
                    if queue.available_time > current_time:
                        next_time = min(next_time, queue.available_time)
                    if queue.bound_until > current_time:
                        next_time = min(next_time, queue.bound_until)
                
                # Check task intervals
                for task in scheduler.tasks.values():
                    if task.last_execution_time > -float('inf'):
                        next_ready = task.last_execution_time + task.min_interval_ms
                        if next_ready > current_time:
                            next_time = min(next_time, next_ready)
                
                current_time = min(next_time, time_window)
            else:
                # Small advance to check for new opportunities
                current_time += TIMING_BUFFER
        
        # Update statistics
        scheduler.segmentation_stats = {
            'total_decisions': len(getattr(scheduler, 'segmentation_decisions_history', [])),
            'segmented_tasks': sum(1 for t in scheduler.tasks.values() if t.is_segmented),
            'total_overhead': sum(t.total_segmentation_overhead for t in scheduler.tasks.values()),
            'average_benefit': 0.0
        }
        
        return scheduler.schedule_history
    
    scheduler.priority_aware_schedule_with_segmentation = schedule_with_precision_fix


def test_final_fix():
    """Test the final segmentation fix"""
    
    from enums import SegmentationStrategy
    from task import NNTask
    from scheduler import MultiResourceScheduler
    from schedule_validator import validate_schedule
    
    print("=== Testing Final Segmentation Fix ===\n")
    
    # Create scheduler
    scheduler = MultiResourceScheduler(enable_segmentation=True)
    
    # Apply final fix
    apply_final_segmentation_fix(scheduler)
    
    # Add resources
    scheduler.add_npu("NPU_0", bandwidth=8.0)
    scheduler.add_npu("NPU_1", bandwidth=4.0)
    scheduler.add_dsp("DSP_0", bandwidth=4.0)
    
    # Add test tasks
    task1 = NNTask("T1", "HighPriority", 
                   priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION)
    task1.set_npu_only({2.0: 30, 4.0: 20, 8.0: 12}, "seg1")
    task1.add_cut_points_to_segment("seg1", [
        ("cut1", 0.33, 0.12),
        ("cut2", 0.67, 0.13)
    ])
    task1.set_performance_requirements(fps=20, latency=50)
    scheduler.add_task(task1)
    
    task2 = NNTask("T2", "MixedTask", 
                   priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION)
    task2.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 10}, 0, "dsp_seg"),
        (ResourceType.NPU, {2.0: 25, 4.0: 15, 8.0: 10}, 10, "npu_seg"),
    ])
    task2.add_cut_points_to_segment("npu_seg", [("mid", 0.5, 0.14)])
    task2.set_performance_requirements(fps=25, latency=45)
    scheduler.add_task(task2)
    
    # Run scheduling
    print("Running scheduling with final fix...")
    results = scheduler.priority_aware_schedule_with_segmentation(time_window=200.0)
    
    print(f"Scheduled {len(results)} events")
    
    # Validate
    is_valid, errors = validate_schedule(scheduler)
    
    if is_valid:
        print("\n✅ Final fix successful - no conflicts!")
    else:
        print(f"\n⚠️ Still have {len(errors)} timing issues")
        for i, error in enumerate(errors[:3]):
            print(f"  {i+1}. {error}")
    
    # Show improvements
    print("\nTiming precision improvements:")
    print(f"  - Added {TIMING_BUFFER}ms buffer between segments")
    print(f"  - Enhanced resource availability tracking")
    print(f"  - Improved floating-point precision handling")
    
    return is_valid


if __name__ == "__main__":
    success = test_final_fix()
    
    if success:
        print("\n🎉 Segmentation feature is now fully functional!")
        print("Use: apply_final_segmentation_fix(scheduler)")
    else:
        print("\n⚠️ Some timing precision issues remain.")
        print("Consider increasing TIMING_BUFFER or adjusting segment durations.")
