#!/usr/bin/env python3
"""
Enhanced segmentation fix that adds timing buffers and scheduling costs
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from enums import ResourceType, TaskPriority, RuntimeType
from models import TaskScheduleInfo, ResourceBinding, SubSegment

# Configuration constants
SEGMENT_BUFFER_MS = 0.1      # Buffer between segments to avoid conflicts
SCHEDULING_COST_MS = 0.12    # Fixed cost for each segment scheduling
MIN_SEGMENT_DURATION = 0.5   # Minimum segment duration to be meaningful


class SchedulingConfig:
    """Configuration for enhanced scheduling"""
    def __init__(self):
        self.segment_buffer_ms = SEGMENT_BUFFER_MS
        self.scheduling_cost_ms = SCHEDULING_COST_MS
        self.min_segment_duration = MIN_SEGMENT_DURATION
        self.enable_dynamic_buffer = True  # Adjust buffer based on task priority
    
    def get_buffer_for_task(self, task):
        """Get appropriate buffer based on task priority"""
        if not self.enable_dynamic_buffer:
            return self.segment_buffer_ms
        
        # Critical tasks get smaller buffer, low priority gets larger
        priority_buffers = {
            TaskPriority.CRITICAL: self.segment_buffer_ms * 0.5,
            TaskPriority.HIGH: self.segment_buffer_ms,
            TaskPriority.NORMAL: self.segment_buffer_ms * 1.5,
            TaskPriority.LOW: self.segment_buffer_ms * 2.0
        }
        return priority_buffers.get(task.priority, self.segment_buffer_ms)


def apply_enhanced_segmentation_fix(scheduler, config: SchedulingConfig = None):
    """
    Apply enhanced segmentation fix with timing buffers and scheduling costs
    """
    if config is None:
        config = SchedulingConfig()
    
    # Store config in scheduler
    scheduler._segmentation_config = config
    
    # Apply the basic segmentation fix first
    from scheduler_segmentation_fix_v2 import patch_scheduler_segmentation_v2
    patch_scheduler_segmentation_v2(scheduler)
    
    # Then apply enhanced fixes
    patch_sub_segment_creation_with_costs(scheduler, config)
    patch_resource_availability_with_buffer(scheduler, config)
    patch_scheduling_loop_enhanced(scheduler, config)
    
    print(f"✅ Enhanced segmentation fix applied:")
    print(f"  - Segment buffer: {config.segment_buffer_ms}ms")
    print(f"  - Scheduling cost: {config.scheduling_cost_ms}ms per segment")
    print(f"  - Dynamic buffer: {config.enable_dynamic_buffer}")


def patch_sub_segment_creation_with_costs(scheduler, config):
    """
    Patch sub-segment creation to include scheduling costs and buffers
    """
    
    def create_enhanced_sub_segment_schedule(task, assigned_resources, current_time):
        """Create sub-segment schedule with costs and buffers"""
        
        sub_segment_schedule = []
        sub_segments = task.get_sub_segments_for_scheduling()
        
        if not sub_segments:
            return sub_segment_schedule
        
        # Get task-specific buffer
        buffer = config.get_buffer_for_task(task)
        
        # Track actual timeline
        timeline_end = current_time
        
        for i, sub_seg in enumerate(sub_segments):
            if sub_seg.resource_type in assigned_resources:
                resource_id = assigned_resources[sub_seg.resource_type]
                resource = next(r for r in scheduler.resources[sub_seg.resource_type] 
                              if r.unit_id == resource_id)
                
                # Calculate base duration
                base_duration = sub_seg.get_duration(resource.bandwidth)
                
                # Add scheduling cost (overhead for each segment)
                total_duration = base_duration + config.scheduling_cost_ms
                
                # Calculate start time
                ideal_start = current_time + sub_seg.start_time
                
                # Ensure we don't start before previous segment ends + buffer
                actual_start = max(ideal_start, timeline_end + buffer)
                
                # Check resource availability
                queue = scheduler.resource_queues[resource_id]
                if queue.available_time > actual_start:
                    actual_start = queue.available_time + buffer
                
                actual_end = actual_start + total_duration
                
                # Validate minimum duration
                if total_duration < config.min_segment_duration:
                    print(f"Warning: Segment {sub_seg.sub_id} duration {total_duration:.2f}ms below minimum")
                
                sub_segment_schedule.append((sub_seg.sub_id, actual_start, actual_end))
                
                # Update resource availability with buffer
                if task.runtime_type == RuntimeType.ACPU_RUNTIME:
                    queue.available_time = actual_end + buffer
                
                # Update timeline
                timeline_end = actual_end
        
        return sub_segment_schedule
    
    # Store the enhanced function
    scheduler._create_enhanced_sub_segment_schedule = create_enhanced_sub_segment_schedule


def patch_resource_availability_with_buffer(scheduler, config):
    """
    Patch resource availability checking to include buffers
    """
    
    # Enhance the resource queue class
    for queue in scheduler.resource_queues.values():
        original_is_available = getattr(queue, 'is_available_for_sub_segment', None)
        
        def is_available_with_buffer(start_time, duration, task_priority=TaskPriority.NORMAL):
            """Check availability with buffer consideration"""
            
            # Get priority-specific buffer
            priority_buffers = {
                TaskPriority.CRITICAL: config.segment_buffer_ms * 0.5,
                TaskPriority.HIGH: config.segment_buffer_ms,
                TaskPriority.NORMAL: config.segment_buffer_ms * 1.5,
                TaskPriority.LOW: config.segment_buffer_ms * 2.0
            }
            buffer = priority_buffers.get(task_priority, config.segment_buffer_ms)
            
            # Add buffer to duration
            buffered_duration = duration + buffer * 2  # Buffer before and after
            buffered_start = start_time - buffer
            
            # Check against existing reservations
            for _, res_start, res_end in queue.sub_segment_reservations:
                if not (buffered_start + buffered_duration <= res_start or 
                       buffered_start >= res_end):
                    return False
            
            # Check basic availability
            return queue.available_time <= buffered_start
        
        queue.is_available_with_buffer = is_available_with_buffer


def patch_scheduling_loop_enhanced(scheduler, config):
    """
    Patch the main scheduling loop with enhanced timing
    """
    
    original_schedule = scheduler.priority_aware_schedule_with_segmentation
    
    def enhanced_scheduling_loop(time_window: float = 1000.0):
        """Enhanced scheduling loop with proper timing management"""
        
        # Reset state
        reset_scheduler_state(scheduler)
        
        # Track execution counts and statistics
        task_execution_count = defaultdict(int)
        total_scheduling_overhead = 0.0
        current_time = 0.0
        
        while current_time < time_window:
            # Clean up expired bindings
            scheduler.cleanup_expired_bindings(current_time)
            
            # Find ready tasks
            ready_tasks = find_ready_tasks_enhanced(scheduler, current_time, task_execution_count)
            
            if not ready_tasks:
                # Advance to next event
                current_time = find_next_event_time(scheduler, current_time, time_window)
                continue
            
            # Schedule tasks in priority order
            scheduled_any = False
            
            for task in ready_tasks:
                # Find available resources
                assigned_resources = scheduler.find_available_resources_for_task_with_segmentation(
                    task, current_time
                )
                
                if assigned_resources:
                    # Create enhanced schedule
                    schedule_info = create_enhanced_task_schedule(
                        scheduler, task, assigned_resources, current_time, config
                    )
                    
                    if schedule_info:
                        # Record schedule
                        task.schedule_info = schedule_info
                        task.last_execution_time = schedule_info.start_time
                        scheduler.schedule_history.append(schedule_info)
                        task_execution_count[task.task_id] += 1
                        scheduled_any = True
                        
                        # Track overhead
                        if schedule_info.sub_segment_schedule:
                            segment_count = len(schedule_info.sub_segment_schedule)
                            overhead = segment_count * config.scheduling_cost_ms
                            total_scheduling_overhead += overhead
                        
                        break  # Schedule one task at a time for better control
            
            # Small time advance
            if scheduled_any:
                current_time += config.segment_buffer_ms
            else:
                current_time += 1.0
        
        # Update statistics
        update_enhanced_statistics(scheduler, total_scheduling_overhead)
        
        return scheduler.schedule_history
    
    scheduler.priority_aware_schedule_with_segmentation = enhanced_scheduling_loop


def create_enhanced_task_schedule(scheduler, task, assigned_resources, current_time, config):
    """
    Create enhanced task schedule with proper timing
    """
    
    actual_start = current_time
    
    # Create sub-segment schedule
    if hasattr(scheduler, '_create_enhanced_sub_segment_schedule'):
        sub_segment_schedule = scheduler._create_enhanced_sub_segment_schedule(
            task, assigned_resources, actual_start
        )
    else:
        sub_segment_schedule = []
    
    # Calculate actual end time
    if sub_segment_schedule:
        actual_end = max(end for _, _, end in sub_segment_schedule)
        
        # Add final buffer
        actual_end += config.get_buffer_for_task(task)
    else:
        # Non-segmented task
        duration = task.get_total_duration({
            res_type: next(r.bandwidth for r in scheduler.resources[res_type] 
                         if r.unit_id == res_id)
            for res_type, res_id in assigned_resources.items()
        })
        actual_end = actual_start + duration + config.scheduling_cost_ms
    
    # Calculate total overhead (including scheduling costs)
    segmentation_overhead = task.total_segmentation_overhead
    if sub_segment_schedule:
        segmentation_overhead += len(sub_segment_schedule) * config.scheduling_cost_ms
    
    # Create schedule info
    schedule_info = TaskScheduleInfo(
        task_id=task.task_id,
        start_time=actual_start,
        end_time=actual_end,
        assigned_resources=assigned_resources,
        actual_latency=actual_end - current_time,
        runtime_type=task.runtime_type,
        used_cuts=task.current_segmentation.copy(),
        segmentation_overhead=segmentation_overhead,
        sub_segment_schedule=sub_segment_schedule
    )
    
    return schedule_info


def find_ready_tasks_enhanced(scheduler, current_time, task_execution_count):
    """
    Find ready tasks with enhanced dependency checking
    """
    ready_tasks = []
    config = scheduler._segmentation_config
    
    for task in scheduler.tasks.values():
        # Check minimum interval with buffer
        min_interval = task.min_interval_ms + config.get_buffer_for_task(task)
        if task.last_execution_time + min_interval > current_time:
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
                    latest_end = max(s.end_time for s in dep_schedules)
                    max_dep_end = max(max_dep_end, latest_end + config.segment_buffer_ms)
        
        if deps_satisfied:
            task.ready_time = max(current_time, max_dep_end)
            if task.ready_time <= current_time + config.segment_buffer_ms:
                ready_tasks.append(task)
    
    # Sort by priority and ready time
    ready_tasks.sort(key=lambda t: (t.priority.value, t.ready_time, t.task_id))
    
    return ready_tasks


def find_next_event_time(scheduler, current_time, time_window):
    """
    Find the next event time in the schedule
    """
    next_time = current_time + 1.0
    
    # Check resource availability
    for queue in scheduler.resource_queues.values():
        if queue.available_time > current_time:
            next_time = min(next_time, queue.available_time)
        if queue.bound_until > current_time:
            next_time = min(next_time, queue.bound_until)
    
    # Check task ready times
    for task in scheduler.tasks.values():
        if task.last_execution_time > -float('inf'):
            next_ready = task.last_execution_time + task.min_interval_ms
            if next_ready > current_time:
                next_time = min(next_time, next_ready)
    
    return min(next_time, time_window)


def reset_scheduler_state(scheduler):
    """
    Reset scheduler state for new scheduling run
    """
    # Reset resource queues
    for queue in scheduler.resource_queues.values():
        queue.available_time = 0.0
        queue.release_binding()
        if hasattr(queue, 'sub_segment_reservations'):
            queue.sub_segment_reservations.clear()
        # Clear priority queues
        for p in TaskPriority:
            queue.queues[p].clear()
    
    # Reset task states
    for task in scheduler.tasks.values():
        task.schedule_info = None
        task.last_execution_time = -float('inf')
        task.ready_time = 0
        task.current_segmentation = {}
        task.total_segmentation_overhead = 0.0
    
    # Clear history
    scheduler.schedule_history.clear()
    scheduler.active_bindings.clear()
    if hasattr(scheduler, 'segmentation_decisions_history'):
        scheduler.segmentation_decisions_history.clear()


def update_enhanced_statistics(scheduler, total_scheduling_overhead):
    """
    Update scheduler statistics with enhanced metrics
    """
    segmented_tasks = sum(1 for t in scheduler.tasks.values() if t.is_segmented)
    total_overhead = sum(t.total_segmentation_overhead for t in scheduler.tasks.values())
    
    scheduler.segmentation_stats = {
        'total_decisions': len(getattr(scheduler, 'segmentation_decisions_history', [])),
        'segmented_tasks': segmented_tasks,
        'total_overhead': total_overhead,
        'scheduling_overhead': total_scheduling_overhead,
        'average_benefit': 0.0
    }


def test_enhanced_segmentation():
    """
    Test the enhanced segmentation fix
    """
    from enums import SegmentationStrategy
    from task import NNTask
    from scheduler import MultiResourceScheduler
    from schedule_validator import validate_schedule
    
    print("=== Testing Enhanced Segmentation Fix ===\n")
    
    # Create custom configuration
    config = SchedulingConfig()
    config.segment_buffer_ms = 0.15  # Slightly larger buffer
    config.scheduling_cost_ms = 0.12  # Your requested scheduling cost
    config.enable_dynamic_buffer = True
    
    # Create scheduler
    scheduler = MultiResourceScheduler(enable_segmentation=True)
    
    # Apply enhanced fix
    apply_enhanced_segmentation_fix(scheduler, config)
    
    # Add resources
    scheduler.add_npu("NPU_0", bandwidth=8.0)
    scheduler.add_npu("NPU_1", bandwidth=4.0)
    scheduler.add_dsp("DSP_0", bandwidth=4.0)
    
    # Create test tasks
    task1 = NNTask("T1", "HighPrioritySegmented", 
                   priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION)
    task1.set_npu_only({2.0: 30, 4.0: 20, 8.0: 12}, "seg1")
    task1.add_cut_points_to_segment("seg1", [
        ("cut1", 0.33, 0.12),
        ("cut2", 0.67, 0.13)
    ])
    task1.set_performance_requirements(fps=20, latency=60)
    scheduler.add_task(task1)
    
    task2 = NNTask("T2", "NormalPriority", 
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION)
    task2.set_npu_only({2.0: 25, 4.0: 15, 8.0: 10}, "seg2")
    task2.add_cut_points_to_segment("seg2", [("mid", 0.5, 0.14)])
    task2.set_performance_requirements(fps=15, latency=80)
    scheduler.add_task(task2)
    
    # Run scheduling
    print("Running enhanced scheduling...")
    results = scheduler.priority_aware_schedule_with_segmentation(time_window=200.0)
    
    print(f"Scheduled {len(results)} events")
    
    # Analyze timing
    print("\nTiming Analysis:")
    for i, schedule in enumerate(results[:5]):
        task = scheduler.tasks[schedule.task_id]
        buffer = config.get_buffer_for_task(task)
        
        print(f"\nEvent {i+1} - {schedule.task_id} ({task.priority.name}):")
        print(f"  Overall: {schedule.start_time:.2f} - {schedule.end_time:.2f}ms")
        print(f"  Priority buffer: {buffer:.2f}ms")
        
        if schedule.sub_segment_schedule:
            print(f"  Sub-segments ({len(schedule.sub_segment_schedule)}):")
            prev_end = None
            for j, (sub_id, start, end) in enumerate(schedule.sub_segment_schedule):
                gap = start - prev_end if prev_end else 0
                duration = end - start
                scheduling_cost = config.scheduling_cost_ms
                actual_work = duration - scheduling_cost
                
                print(f"    {j+1}: {start:.2f}-{end:.2f}ms (work: {actual_work:.2f}ms + cost: {scheduling_cost:.2f}ms)")
                if prev_end and gap > 0:
                    print(f"        Gap from previous: {gap:.2f}ms")
                prev_end = end
    
    # Validate
    is_valid, errors = validate_schedule(scheduler)
    
    if is_valid:
        print("\n✅ Enhanced segmentation working perfectly!")
        print(f"  No timing conflicts with {config.segment_buffer_ms}ms buffer")
        print(f"  Scheduling cost of {config.scheduling_cost_ms}ms per segment applied")
    else:
        print(f"\n❌ Still found {len(errors)} timing issues")
        for error in errors[:3]:
            print(f"  - {error}")
    
    # Show statistics
    stats = scheduler.segmentation_stats
    print(f"\nStatistics:")
    print(f"  Segmented tasks: {stats['segmented_tasks']}")
    print(f"  Total segmentation overhead: {stats['total_overhead']:.2f}ms")
    print(f"  Total scheduling overhead: {stats['scheduling_overhead']:.2f}ms")
    
    return is_valid, scheduler


if __name__ == "__main__":
    success, scheduler = test_enhanced_segmentation()
    
    if success:
        print("\n🎉 Enhanced segmentation fix is ready!")
        print("\nUsage:")
        print("  config = SchedulingConfig()")
        print("  config.segment_buffer_ms = 0.1  # Adjust as needed")
        print("  config.scheduling_cost_ms = 0.12  # Your scheduling cost")
        print("  apply_enhanced_segmentation_fix(scheduler, config)")
    else:
        print("\n⚠️ Consider increasing buffer or adjusting costs")
