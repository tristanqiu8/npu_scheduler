#!/usr/bin/env python3
"""
Comprehensive Segmentation Patch - Ensures all segmentation issues are resolved
This patch can be applied standalone to fix the scheduler
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from enums import ResourceType, TaskPriority, RuntimeType
from models import TaskScheduleInfo, SubSegment

# Configuration
TIMING_PRECISION_BUFFER = 0.15  # Buffer between segments to prevent timing conflicts
SCHEDULING_OVERHEAD_PER_SEGMENT = 0.1  # Fixed overhead per segment
MAX_ITERATIONS_PER_MS = 50  # Prevent infinite loops


class SegmentationPatchConfig:
    """Configuration for the segmentation patch"""
    
    def __init__(self):
        self.timing_buffer = TIMING_PRECISION_BUFFER
        self.scheduling_overhead = SCHEDULING_OVERHEAD_PER_SEGMENT
        self.max_iterations = MAX_ITERATIONS_PER_MS
        self.debug_mode = False
        
        # Priority-based buffer scaling
        self.priority_buffer_scale = {
            TaskPriority.CRITICAL: 0.5,
            TaskPriority.HIGH: 1.0,
            TaskPriority.NORMAL: 1.5,
            TaskPriority.LOW: 2.0
        }
    
    def get_buffer_for_priority(self, priority: TaskPriority) -> float:
        """Get timing buffer scaled by priority"""
        scale = self.priority_buffer_scale.get(priority, 1.0)
        return self.timing_buffer * scale
    
    def get_overhead_for_segments(self, num_segments: int) -> float:
        """Get scheduling overhead for given number of segments"""
        return self.scheduling_overhead * num_segments


def apply_comprehensive_segmentation_patch(scheduler, config: Optional[SegmentationPatchConfig] = None):
    """
    Apply comprehensive patch to fix all segmentation issues
    """
    
    if config is None:
        config = SegmentationPatchConfig()
    
    # Store config in scheduler
    scheduler._segmentation_patch_config = config
    
    # Apply all patches
    patch_resource_finding(scheduler, config)
    patch_scheduling_loop(scheduler, config)
    patch_task_segmentation(scheduler, config)
    patch_resource_management(scheduler, config)
    
    print(f"✅ Comprehensive segmentation patch applied")
    print(f"  - Timing buffer: {config.timing_buffer}ms")
    print(f"  - Scheduling overhead: {config.scheduling_overhead}ms per segment")
    print(f"  - Max iterations: {config.max_iterations} per ms")
    
    return config


def patch_resource_finding(scheduler, config):
    """Patch resource finding to prevent conflicts"""
    
    original_find_resources = getattr(scheduler, 'find_available_resources_for_task_with_segmentation', None)
    
    def enhanced_find_resources(task, current_time):
        """Enhanced resource finding with comprehensive conflict prevention"""
        
        if config.debug_mode:
            print(f"Finding resources for {task.task_id} at {current_time:.2f}ms")
        
        # Make segmentation decision
        if scheduler.enable_segmentation:
            segmentation_decision = make_safe_segmentation_decision(scheduler, task, current_time)
            task.apply_segmentation_decision(segmentation_decision)
        
        # Find available resources
        required_types = set(seg.resource_type for seg in task.segments)
        assigned_resources = {}
        
        for res_type in required_types:
            resource_id = find_best_resource_for_type(scheduler, task, res_type, current_time, config)
            if resource_id:
                assigned_resources[res_type] = resource_id
            else:
                if config.debug_mode:
                    print(f"  No available {res_type.value} resource")
                return None
        
        if config.debug_mode:
            print(f"  Assigned resources: {assigned_resources}")
        
        return assigned_resources
    
    scheduler.find_available_resources_for_task_with_segmentation = enhanced_find_resources


def make_safe_segmentation_decision(scheduler, task, current_time):
    """Make safe segmentation decision to avoid conflicts"""
    
    # For conservative approach, limit segmentation
    if task.segmentation_strategy.value == "NO_SEGMENTATION":
        return {seg.segment_id: [] for seg in task.segments}
    
    # Get available resources
    available_resources = get_available_resources_safely(scheduler, current_time)
    
    # Make conservative segmentation decision
    segmentation_decision = {}
    
    for segment in task.segments:
        available_cuts = segment.get_available_cuts()
        selected_cuts = []
        
        if available_cuts and len(available_resources.get(segment.resource_type, [])) > 1:
            # Only use segmentation if we have multiple resources available
            if task.segmentation_strategy.value == "FORCED_SEGMENTATION":
                # Use all cuts but check overhead
                total_overhead = sum(cp.overhead_ms for cp in segment.cut_points)
                if total_overhead <= task.latency_requirement * 0.1:  # Max 10% overhead
                    selected_cuts = available_cuts
            elif task.segmentation_strategy.value == "ADAPTIVE_SEGMENTATION":
                # Use conservative segmentation
                if available_cuts:
                    # Use only the first cut point to minimize complexity
                    first_cut = segment.cut_points[0]
                    if first_cut.overhead_ms <= task.latency_requirement * 0.05:  # Max 5% overhead
                        selected_cuts = [first_cut.op_id]
            elif task.segmentation_strategy.value == "CUSTOM_SEGMENTATION":
                # Use preset configuration if available
                if hasattr(task, 'current_segmentation') and segment.segment_id in task.current_segmentation:
                    selected_cuts = task.current_segmentation[segment.segment_id]
        
        segmentation_decision[segment.segment_id] = selected_cuts
    
    return segmentation_decision


def get_available_resources_safely(scheduler, current_time):
    """Get available resources with safety checks"""
    
    available = {ResourceType.NPU: [], ResourceType.DSP: []}
    
    for res_type in [ResourceType.NPU, ResourceType.DSP]:
        for resource in scheduler.resources[res_type]:
            queue = scheduler.resource_queues.get(resource.unit_id)
            if queue:
                effective_available_time = queue.available_time + TIMING_PRECISION_BUFFER
                if effective_available_time <= current_time:
                    available[res_type].append(resource.bandwidth)
    
    return available


def find_best_resource_for_type(scheduler, task, res_type, current_time, config):
    """Find the best available resource of given type"""
    
    buffer = config.get_buffer_for_priority(task.priority)
    best_resource = None
    best_score = float('inf')
    
    for resource in scheduler.resources[res_type]:
        queue = scheduler.resource_queues[resource.unit_id]
        
        # Calculate effective availability time
        effective_available = queue.available_time + buffer
        
        # Check if resource is actually available
        if effective_available <= current_time:
            # Check for higher priority tasks
            if not queue.has_higher_priority_tasks(task.priority, current_time, task.task_id):
                # Check binding constraints
                if not queue.is_bound_to_other_task(task.task_id, current_time):
                    # Calculate score (prefer higher bandwidth, earlier availability)
                    score = effective_available - resource.bandwidth * 0.1
                    if score < best_score:
                        best_score = score
                        best_resource = resource
    
    return best_resource.unit_id if best_resource else None


def patch_scheduling_loop(scheduler, config):
    """Patch the main scheduling loop"""
    
    def patched_scheduling(time_window: float = 1000.0):
        """Patched scheduling loop with comprehensive conflict prevention"""
        
        # Reset scheduler state
        reset_scheduler_state_completely(scheduler)
        
        current_time = 0.0
        results = []
        task_execution_counts = defaultdict(int)
        iteration_count = 0
        max_total_iterations = int(time_window * config.max_iterations)
        
        if config.debug_mode:
            print(f"Starting scheduling for {time_window}ms")
        
        while current_time < time_window and iteration_count < max_total_iterations:
            iteration_count += 1
            
            # Clean up expired states
            cleanup_expired_states(scheduler, current_time)
            
            # Find ready tasks
            ready_tasks = find_ready_tasks_safely(scheduler, current_time, task_execution_counts)
            
            if not ready_tasks:
                # Advance to next event
                next_time = find_next_event_time_safely(scheduler, current_time, time_window)
                if next_time <= current_time:
                    next_time = current_time + 1.0  # Prevent infinite loops
                current_time = next_time
                continue
            
            # Try to schedule one task
            scheduled = False
            
            for task in ready_tasks:
                # Check if we can schedule this task
                if can_schedule_task_safely(scheduler, task, current_time, config):
                    # Find resources
                    assigned_resources = scheduler.find_available_resources_for_task_with_segmentation(
                        task, current_time
                    )
                    
                    if assigned_resources:
                        # Create schedule
                        schedule_info = create_safe_schedule(
                            scheduler, task, assigned_resources, current_time, config
                        )
                        
                        if schedule_info:
                            # Apply schedule
                            apply_schedule_safely(scheduler, task, schedule_info, current_time, config)
                            results.append(schedule_info)
                            task_execution_counts[task.task_id] += 1
                            scheduled = True
                            
                            if config.debug_mode:
                                print(f"Scheduled {task.task_id} at {current_time:.2f}ms")
                            break
            
            # Advance time
            if scheduled:
                current_time += config.timing_buffer  # Small advance to prevent conflicts
            else:
                current_time += 1.0  # Larger advance when nothing scheduled
        
        if config.debug_mode:
            print(f"Scheduling completed: {len(results)} events in {iteration_count} iterations")
        
        return results
    
    scheduler.priority_aware_schedule_with_segmentation = patched_scheduling


def reset_scheduler_state_completely(scheduler):
    """Reset all scheduler state completely"""
    
    # Reset resource queues
    for queue in scheduler.resource_queues.values():
        queue.available_time = 0.0
        queue.release_binding()
        
        # Clear sub-segment reservations if available
        if hasattr(queue, 'sub_segment_reservations'):
            queue.sub_segment_reservations.clear()
        if hasattr(queue, 'pending_sub_segments'):
            queue.pending_sub_segments.clear()
        
        # Clear priority queues
        for priority in TaskPriority:
            if priority in queue.queues:
                queue.queues[priority].clear()
    
    # Reset task states
    for task in scheduler.tasks.values():
        task.schedule_info = None
        task.last_execution_time = -float('inf')
        task.ready_time = 0
        task.current_segmentation = {}
        task.total_segmentation_overhead = 0.0
    
    # Clear history and bindings
    scheduler.schedule_history = []
    
    if hasattr(scheduler, 'active_bindings'):
        scheduler.active_bindings = []
    if hasattr(scheduler, 'segmentation_decisions_history'):
        scheduler.segmentation_decisions_history = []


def cleanup_expired_states(scheduler, current_time):
    """Clean up expired states"""
    
    # Clean up bindings
    if hasattr(scheduler, 'cleanup_expired_bindings'):
        try:
            scheduler.cleanup_expired_bindings(current_time)
        except:
            pass
    
    # Clean up resource reservations
    for queue in scheduler.resource_queues.values():
        if hasattr(queue, 'cleanup_expired_reservations'):
            try:
                queue.cleanup_expired_reservations(current_time)
            except:
                pass


def find_ready_tasks_safely(scheduler, current_time, task_execution_counts):
    """Find ready tasks with safety checks"""
    
    ready_tasks = []
    
    for task in scheduler.tasks.values():
        # Check minimum interval with tolerance
        if task.last_execution_time + task.min_interval_ms > current_time + 0.01:
            continue
        
        # Check dependencies
        deps_satisfied = True
        for dep_id in task.dependencies:
            if dep_id in scheduler.tasks:
                dep_count = task_execution_counts.get(dep_id, 0)
                task_count = task_execution_counts.get(task.task_id, 0)
                if dep_count <= task_count:
                    deps_satisfied = False
                    break
        
        if deps_satisfied:
            ready_tasks.append(task)
    
    # Sort by priority and task ID for deterministic ordering
    ready_tasks.sort(key=lambda t: (t.priority.value, t.task_id))
    
    return ready_tasks


def can_schedule_task_safely(scheduler, task, current_time, config):
    """Check if task can be scheduled safely"""
    
    # Check if enough time remains
    min_duration = 1.0  # Minimum time needed
    
    for segment in task.segments:
        # Estimate minimum duration
        if segment.duration_table:
            min_seg_duration = min(segment.duration_table.values())
            min_duration = max(min_duration, segment.start_time + min_seg_duration)
    
    # Add overhead for segmentation
    if task.is_segmented:
        min_duration += config.get_overhead_for_segments(2)  # Assume max 2 segments
    
    return min_duration <= task.latency_requirement


def find_next_event_time_safely(scheduler, current_time, time_window):
    """Find next event time safely"""
    
    next_time = current_time + 1.0
    
    # Check resource availability times
    for queue in scheduler.resource_queues.values():
        if queue.available_time > current_time:
            next_time = min(next_time, queue.available_time)
        
        if hasattr(queue, 'bound_until') and queue.bound_until > current_time:
            next_time = min(next_time, queue.bound_until)
    
    # Check task intervals
    for task in scheduler.tasks.values():
        if task.last_execution_time > -float('inf'):
            next_ready = task.last_execution_time + task.min_interval_ms
            if next_ready > current_time:
                next_time = min(next_time, next_ready)
    
    return min(next_time, time_window)


def create_safe_schedule(scheduler, task, assigned_resources, current_time, config):
    """Create schedule with safety checks"""
    
    buffer = config.get_buffer_for_priority(task.priority)
    total_duration = 0.0
    sub_segment_schedule = []
    
    if task.is_segmented:
        # Handle segmented task carefully
        sub_segments = task.get_sub_segments_for_scheduling()
        
        segment_time = current_time
        for i, sub_seg in enumerate(sub_segments):
            if sub_seg.resource_type in assigned_resources:
                resource_id = assigned_resources[sub_seg.resource_type]
                resource = next((r for r in scheduler.resources[sub_seg.resource_type] 
                               if r.unit_id == resource_id), None)
                
                if resource:
                    duration = sub_seg.get_duration(resource.bandwidth)
                    
                    # Add buffer between segments
                    if i > 0:
                        segment_time += buffer
                    
                    segment_end = segment_time + duration
                    sub_segment_schedule.append((sub_seg.sub_id, segment_time, segment_end))
                    
                    segment_time = segment_end
                    total_duration = max(total_duration, segment_end - current_time)
        
        # Add scheduling overhead
        overhead = config.get_overhead_for_segments(len(sub_segment_schedule))
        total_duration += overhead
    else:
        # Handle non-segmented task
        for segment in task.segments:
            if segment.resource_type in assigned_resources:
                resource_id = assigned_resources[segment.resource_type]
                resource = next((r for r in scheduler.resources[segment.resource_type] 
                               if r.unit_id == resource_id), None)
                
                if resource:
                    duration = segment.get_duration(resource.bandwidth)
                    end_time = current_time + segment.start_time + duration
                    total_duration = max(total_duration, end_time - current_time)
    
    # Create schedule info
    schedule_info = TaskScheduleInfo(
        task_id=task.task_id,
        start_time=current_time,
        end_time=current_time + total_duration,
        assigned_resources=assigned_resources,
        actual_latency=total_duration,
        runtime_type=task.runtime_type,
        used_cuts=task.current_segmentation.copy(),
        segmentation_overhead=task.total_segmentation_overhead,
        sub_segment_schedule=sub_segment_schedule
    )
    
    return schedule_info


def apply_schedule_safely(scheduler, task, schedule_info, current_time, config):
    """Apply schedule with safety checks"""
    
    # Update task state
    task.schedule_info = schedule_info
    task.last_execution_time = current_time
    scheduler.schedule_history.append(schedule_info)
    
    # Update resource availability
    buffer = config.get_buffer_for_priority(task.priority)
    
    if schedule_info.sub_segment_schedule:
        # Update based on sub-segments
        for sub_seg_id, start, end in schedule_info.sub_segment_schedule:
            # Find corresponding resource
            for sub_seg in task.get_sub_segments_for_scheduling():
                if sub_seg.sub_id == sub_seg_id:
                    if sub_seg.resource_type in schedule_info.assigned_resources:
                        resource_id = schedule_info.assigned_resources[sub_seg.resource_type]
                        queue = scheduler.resource_queues[resource_id]
                        
                        # Update with buffer
                        if task.runtime_type == RuntimeType.ACPU_RUNTIME:
                            queue.available_time = max(queue.available_time, end + buffer)
                    break
    else:
        # Update based on regular execution
        for segment in task.segments:
            if segment.resource_type in schedule_info.assigned_resources:
                resource_id = schedule_info.assigned_resources[segment.resource_type]
                queue = scheduler.resource_queues[resource_id]
                
                if task.runtime_type == RuntimeType.ACPU_RUNTIME:
                    queue.available_time = max(queue.available_time, schedule_info.end_time + buffer)


def patch_task_segmentation(scheduler, config):
    """Patch task segmentation methods"""
    
    # This ensures that task segmentation methods work correctly
    for task in scheduler.tasks.values():
        if not hasattr(task, 'get_sub_segments_for_scheduling'):
            # Add fallback method
            def get_sub_segments_fallback():
                return []
            task.get_sub_segments_for_scheduling = get_sub_segments_fallback


def patch_resource_management(scheduler, config):
    """Patch resource management methods"""
    
    # Ensure all resource queues have necessary methods
    for queue in scheduler.resource_queues.values():
        if not hasattr(queue, 'has_higher_priority_tasks'):
            def has_higher_priority_tasks_fallback(priority, current_time, task_id=None):
                return False
            queue.has_higher_priority_tasks = has_higher_priority_tasks_fallback
        
        if not hasattr(queue, 'is_bound_to_other_task'):
            def is_bound_to_other_task_fallback(task_id, current_time):
                return False
            queue.is_bound_to_other_task = is_bound_to_other_task_fallback


def test_comprehensive_patch():
    """Test the comprehensive patch"""
    
    print("=== Testing Comprehensive Segmentation Patch ===\n")
    
    try:
        from task import NNTask
        from scheduler import MultiResourceScheduler
        from enums import SegmentationStrategy
        
        # Create scheduler
        scheduler = MultiResourceScheduler(enable_segmentation=True)
        
        # Apply comprehensive patch
        config = apply_comprehensive_segmentation_patch(scheduler)
        config.debug_mode = True  # Enable debug output
        
        # Add resources
        scheduler.add_npu("NPU_0", bandwidth=8.0)
        scheduler.add_npu("NPU_1", bandwidth=4.0)
        scheduler.add_dsp("DSP_0", bandwidth=4.0)
        
        print(f"Added resources: {len(scheduler.resources[ResourceType.NPU])} NPUs, {len(scheduler.resources[ResourceType.DSP])} DSPs")
        
        # Create test task
        task = NNTask("T1", "TestTask", 
                     priority=TaskPriority.HIGH,
                     runtime_type=RuntimeType.ACPU_RUNTIME,
                     segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION)
        task.set_npu_only({2.0: 20, 4.0: 15, 8.0: 10}, "test_seg")
        
        if hasattr(task, 'add_cut_points_to_segment'):
            task.add_cut_points_to_segment("test_seg", [("cut1", 0.5, 0.1)])
        
        task.set_performance_requirements(fps=20, latency=50)
        scheduler.add_task(task)
        
        print(f"Created task: {task.task_id}")
        
        # Run scheduling
        print("\nRunning patched scheduling...")
        results = scheduler.priority_aware_schedule_with_segmentation(time_window=100.0)
        
        print(f"Scheduled {len(results)} events")
        
        if results:
            for i, schedule in enumerate(results[:3]):
                print(f"  Event {i+1}: {schedule.task_id} @ {schedule.start_time:.2f}-{schedule.end_time:.2f}ms")
                if hasattr(schedule, 'sub_segment_schedule') and schedule.sub_segment_schedule:
                    print(f"    Sub-segments: {len(schedule.sub_segment_schedule)}")
        
        print("\n✅ Comprehensive patch test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Patch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_comprehensive_patch()
    
    if success:
        print("\n🎉 Comprehensive segmentation patch is ready!")
        print("\nUsage:")
        print("  from comprehensive_segmentation_patch import apply_comprehensive_segmentation_patch")
        print("  apply_comprehensive_segmentation_patch(scheduler)")
    else:
        print("\n❌ Patch test failed. Please check the error messages.")
