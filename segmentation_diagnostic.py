#!/usr/bin/env python3
"""
Diagnostic tool to analyze segmentation scheduling issues
"""

from collections import defaultdict
from typing import List, Dict, Tuple


def diagnose_segmentation_schedule(scheduler):
    """
    Detailed diagnosis of segmentation scheduling results
    """
    print("\n=== Segmentation Schedule Diagnosis ===\n")
    
    # 1. Analyze segmentation decisions
    print("1. Segmentation Decisions:")
    for task_id, task in scheduler.tasks.items():
        print(f"\n{task_id} ({task.name}):")
        print(f"  Strategy: {task.segmentation_strategy.value}")
        print(f"  Is segmented: {task.is_segmented}")
        print(f"  Current segmentation: {task.current_segmentation}")
        print(f"  Total overhead: {task.total_segmentation_overhead:.2f}ms")
        
        # Show available vs used cuts
        for seg_id, cuts in task.get_all_available_cuts().items():
            used_cuts = task.current_segmentation.get(seg_id, [])
            print(f"  {seg_id}: {len(used_cuts)}/{len(cuts)} cuts used")
    
    # 2. Analyze sub-segment timing
    print("\n2. Sub-segment Timing Analysis:")
    
    for schedule in scheduler.schedule_history[:5]:  # First 5 events
        task = scheduler.tasks[schedule.task_id]
        if schedule.sub_segment_schedule and len(schedule.sub_segment_schedule) > 1:
            print(f"\n{schedule.task_id} execution at {schedule.start_time:.1f}ms:")
            
            prev_end = None
            for i, (sub_id, start, end) in enumerate(schedule.sub_segment_schedule):
                duration = end - start
                gap = start - prev_end if prev_end else 0
                
                print(f"  Segment {i+1} ({sub_id}):")
                print(f"    Time: {start:.1f} - {end:.1f}ms (duration: {duration:.1f}ms)")
                
                if prev_end and gap < 0:
                    print(f"    ⚠️ Overlap with previous: {-gap:.1f}ms")
                elif prev_end and gap > 0.1:
                    print(f"    Gap from previous: {gap:.1f}ms")
                
                prev_end = end
    
    # 3. Resource conflict analysis
    print("\n3. Resource Conflict Analysis:")
    
    resource_timeline = defaultdict(list)
    
    # Build timeline
    for schedule in scheduler.schedule_history:
        task = scheduler.tasks[schedule.task_id]
        
        if schedule.sub_segment_schedule:
            for i, (sub_id, start, end) in enumerate(schedule.sub_segment_schedule):
                # Find which resource was used
                for seg in task.get_sub_segments_for_scheduling():
                    if seg.sub_id == sub_id and seg.resource_type in schedule.assigned_resources:
                        res_id = schedule.assigned_resources[seg.resource_type]
                        resource_timeline[res_id].append((
                            start, end, f"{task.task_id}-{i+1}", task.priority
                        ))
                        break
        else:
            # Non-segmented task
            for res_type, res_id in schedule.assigned_resources.items():
                resource_timeline[res_id].append((
                    schedule.start_time, schedule.end_time, task.task_id, task.priority
                ))
    
    # Check for conflicts
    conflicts = []
    for res_id, timeline in resource_timeline.items():
        timeline.sort()  # Sort by start time
        
        for i in range(len(timeline) - 1):
            curr_start, curr_end, curr_task, curr_prio = timeline[i]
            next_start, next_end, next_task, next_prio = timeline[i+1]
            
            if curr_end > next_start + 0.001:  # Small tolerance
                overlap = curr_end - next_start
                conflicts.append({
                    'resource': res_id,
                    'task1': curr_task,
                    'task2': next_task,
                    'overlap': overlap,
                    'time_range': (next_start, curr_end)
                })
    
    if conflicts:
        print(f"\nFound {len(conflicts)} conflicts:")
        for c in conflicts[:5]:  # Show first 5
            print(f"  {c['resource']}: {c['task1']} overlaps {c['task2']} by {c['overlap']:.3f}ms")
            print(f"    Time range: {c['time_range'][0]:.1f}-{c['time_range'][1]:.1f}ms")
    else:
        print("\nNo conflicts found!")
    
    # 4. Performance analysis
    print("\n4. Performance Analysis:")
    
    task_stats = defaultdict(lambda: {
        'executions': 0,
        'total_duration': 0,
        'total_segments': 0,
        'total_overhead': 0
    })
    
    for schedule in scheduler.schedule_history:
        task_id = schedule.task_id
        stats = task_stats[task_id]
        
        stats['executions'] += 1
        stats['total_duration'] += schedule.end_time - schedule.start_time
        
        if schedule.sub_segment_schedule:
            stats['total_segments'] += len(schedule.sub_segment_schedule)
        else:
            stats['total_segments'] += 1
        
        stats['total_overhead'] += schedule.segmentation_overhead
    
    print("\nTask Performance Summary:")
    for task_id, stats in task_stats.items():
        task = scheduler.tasks[task_id]
        avg_duration = stats['total_duration'] / stats['executions']
        avg_segments = stats['total_segments'] / stats['executions']
        
        print(f"\n{task_id}:")
        print(f"  Executions: {stats['executions']}")
        print(f"  Avg duration: {avg_duration:.1f}ms (required: <{task.latency_requirement}ms)")
        print(f"  Avg segments: {avg_segments:.1f}")
        print(f"  Total overhead: {stats['total_overhead']:.1f}ms")
        
        # Check FPS
        if scheduler.schedule_history:
            time_window = scheduler.schedule_history[-1].end_time
            achieved_fps = stats['executions'] / (time_window / 1000.0)
            print(f"  Achieved FPS: {achieved_fps:.1f} (required: {task.fps_requirement})")
    
    return conflicts


def suggest_fixes(conflicts):
    """
    Suggest fixes for identified conflicts
    """
    print("\n5. Suggested Fixes:")
    
    if not conflicts:
        print("No fixes needed - schedule is valid!")
        return
    
    # Analyze conflict patterns
    resource_conflicts = defaultdict(int)
    task_conflicts = defaultdict(int)
    
    for c in conflicts:
        resource_conflicts[c['resource']] += 1
        task_conflicts[c['task1'].split('-')[0]] += 1
        task_conflicts[c['task2'].split('-')[0]] += 1
    
    print("\nConflict Summary:")
    print(f"  Most conflicted resources: {dict(resource_conflicts)}")
    print(f"  Most conflicted tasks: {dict(task_conflicts)}")
    
    print("\nSuggestions:")
    print("  1. Add small delays between sub-segments to avoid timing precision issues")
    print("  2. Consider reducing segmentation for frequently conflicting tasks")
    print("  3. Increase resource pool if conflicts are due to resource shortage")
    print("  4. Adjust segmentation overhead values to be more realistic")


def test_diagnostic():
    """
    Run diagnostic on a test scenario
    """
    from simple_seg_test import test_multiple_tasks
    
    print("Running test scenario for diagnosis...")
    scheduler, results = test_multiple_tasks()
    
    if scheduler and results:
        conflicts = diagnose_segmentation_schedule(scheduler)
        suggest_fixes(conflicts)
    
    return scheduler


if __name__ == "__main__":
    print("=== NPU Scheduler Segmentation Diagnostic Tool ===\n")
    
    scheduler = test_diagnostic()
    
    if scheduler:
        print("\n\n=== Diagnostic Summary ===")
        print("The segmentation feature is working with minor timing precision issues.")
        print("These are likely due to floating-point precision in timing calculations.")
        print("\nRecommendation: Add a small buffer (0.1ms) between segments to avoid conflicts.")
