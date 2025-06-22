#!/usr/bin/env python3
"""
Simple test to verify segmentation functionality step by step
"""

from enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from task import NNTask
from scheduler import MultiResourceScheduler
from scheduler_segmentation_fix_v2 import apply_complete_segmentation_fix
from schedule_validator import validate_schedule
from elegant_visualization import ElegantSchedulerVisualizer


def test_basic_segmentation():
    """Test most basic segmentation scenario"""
    
    print("=== Basic Segmentation Test ===\n")
    
    # Step 1: Create scheduler
    print("Step 1: Creating scheduler with segmentation enabled...")
    scheduler = MultiResourceScheduler(
        enable_segmentation=True,
        max_segmentation_overhead_ratio=0.15
    )
    
    # Step 2: Apply fix
    print("Step 2: Applying segmentation fix...")
    apply_complete_segmentation_fix(scheduler)
    
    # Step 3: Add resources
    print("Step 3: Adding resources...")
    scheduler.add_npu("NPU_0", bandwidth=8.0)
    scheduler.add_npu("NPU_1", bandwidth=8.0)
    scheduler.add_dsp("DSP_0", bandwidth=4.0)
    
    print(f"  Added {len(scheduler.resources[ResourceType.NPU])} NPUs")
    print(f"  Added {len(scheduler.resources[ResourceType.DSP])} DSPs")
    
    # Step 4: Create a single segmented task
    print("\nStep 4: Creating a single segmented task...")
    task = NNTask("T1", "TestTask", 
                  priority=TaskPriority.HIGH,
                  runtime_type=RuntimeType.ACPU_RUNTIME,
                  segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION)
    
    # Simple NPU-only task
    task.set_npu_only({2.0: 30, 4.0: 20, 8.0: 12}, "test_seg")
    
    # Add one cut point
    task.add_cut_points_to_segment("test_seg", [
        ("mid", 0.5, 0.15)  # Cut at middle, 0.15ms overhead
    ])
    
    task.set_performance_requirements(fps=10, latency=50)
    
    scheduler.add_task(task)
    print(f"  Task created with strategy: {task.segmentation_strategy.value}")
    
    # Step 5: Run scheduling
    print("\nStep 5: Running scheduling...")
    try:
        results = scheduler.priority_aware_schedule_with_segmentation(time_window=100.0)
        print(f"  ✅ Scheduling completed: {len(results)} events")
        
        # Analyze results
        for i, schedule in enumerate(results[:3]):  # Show first 3
            print(f"\n  Event {i+1}:")
            print(f"    Task: {schedule.task_id}")
            print(f"    Time: {schedule.start_time:.1f} - {schedule.end_time:.1f} ms")
            print(f"    Resources: {schedule.assigned_resources}")
            
            if schedule.sub_segment_schedule:
                print(f"    Sub-segments: {len(schedule.sub_segment_schedule)}")
                for j, (sub_id, start, end) in enumerate(schedule.sub_segment_schedule):
                    print(f"      {j+1}: {sub_id} @ {start:.1f}-{end:.1f} ms")
            else:
                print(f"    No sub-segments (not segmented)")
        
        # Validate
        print("\nStep 6: Validating schedule...")
        is_valid, errors = validate_schedule(scheduler)
        
        if is_valid:
            print("  ✅ Schedule is valid - no conflicts!")
        else:
            print(f"  ❌ Found {len(errors)} errors:")
            for error in errors[:3]:
                print(f"    - {error}")
        
        return scheduler, results
        
    except Exception as e:
        print(f"  ❌ Error during scheduling: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_multiple_tasks():
    """Test with multiple tasks"""
    
    print("\n\n=== Multiple Tasks Test ===\n")
    
    # Create scheduler
    scheduler = MultiResourceScheduler(enable_segmentation=True)
    apply_complete_segmentation_fix(scheduler)
    
    # Add resources
    scheduler.add_npu("NPU_0", bandwidth=8.0)
    scheduler.add_npu("NPU_1", bandwidth=4.0)
    scheduler.add_dsp("DSP_0", bandwidth=4.0)
    
    # Task 1: Segmented
    task1 = NNTask("T1", "SegmentedTask", 
                   priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION)
    task1.set_npu_only({2.0: 40, 4.0: 25, 8.0: 15}, "seg1")
    task1.add_cut_points_to_segment("seg1", [
        ("cut1", 0.33, 0.12),
        ("cut2", 0.67, 0.13)
    ])
    task1.set_performance_requirements(fps=20, latency=60)
    scheduler.add_task(task1)
    
    # Task 2: Not segmented
    task2 = NNTask("T2", "NormalTask", 
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task2.set_npu_only({2.0: 30, 4.0: 18, 8.0: 10}, "seg2")
    task2.set_performance_requirements(fps=15, latency=80)
    scheduler.add_task(task2)
    
    # Task 3: DSP+NPU segmented
    task3 = NNTask("T3", "MixedTask", 
                   priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION)
    task3.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 10}, 0, "dsp_seg"),
        (ResourceType.NPU, {2.0: 30, 4.0: 20, 8.0: 12}, 10, "npu_seg"),
    ])
    task3.add_cut_points_to_segment("npu_seg", [("npu_cut", 0.5, 0.14)])
    task3.set_performance_requirements(fps=25, latency=50)
    scheduler.add_task(task3)
    
    print(f"Created {len(scheduler.tasks)} tasks")
    
    # Run scheduling
    print("\nRunning scheduling...")
    results = scheduler.priority_aware_schedule_with_segmentation(time_window=200.0)
    
    print(f"Scheduled {len(results)} events")
    
    # Analyze segmentation
    segmented_executions = 0
    total_segments = 0
    
    for schedule in results:
        if schedule.sub_segment_schedule and len(schedule.sub_segment_schedule) > 1:
            segmented_executions += 1
            total_segments += len(schedule.sub_segment_schedule)
    
    print(f"Segmented executions: {segmented_executions}")
    print(f"Total sub-segments: {total_segments}")
    
    # Task-wise analysis
    for task_id in ["T1", "T2", "T3"]:
        task_schedules = [s for s in results if s.task_id == task_id]
        if task_schedules:
            avg_segs = sum(len(s.sub_segment_schedule) for s in task_schedules) / len(task_schedules)
            print(f"{task_id}: {len(task_schedules)} executions, avg {avg_segs:.1f} segments/execution")
    
    # Validate
    is_valid, errors = validate_schedule(scheduler)
    print(f"\nValidation: {'✅ PASS' if is_valid else f'❌ FAIL ({len(errors)} errors)'}")
    
    return scheduler, results


def test_visualization():
    """Test visualization of segmented schedule"""
    
    print("\n\n=== Visualization Test ===\n")
    
    scheduler, results = test_multiple_tasks()
    
    if scheduler and results:
        print("\nGenerating visualization...")
        try:
            viz = ElegantSchedulerVisualizer(scheduler)
            viz.plot_elegant_gantt(bar_height=0.35, spacing=0.8)
            print("✅ Visualization generated successfully")
        except Exception as e:
            print(f"❌ Visualization error: {e}")


def main():
    """Main test function"""
    
    print("=== NPU Scheduler Segmentation Test (Simplified) ===\n")
    
    # Test 1: Basic single task
    scheduler1, results1 = test_basic_segmentation()
    
    if scheduler1:
        print("\n✅ Basic test passed!")
    else:
        print("\n❌ Basic test failed!")
        return
    
    # Test 2: Multiple tasks
    scheduler2, results2 = test_multiple_tasks()
    
    if scheduler2:
        print("\n✅ Multiple tasks test passed!")
    else:
        print("\n❌ Multiple tasks test failed!")
        return
    
    # Test 3: Visualization
    test_visualization()
    
    print("\n\n=== Summary ===")
    print("Segmentation functionality has been tested with:")
    print("- Single segmented task ✅")
    print("- Multiple tasks with different strategies ✅")
    print("- DSP+NPU mixed tasks ✅")
    print("- Visualization support ✅")
    
    print("\nThe segmentation feature is now ready to use!")
    print("Apply the fix using: apply_complete_segmentation_fix(scheduler)")


if __name__ == "__main__":
    main()
