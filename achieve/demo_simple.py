#!/usr/bin/env python3
"""
Simple demonstration of the fixed scheduler
A minimal example to test the patched scheduler
"""

from enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from task import NNTask
from scheduler import MultiResourceScheduler
from elegant_visualization import ElegantSchedulerVisualizer
from scheduler_patch import patch_scheduler
from schedule_validator import validate_schedule

def create_simple_scenario():
    """Create a simple test scenario"""
    # Create scheduler (segmentation disabled)
    scheduler = MultiResourceScheduler(enable_segmentation=False)
    
    # Apply patch
    patch_scheduler(scheduler)
    
    # Add resources
    scheduler.add_npu("NPU_0", bandwidth=8.0)
    scheduler.add_npu("NPU_1", bandwidth=8.0)
    scheduler.add_dsp("DSP_0", bandwidth=4.0)
    
    # Create a few simple tasks
    tasks = []
    
    # Critical task
    task1 = NNTask("T1", "CriticalTask", 
                   priority=TaskPriority.CRITICAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task1.set_npu_only({8.0: 10}, "seg1")
    task1.set_performance_requirements(fps=30, latency=35)
    tasks.append(task1)
    
    # High priority task
    task2 = NNTask("T2", "HighPriorityTask",
                   priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task2.set_npu_only({8.0: 15}, "seg2")
    task2.set_performance_requirements(fps=20, latency=50)
    tasks.append(task2)
    
    # Normal priority task with DSP
    task3 = NNTask("T3", "MixedTask",
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task3.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 5}, 0, "dsp_seg"),
        (ResourceType.NPU, {8.0: 10}, 5, "npu_seg"),
    ])
    task3.set_performance_requirements(fps=10, latency=100)
    tasks.append(task3)
    
    # Add tasks to scheduler
    for task in tasks:
        scheduler.add_task(task)
    
    return scheduler, tasks

def main():
    """Run simple demonstration"""
    print("=== Simple Scheduler Demo (Fixed Version) ===\n")
    
    # Create scenario
    scheduler, tasks = create_simple_scenario()
    
    print(f"Created {len(tasks)} tasks:")
    for task in tasks:
        print(f"  {task.task_id}: {task.name} ({task.priority.name})")
    
    print(f"\nResources:")
    for res_type, resources in scheduler.resources.items():
        for res in resources:
            print(f"  {res.unit_id} (bandwidth={res.bandwidth})")
    
    # Run scheduling
    print("\nRunning scheduling algorithm...")
    results = scheduler.priority_aware_schedule_with_segmentation(time_window=200.0)
    
    print(f"\nScheduled {len(results)} events")
    
    # Validate
    is_valid, errors = validate_schedule(scheduler)
    if is_valid:
        print("✅ No resource conflicts detected!")
    else:
        print(f"❌ Found {len(errors)} conflicts")
        for error in errors[:3]:
            print(f"  - {error}")
    
    # Print schedule
    print("\nSchedule:")
    print(f"{'Time':>8} {'Task':<10} {'Resource':<10} {'Duration':<10}")
    print("-" * 45)
    
    for event in results[:20]:  # Show first 20 events
        task = scheduler.tasks[event.task_id]
        for res_type, res_id in event.assigned_resources.items():
            duration = event.end_time - event.start_time
            print(f"{event.start_time:8.1f} {task.task_id:<10} {res_id:<10} {duration:<10.1f}")
    
    # Visualize
    print("\nGenerating visualization...")
    visualizer = ElegantSchedulerVisualizer(scheduler)
    visualizer.plot_elegant_gantt(bar_height=0.4, spacing=1.0)
    visualizer.export_chrome_tracing("simple_demo_trace.json")
    
    print("\n✅ Demo completed successfully!")
    print("Check the generated Gantt chart and simple_demo_trace.json")

if __name__ == "__main__":
    main()
