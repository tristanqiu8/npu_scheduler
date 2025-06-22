#!/usr/bin/env python3
"""
Enhanced Neural Network Task Resource Structure - Fixed Version
This version includes the scheduler patch to prevent resource conflicts
"""

from enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from task import NNTask
from scheduler import MultiResourceScheduler
from visualization import SchedulerVisualizer
from elegant_visualization import ElegantSchedulerVisualizer
from scheduler_patch import patch_scheduler

def create_sample_tasks():
    """Create sample tasks for demonstration (segmentation disabled for now)"""
    tasks = []
    
    # CRITICAL priority task - DSP_Runtime
    task1 = NNTask("T1", "SafetyMonitor", priority=TaskPriority.CRITICAL, 
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task1.set_npu_only({2.0: 20, 4.0: 12, 8.0: 8}, "safety_npu_seg")
    task1.set_performance_requirements(fps=30, latency=30)
    tasks.append(task1)
    
    # HIGH priority task - DSP_Runtime with DSP+NPU
    task2 = NNTask("T2", "ObstacleDetection", priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task2.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 8, 8.0: 5}, 0, "obstacle_dsp_seg"),
        (ResourceType.NPU, {2.0: 25, 4.0: 15, 8.0: 10}, 8, "obstacle_npu_seg"),
    ])
    task2.set_performance_requirements(fps=20, latency=50)
    tasks.append(task2)
    
    # HIGH priority task - ACPU_Runtime
    task3 = NNTask("T3", "LaneDetection", priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task3.set_npu_only({2.0: 30, 4.0: 18, 8.0: 12}, "lane_npu_seg")
    task3.set_performance_requirements(fps=15, latency=60)
    task3.add_dependency("T1")  # Depends on safety monitor
    tasks.append(task3)
    
    # NORMAL priority task
    task4 = NNTask("T4", "TrafficSignRecog", priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task4.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 10}, 0, "sign_dsp_seg"),
        (ResourceType.NPU, {2.0: 20, 4.0: 12, 8.0: 8}, 10, "sign_npu_seg"),
        (ResourceType.DSP, {4.0: 5}, 22, "sign_dsp_post_seg"),
    ])
    task4.set_performance_requirements(fps=10, latency=80)
    tasks.append(task4)
    
    # NORMAL priority task
    task5 = NNTask("T5", "PedestrianTracking", priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task5.set_npu_only({2.0: 35, 4.0: 20, 8.0: 12}, "pedestrian_npu_seg")
    task5.set_performance_requirements(fps=10, latency=100)
    tasks.append(task5)
    
    # LOW priority task
    task6 = NNTask("T6", "SceneUnderstanding", priority=TaskPriority.LOW,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task6.set_npu_only({2.0: 50, 4.0: 30, 8.0: 20}, "scene_npu_seg")
    task6.set_performance_requirements(fps=5, latency=200)
    tasks.append(task6)
    
    # LOW priority task
    task7 = NNTask("T7", "MapUpdate", priority=TaskPriority.LOW,
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task7.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 15}, 0, "map_dsp_seg"),
        (ResourceType.NPU, {2.0: 40, 4.0: 25, 8.0: 15}, 15, "map_npu_seg"),
    ])
    task7.set_performance_requirements(fps=2, latency=500)
    tasks.append(task7)
    
    # Additional background tasks
    for i in range(8, 12):
        priority = TaskPriority.NORMAL if i % 2 == 0 else TaskPriority.LOW
        runtime_type = RuntimeType.DSP_RUNTIME if i % 3 == 0 else RuntimeType.ACPU_RUNTIME
        
        task = NNTask(f"T{i}", f"Task_{i}", priority=priority, runtime_type=runtime_type,
                      segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
        
        if runtime_type == RuntimeType.DSP_RUNTIME and i % 4 == 0:
            task.set_dsp_npu_sequence([
                (ResourceType.DSP, {4.0: 8}, 0, f"task{i}_dsp_seg"),
                (ResourceType.NPU, {2.0: 20+i*2, 4.0: 15+i, 8.0: 10+i//2}, 8, f"task{i}_npu_seg"),
            ])
        else:
            task.set_npu_only({2.0: 20+i*2, 4.0: 15+i, 8.0: 10+i//2}, f"task{i}_npu_seg")
            
        task.set_performance_requirements(fps=8, latency=150)
        tasks.append(task)
    
    return tasks

def setup_scheduler():
    """Setup scheduler with fixes applied"""
    # Create scheduler with segmentation disabled until fully fixed
    scheduler = MultiResourceScheduler(
        enable_segmentation=False,
        max_segmentation_overhead_ratio=0.15
    )
    
    # Apply the patch to fix resource conflicts
    patch_scheduler(scheduler)
    
    # Add NPU resources (different bandwidths)
    scheduler.add_npu("NPU_0", bandwidth=8.0)  # High performance NPU
    scheduler.add_npu("NPU_1", bandwidth=4.0)  # Medium performance NPU
    scheduler.add_npu("NPU_2", bandwidth=2.0)  # Low performance NPU
    
    # Add DSP resources
    scheduler.add_dsp("DSP_0", bandwidth=4.0)
    scheduler.add_dsp("DSP_1", bandwidth=4.0)
    
    return scheduler

def main():
    """Main application entry point"""
    print("=== Enhanced Neural Network Task Scheduler (Fixed Version) ===")
    print("Features:")
    print("- ✅ Resource conflict issue fixed")
    print("- ✅ Priority-aware scheduling")
    print("- ✅ Runtime type support (DSP_Runtime & ACPU_Runtime)")
    print("- ⚠️  Network segmentation temporarily disabled")
    print("- ✅ Enhanced visualization")
    
    # Create scheduler with fixes
    scheduler = setup_scheduler()
    
    # Add sample tasks
    tasks = create_sample_tasks()
    for task in tasks:
        scheduler.add_task(task)
    
    print(f"\nSetup complete:")
    print(f"- NPU Resources: {len(scheduler.resources[ResourceType.NPU])}")
    print(f"- DSP Resources: {len(scheduler.resources[ResourceType.DSP])}")
    print(f"- Total Tasks: {len(scheduler.tasks)}")
    print(f"- Segmentation enabled: {scheduler.enable_segmentation}")
    
    # Show task configuration
    print("\nTask Configuration:")
    for task in scheduler.tasks.values():
        print(f"  {task.task_id} ({task.name}) - {task.priority.name} - {task.runtime_type.value}")
    
    # Execute scheduling
    print("\nStarting scheduling algorithm...")
    schedule_results = scheduler.priority_aware_schedule_with_segmentation(time_window=400.0)
    
    # Print results
    print(f"\nScheduling complete!")
    print(f"Total scheduled events: {len(schedule_results)}")
    
    # Validate results
    try:
        from schedule_validator import validate_schedule
        is_valid, errors = validate_schedule(scheduler)
        if is_valid:
            print("✅ Schedule validation passed - no resource conflicts!")
        else:
            print(f"⚠️ Found {len(errors)} validation errors")
    except ImportError:
        print("ℹ️ Schedule validator not found, skipping validation")
    
    # Print summary
    scheduler.print_schedule_summary()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Use elegant visualizer
    try:
        visualizer = ElegantSchedulerVisualizer(scheduler)
        
        # Generate Gantt chart
        print("Creating elegant Gantt chart...")
        visualizer.plot_elegant_gantt(
            bar_height=0.35,
            spacing=0.8,
            use_alt_colors=False
        )
        
        # Export Chrome Tracing
        print("Exporting Chrome Tracing format...")
        visualizer.export_chrome_tracing("schedule_trace.json")
        print(f"✅ Chrome tracing data exported to schedule_trace.json")
        print(f"   Open chrome://tracing and load this file to visualize")
        
    except ImportError:
        print("Using standard visualizer...")
        visualizer = SchedulerVisualizer(scheduler)
        visualizer.plot_pipeline_schedule(time_window=250.0)
    
    # Print first few scheduling events
    print("\nFirst 15 scheduling events:")
    for i, schedule in enumerate(schedule_results[:15]):
        task = scheduler.tasks[schedule.task_id]
        runtime_symbol = 'B' if task.runtime_type == RuntimeType.DSP_RUNTIME else 'P'
        
        print(f"{i+1:2d}. [{task.priority.name:8s}] {task.name:20s} ({runtime_symbol}) @ "
              f"{schedule.start_time:5.1f}-{schedule.end_time:5.1f}ms")
    
    print("\nScheduling completed successfully!")
    print("\nNote: Network segmentation is temporarily disabled.")
    print("Once the segmentation scheduling logic is fully fixed, it can be re-enabled.")

if __name__ == "__main__":
    main()
