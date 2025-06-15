#!/usr/bin/env python3
"""
Neural Network Task Resource Structure - Enhanced Main Application

This module demonstrates the usage of the multi-resource scheduler with runtime configurations.
"""

from enums import ResourceType, TaskPriority, RuntimeType
from task import NNTask
from scheduler import MultiResourceScheduler
from visualization import SchedulerVisualizer

def create_sample_tasks_with_runtime():
    """Create sample tasks with different runtime configurations for demonstration"""
    tasks = []
    
    # CRITICAL priority task - DSP_Runtime (bound execution)
    task1 = NNTask("T1", "SafetyMonitor", priority=TaskPriority.CRITICAL, 
                   runtime_type=RuntimeType.DSP_RUNTIME)
    task1.set_npu_only({2.0: 20, 4.0: 12, 8.0: 8})
    task1.set_performance_requirements(fps=30, latency=30)
    tasks.append(task1)
    
    # HIGH priority tasks - Mixed runtime types
    task2 = NNTask("T2", "ObstacleDetection", priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.DSP_RUNTIME)  # Bound DSP+NPU execution
    task2.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 8, 8.0: 5}, 0),
        (ResourceType.NPU, {2.0: 25, 4.0: 15, 8.0: 10}, 8),
    ])
    task2.set_performance_requirements(fps=20, latency=50)
    tasks.append(task2)
    
    task3 = NNTask("T3", "LaneDetection", priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.ACPU_RUNTIME)  # Pipelined execution
    task3.set_npu_only({2.0: 30, 4.0: 18, 8.0: 12})
    task3.set_performance_requirements(fps=15, latency=60)
    task3.add_dependency("T1")  # Depends on safety monitor
    tasks.append(task3)
    
    # NORMAL priority tasks - Different runtime configurations
    task4 = NNTask("T4", "TrafficSignRecog", priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.DSP_RUNTIME)  # Bound multi-segment execution
    task4.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 10}, 0),
        (ResourceType.NPU, {2.0: 20, 4.0: 12, 8.0: 8}, 10),
        (ResourceType.DSP, {4.0: 5}, 22),
    ])
    task4.set_performance_requirements(fps=10, latency=80)
    tasks.append(task4)
    
    task5 = NNTask("T5", "PedestrianTracking", priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME)  # Pipelined execution
    task5.set_npu_only({2.0: 35, 4.0: 20, 8.0: 12})
    task5.set_performance_requirements(fps=10, latency=100)
    tasks.append(task5)
    
    # LOW priority tasks - Mixed runtime types
    task6 = NNTask("T6", "SceneUnderstanding", priority=TaskPriority.LOW,
                   runtime_type=RuntimeType.ACPU_RUNTIME)  # Pipelined
    task6.set_npu_only({2.0: 50, 4.0: 30, 8.0: 20})
    task6.set_performance_requirements(fps=5, latency=200)
    tasks.append(task6)
    
    task7 = NNTask("T7", "MapUpdate", priority=TaskPriority.LOW,
                   runtime_type=RuntimeType.DSP_RUNTIME)  # Bound execution
    task7.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 15}, 0),
        (ResourceType.NPU, {2.0: 40, 4.0: 25, 8.0: 15}, 15),
    ])
    task7.set_performance_requirements(fps=2, latency=500)
    tasks.append(task7)
    
    # Additional tasks to show runtime effects
    for i in range(8, 12):
        priority = TaskPriority.NORMAL if i % 2 == 0 else TaskPriority.LOW
        runtime_type = RuntimeType.DSP_RUNTIME if i % 3 == 0 else RuntimeType.ACPU_RUNTIME
        
        task = NNTask(f"T{i}", f"Task_{i}", priority=priority, runtime_type=runtime_type)
        
        if runtime_type == RuntimeType.DSP_RUNTIME and i % 4 == 0:
            # Some DSP_Runtime tasks with mixed resources
            task.set_dsp_npu_sequence([
                (ResourceType.DSP, {4.0: 8}, 0),
                (ResourceType.NPU, {2.0: 20+i*2, 4.0: 15+i, 8.0: 10+i//2}, 8),
            ])
        else:
            # NPU-only tasks
            task.set_npu_only({2.0: 20+i*2, 4.0: 15+i, 8.0: 10+i//2})
            
        task.set_performance_requirements(fps=8, latency=150)
        tasks.append(task)
    
    return tasks

def setup_scheduler():
    """Setup scheduler with resources"""
    scheduler = MultiResourceScheduler()
    
    # Add multiple NPU resources (different bandwidths)
    scheduler.add_npu("NPU_0", bandwidth=8.0)  # High performance NPU
    scheduler.add_npu("NPU_1", bandwidth=4.0)  # Medium performance NPU
    scheduler.add_npu("NPU_2", bandwidth=2.0)  # Low performance NPU
    
    # Add DSP resources
    scheduler.add_dsp("DSP_0", bandwidth=4.0)
    scheduler.add_dsp("DSP_1", bandwidth=4.0)
    
    return scheduler

def analyze_runtime_performance(scheduler: MultiResourceScheduler, schedule_results):
    """Analyze performance differences between runtime types"""
    print("\n=== Runtime Performance Analysis ===")
    
    dsp_runtime_tasks = [t for t in scheduler.tasks.values() if t.runtime_type == RuntimeType.DSP_RUNTIME]
    acpu_runtime_tasks = [t for t in scheduler.tasks.values() if t.runtime_type == RuntimeType.ACPU_RUNTIME]
    
    print(f"DSP_Runtime tasks: {len(dsp_runtime_tasks)}")
    print(f"ACPU_Runtime tasks: {len(acpu_runtime_tasks)}")
    
    # Count scheduling frequency by runtime type
    dsp_schedules = [s for s in schedule_results if s.runtime_type == RuntimeType.DSP_RUNTIME]
    acpu_schedules = [s for s in schedule_results if s.runtime_type == RuntimeType.ACPU_RUNTIME]
    
    print(f"DSP_Runtime scheduling events: {len(dsp_schedules)}")
    print(f"ACPU_Runtime scheduling events: {len(acpu_schedules)}")
    
    if dsp_schedules:
        avg_dsp_latency = sum(s.actual_latency for s in dsp_schedules) / len(dsp_schedules)
        print(f"Average DSP_Runtime latency: {avg_dsp_latency:.2f}ms")
    
    if acpu_schedules:
        avg_acpu_latency = sum(s.actual_latency for s in acpu_schedules) / len(acpu_schedules)
        print(f"Average ACPU_Runtime latency: {avg_acpu_latency:.2f}ms")
    
    # Analyze resource binding effectiveness
    binding_tasks = [t for t in dsp_runtime_tasks if t.requires_resource_binding()]
    print(f"Tasks requiring resource binding: {len(binding_tasks)}")
    
    for binding in scheduler.active_bindings:
        binding_duration = binding.binding_end - binding.binding_start
        print(f"  Binding for {binding.task_id}: {binding_duration:.1f}ms duration, "
              f"{len(binding.bound_resources)} resources")

def main():
    """Enhanced main application entry point with runtime configurations"""
    print("=== Enhanced Neural Network Task Resource Structure Demo ===")
    print("Features: Priority-aware scheduling with Runtime Configurations")
    print("- DSP_Runtime: Bound resource execution (no interruption)")
    print("- ACPU_Runtime: Pipelined execution (allows task interleaving)")
    
    # Create scheduler
    scheduler = setup_scheduler()
    
    # Add sample tasks with runtime configurations
    tasks = create_sample_tasks_with_runtime()
    for task in tasks:
        scheduler.add_task(task)
    
    print(f"\nSetup complete:")
    print(f"- NPU Resources: {len(scheduler.resources[ResourceType.NPU])}")
    print(f"- DSP Resources: {len(scheduler.resources[ResourceType.DSP])}")
    print(f"- Total Tasks: {len(scheduler.tasks)}")
    
    # Show runtime type distribution
    runtime_counts = {}
    for runtime_type in RuntimeType:
        count = sum(1 for task in scheduler.tasks.values() if task.runtime_type == runtime_type)
        runtime_counts[runtime_type.value] = count
        print(f"- {runtime_type.value}: {count} tasks")
    
    # Execute priority-aware scheduling with runtime configurations
    print("\nStarting priority-aware scheduling with runtime configurations...")
    schedule_results = scheduler.priority_aware_schedule(time_window=500.0)
    
    # Print results
    scheduler.print_schedule_summary()
    
    # Create visualizer and generate plots
    visualizer = SchedulerVisualizer(scheduler)
    
    # Plot task overview with runtime information
    print("\nGenerating enhanced task overview plot...")
    try:
        visualizer.plot_task_overview_with_runtime(selected_bw=4.0)
    except Exception as e:
        print(f"Could not generate task overview plot: {e}")
    
    # Plot scheduling Gantt chart with runtime visualization
    print("\nGenerating enhanced scheduling Gantt chart...")
    try:
        visualizer.plot_pipeline_schedule_with_runtime(time_window=300.0)  # Show first 300ms
    except Exception as e:
        print(f"Could not generate Gantt chart: {e}")
    
    # Print first 20 scheduling events with runtime info
    print("\nFirst 20 scheduling events with runtime configurations:")
    for i, schedule in enumerate(schedule_results[:20]):
        task = scheduler.tasks[schedule.task_id]
        runtime_symbol = 'B' if task.runtime_type == RuntimeType.DSP_RUNTIME else 'P'
        print(f"{i+1:2d}. [{task.priority.name:8s}] {task.name:20s} ({runtime_symbol}) @ "
              f"{schedule.start_time:5.1f}-{schedule.end_time:5.1f}ms, "
              f"Resources: {list(schedule.assigned_resources.values())}")
    
    # Analyze runtime performance differences
    analyze_runtime_performance(scheduler, schedule_results)
    
    print("\nEnhanced demo completed successfully!")
    print("\nKey observations:")
    print("- 'B' tasks (DSP_Runtime) show bound resource execution")
    print("- 'P' tasks (ACPU_Runtime) allow pipelined resource sharing")
    print("- Diagonal hatching in plots indicates DSP_Runtime tasks")
    print("- Vertical dashed lines show resource binding periods")

if __name__ == "__main__":
    main()