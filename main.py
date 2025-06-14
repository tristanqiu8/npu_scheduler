#!/usr/bin/env python3
"""
Neural Network Task Resource Structure - Main Application

This module demonstrates the usage of the multi-resource scheduler for neural network tasks.
"""

from enums import ResourceType, TaskPriority
from task import NNTask
from scheduler import MultiResourceScheduler
from visualization import SchedulerVisualizer

def create_sample_tasks():
    """Create sample tasks for demonstration"""
    tasks = []
    
    # CRITICAL priority task
    task1 = NNTask("T1", "SafetyMonitor", priority=TaskPriority.CRITICAL)
    task1.set_npu_only({2.0: 20, 4.0: 12, 8.0: 8})
    task1.set_performance_requirements(fps=30, latency=30)
    tasks.append(task1)
    
    # HIGH priority tasks
    task2 = NNTask("T2", "ObstacleDetection", priority=TaskPriority.HIGH)
    task2.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 8, 8.0: 5}, 0),
        (ResourceType.NPU, {2.0: 25, 4.0: 15, 8.0: 10}, 8),
    ])
    task2.set_performance_requirements(fps=20, latency=50)
    tasks.append(task2)
    
    task3 = NNTask("T3", "LaneDetection", priority=TaskPriority.HIGH)
    task3.set_npu_only({2.0: 30, 4.0: 18, 8.0: 12})
    task3.set_performance_requirements(fps=15, latency=60)
    task3.add_dependency("T1")  # Depends on safety monitor
    tasks.append(task3)
    
    # NORMAL priority tasks
    task4 = NNTask("T4", "TrafficSignRecog", priority=TaskPriority.NORMAL)
    task4.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 10}, 0),
        (ResourceType.NPU, {2.0: 20, 4.0: 12, 8.0: 8}, 10),
        (ResourceType.DSP, {4.0: 5}, 22),
    ])
    task4.set_performance_requirements(fps=10, latency=80)
    tasks.append(task4)
    
    task5 = NNTask("T5", "PedestrianTracking", priority=TaskPriority.NORMAL)
    task5.set_npu_only({2.0: 35, 4.0: 20, 8.0: 12})
    task5.set_performance_requirements(fps=10, latency=100)
    tasks.append(task5)
    
    # LOW priority tasks
    task6 = NNTask("T6", "SceneUnderstanding", priority=TaskPriority.LOW)
    task6.set_npu_only({2.0: 50, 4.0: 30, 8.0: 20})
    task6.set_performance_requirements(fps=5, latency=200)
    tasks.append(task6)
    
    task7 = NNTask("T7", "MapUpdate", priority=TaskPriority.LOW)
    task7.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 15}, 0),
        (ResourceType.NPU, {2.0: 40, 4.0: 25, 8.0: 15}, 15),
    ])
    task7.set_performance_requirements(fps=2, latency=500)
    tasks.append(task7)
    
    # Additional tasks to demonstrate scaling
    for i in range(8, 12):
        priority = TaskPriority.NORMAL if i % 2 == 0 else TaskPriority.LOW
        task = NNTask(f"T{i}", f"Task_{i}", priority=priority)
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

def main():
    """Main application entry point"""
    print("=== Neural Network Task Resource Structure Demo ===")
    
    # Create scheduler
    scheduler = setup_scheduler()
    
    # Add sample tasks
    tasks = create_sample_tasks()
    for task in tasks:
        scheduler.add_task(task)
    
    print(f"Setup complete:")
    print(f"- NPU Resources: {len(scheduler.resources[ResourceType.NPU])}")
    print(f"- DSP Resources: {len(scheduler.resources[ResourceType.DSP])}")
    print(f"- Total Tasks: {len(scheduler.tasks)}")
    
    # Execute priority-aware scheduling
    print("\nStarting priority-aware scheduling...")
    schedule_results = scheduler.priority_aware_schedule(time_window=500.0)
    
    # Print results
    scheduler.print_schedule_summary()
    
    # Create visualizer and generate plots
    visualizer = SchedulerVisualizer(scheduler)
    
    # Plot task overview with priorities
    print("\nGenerating task overview plot...")
    try:
        visualizer.plot_task_overview(selected_bw=4.0)
    except Exception as e:
        print(f"Could not generate task overview plot: {e}")
    
    # Plot scheduling Gantt chart with priority colors
    print("\nGenerating priority-aware scheduling Gantt chart...")
    try:
        visualizer.plot_pipeline_schedule(time_window=200.0)  # Show first 200ms
    except Exception as e:
        print(f"Could not generate Gantt chart: {e}")
    
    # Print first 15 scheduling events
    print("\nFirst 15 scheduling events:")
    for i, schedule in enumerate(schedule_results[:15]):
        task = scheduler.tasks[schedule.task_id]
        print(f"{i+1:2d}. [{task.priority.name:8s}] {task.name:20s} @ "
              f"{schedule.start_time:5.1f}-{schedule.end_time:5.1f}ms, "
              f"Resources: {list(schedule.assigned_resources.values())}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
