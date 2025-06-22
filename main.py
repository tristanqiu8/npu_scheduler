#!/usr/bin/env python3
"""
Enhanced Neural Network Task Resource Structure - Network Segmentation Demo

This module demonstrates the usage of the multi-resource scheduler with network segmentation capabilities.
Features:
- Adaptive network segmentation for improved resource utilization
- Cut point management with configurable overhead
- Multiple segmentation strategies
- Enhanced scheduling with sub-segment granularity
"""

from enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from task import NNTask
from scheduler import MultiResourceScheduler
from visualization import SchedulerVisualizer

def create_sample_tasks_with_segmentation():
    """Create sample tasks with network segmentation support for demonstration"""
    tasks = []
    
    # CRITICAL priority task - DSP_Runtime with aggressive segmentation
    task1 = NNTask("T1", "SafetyMonitor", priority=TaskPriority.CRITICAL, 
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION)
    task1.set_npu_only({2.0: 20, 4.0: 12, 8.0: 8}, "safety_npu_seg")
    # Add cut points for fine-grained control
    task1.add_cut_points_to_segment("safety_npu_seg", [
        ("op1", 0.2, 0.15),    # Early cut point
        ("op10", 0.6, 0.12),   # Middle cut point  
        ("op23", 0.85, 0.18)   # Late cut point
    ])
    task1.set_performance_requirements(fps=30, latency=30)
    tasks.append(task1)
    
    # HIGH priority task - DSP_Runtime with multi-segment cutting
    task2 = NNTask("T2", "ObstacleDetection", priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION)
    task2.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 8, 8.0: 5}, 0, "obstacle_dsp_seg"),
        (ResourceType.NPU, {2.0: 25, 4.0: 15, 8.0: 10}, 8, "obstacle_npu_seg"),
    ])
    # Add cut points to both segments
    task2.add_cut_points_to_segment("obstacle_dsp_seg", [
        ("op2", 0.4, 0.10),
        ("op7", 0.8, 0.14)
    ])
    task2.add_cut_points_to_segment("obstacle_npu_seg", [
        ("op12", 0.3, 0.16),
        ("op18", 0.7, 0.13)
    ])
    task2.set_performance_requirements(fps=20, latency=50)
    tasks.append(task2)
    
    # HIGH priority task - ACPU_Runtime with conservative segmentation  
    task3 = NNTask("T3", "LaneDetection", priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION)
    task3.set_npu_only({2.0: 30, 4.0: 18, 8.0: 12}, "lane_npu_seg")
    task3.add_cut_points_to_segment("lane_npu_seg", [
        ("op5", 0.25, 0.11),
        ("op15", 0.5, 0.15),
        ("op25", 0.75, 0.17)
    ])
    task3.set_performance_requirements(fps=15, latency=60)
    task3.add_dependency("T1")  # Depends on safety monitor
    tasks.append(task3)
    
    # NORMAL priority task - Forced segmentation strategy
    task4 = NNTask("T4", "TrafficSignRecog", priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION)
    task4.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 10}, 0, "sign_dsp_seg"),
        (ResourceType.NPU, {2.0: 20, 4.0: 12, 8.0: 8}, 10, "sign_npu_seg"),
        (ResourceType.DSP, {4.0: 5}, 22, "sign_dsp_post_seg"),
    ])
    # Add extensive cut points for forced segmentation
    task4.add_cut_points_to_segment("sign_dsp_seg", [("op3", 0.5, 0.12)])
    task4.add_cut_points_to_segment("sign_npu_seg", [
        ("op8", 0.2, 0.14),
        ("op16", 0.6, 0.13),
        ("op24", 0.9, 0.16)
    ])
    task4.add_cut_points_to_segment("sign_dsp_post_seg", [("op30", 0.4, 0.10)])
    task4.set_performance_requirements(fps=10, latency=80)
    tasks.append(task4)
    
    # NORMAL priority task - No segmentation strategy
    task5 = NNTask("T5", "PedestrianTracking", priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task5.set_npu_only({2.0: 35, 4.0: 20, 8.0: 12}, "pedestrian_npu_seg")
    # Add cut points but they won't be used due to NO_SEGMENTATION strategy
    task5.add_cut_points_to_segment("pedestrian_npu_seg", [
        ("op6", 0.3, 0.15),
        ("op20", 0.7, 0.18)
    ])
    task5.set_performance_requirements(fps=10, latency=100)
    tasks.append(task5)
    
    # LOW priority task - Adaptive segmentation with many cut points
    task6 = NNTask("T6", "SceneUnderstanding", priority=TaskPriority.LOW,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION)
    task6.set_npu_only({2.0: 50, 4.0: 30, 8.0: 20}, "scene_npu_seg")
    task6.add_cut_points_to_segment("scene_npu_seg", [
        ("op4", 0.15, 0.12),
        ("op11", 0.35, 0.14),
        ("op19", 0.55, 0.16),
        ("op26", 0.75, 0.13),
        ("op32", 0.90, 0.19)
    ])
    task6.set_performance_requirements(fps=5, latency=200)
    tasks.append(task6)
    
    # LOW priority task - Custom segmentation (will use current settings)
    task7 = NNTask("T7", "MapUpdate", priority=TaskPriority.LOW,
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.CUSTOM_SEGMENTATION)
    task7.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 15}, 0, "map_dsp_seg"),
        (ResourceType.NPU, {2.0: 40, 4.0: 25, 8.0: 15}, 15, "map_npu_seg"),
    ])
    task7.add_cut_points_to_segment("map_dsp_seg", [("op9", 0.6, 0.11)])
    task7.add_cut_points_to_segment("map_npu_seg", [
        ("op14", 0.4, 0.15),
        ("op22", 0.8, 0.17)
    ])
    # Pre-configure custom segmentation
    task7.current_segmentation = {
        "map_dsp_seg": ["op9"],          # Use one cut in DSP segment
        "map_npu_seg": []                # No cuts in NPU segment
    }
    task7.set_performance_requirements(fps=2, latency=500)
    tasks.append(task7)
    
    # Additional tasks to show segmentation scaling effects
    for i in range(8, 12):
        priority = TaskPriority.NORMAL if i % 2 == 0 else TaskPriority.LOW
        runtime_type = RuntimeType.DSP_RUNTIME if i % 3 == 0 else RuntimeType.ACPU_RUNTIME
        strategy = SegmentationStrategy.ADAPTIVE_SEGMENTATION if i % 2 == 0 else SegmentationStrategy.NO_SEGMENTATION
        
        task = NNTask(f"T{i}", f"Task_{i}", priority=priority, runtime_type=runtime_type,
                       segmentation_strategy=strategy)
        
        if runtime_type == RuntimeType.DSP_RUNTIME and i % 4 == 0:
            # Some DSP_Runtime tasks with mixed resources and segmentation
            task.set_dsp_npu_sequence([
                (ResourceType.DSP, {4.0: 8}, 0, f"task{i}_dsp_seg"),
                (ResourceType.NPU, {2.0: 20+i*2, 4.0: 15+i, 8.0: 10+i//2}, 8, f"task{i}_npu_seg"),
            ])
            # Add random cut points
            task.add_cut_points_to_segment(f"task{i}_dsp_seg", [(f"op{i}_1", 0.5, 0.12)])
            task.add_cut_points_to_segment(f"task{i}_npu_seg", [
                (f"op{i}_2", 0.3, 0.14),
                (f"op{i}_3", 0.7, 0.16)
            ])
        else:
            # NPU-only tasks with varying cut point complexity
            task.set_npu_only({2.0: 20+i*2, 4.0: 15+i, 8.0: 10+i//2}, f"task{i}_npu_seg")
            if strategy != SegmentationStrategy.NO_SEGMENTATION:
                num_cuts = (i % 3) + 1  # 1-3 cut points
                for j in range(num_cuts):
                    position = (j + 1) / (num_cuts + 1)  # Evenly distribute cuts
                    overhead = 0.12 + (j * 0.02)  # Slightly increasing overhead
                    task.add_cut_points_to_segment(f"task{i}_npu_seg", [(f"op{i}_{j}", position, overhead)])
            
        task.set_performance_requirements(fps=8, latency=150)
        tasks.append(task)
    
    return tasks

def setup_scheduler_with_segmentation(enable_segmentation: bool = True, max_overhead_ratio: float = 0.15):
    """Setup scheduler with segmentation configuration"""
    scheduler = MultiResourceScheduler(
        enable_segmentation=enable_segmentation,
        max_segmentation_overhead_ratio=max_overhead_ratio
    )
    
    # Add multiple NPU resources (different bandwidths)
    scheduler.add_npu("NPU_0", bandwidth=8.0)  # High performance NPU
    scheduler.add_npu("NPU_1", bandwidth=4.0)  # Medium performance NPU
    scheduler.add_npu("NPU_2", bandwidth=2.0)  # Low performance NPU
    
    # Add DSP resources
    scheduler.add_dsp("DSP_0", bandwidth=4.0)
    scheduler.add_dsp("DSP_1", bandwidth=4.0)
    
    return scheduler

def analyze_segmentation_impact(scheduler: MultiResourceScheduler, schedule_results):
    """Analyze the impact of network segmentation on scheduling performance"""
    print("\n=== Network Segmentation Impact Analysis ===")
    
    segmented_tasks = [t for t in scheduler.tasks.values() if t.is_segmented]
    non_segmented_tasks = [t for t in scheduler.tasks.values() if not t.is_segmented]
    
    print(f"Segmented tasks: {len(segmented_tasks)}")
    print(f"Non-segmented tasks: {len(non_segmented_tasks)}")
    
    if segmented_tasks:
        print("\nSegmentation Statistics by Task:")
        for task in segmented_tasks:
            cuts_info = []
            for segment_id, cuts in task.current_segmentation.items():
                if cuts:
                    cuts_info.append(f"{segment_id}: {len(cuts)} cuts")
            
            cuts_summary = ", ".join(cuts_info) if cuts_info else "no active cuts"
            print(f"  {task.task_id} ({task.name}): {cuts_summary}")
            print(f"    Total overhead: {task.total_segmentation_overhead:.2f}ms")
            print(f"    Strategy: {task.segmentation_strategy.value}")
    
    # Compare performance metrics
    segmented_schedules = [s for s in schedule_results if scheduler.tasks[s.task_id].is_segmented]
    non_segmented_schedules = [s for s in schedule_results if not scheduler.tasks[s.task_id].is_segmented]
    
    if segmented_schedules and non_segmented_schedules:
        avg_segmented_latency = sum(s.actual_latency for s in segmented_schedules) / len(segmented_schedules)
        avg_non_segmented_latency = sum(s.actual_latency for s in non_segmented_schedules) / len(non_segmented_schedules)
        
        print(f"\nLatency Comparison:")
        print(f"  Segmented tasks average latency: {avg_segmented_latency:.2f}ms")
        print(f"  Non-segmented tasks average latency: {avg_non_segmented_latency:.2f}ms")
        print(f"  Difference: {avg_segmented_latency - avg_non_segmented_latency:.2f}ms")
    
    # Analyze segmentation strategies effectiveness
    strategy_stats = {}
    for task in scheduler.tasks.values():
        strategy = task.segmentation_strategy.value
        if strategy not in strategy_stats:
            strategy_stats[strategy] = {'count': 0, 'overhead': 0.0, 'segmented': 0}
        
        strategy_stats[strategy]['count'] += 1
        strategy_stats[strategy]['overhead'] += task.total_segmentation_overhead
        if task.is_segmented:
            strategy_stats[strategy]['segmented'] += 1
    
    print("\nSegmentation Strategy Effectiveness:")
    for strategy, stats in strategy_stats.items():
        segmentation_rate = (stats['segmented'] / stats['count']) * 100 if stats['count'] > 0 else 0
        avg_overhead = stats['overhead'] / stats['count'] if stats['count'] > 0 else 0
        print(f"  {strategy}:")
        print(f"    Tasks: {stats['count']}, Segmented: {stats['segmented']} ({segmentation_rate:.1f}%)")
        print(f"    Average overhead: {avg_overhead:.2f}ms")

def demonstrate_segmentation_comparison():
    """Demonstrate scheduling with and without segmentation for comparison"""
    print("\n=== Segmentation Comparison Demo ===")
    
    # Create tasks
    tasks = create_sample_tasks_with_segmentation()
    
    # Test with segmentation enabled
    print("\n--- Testing WITH Segmentation ---")
    scheduler_with_seg = setup_scheduler_with_segmentation(enable_segmentation=True)
    for task in tasks:
        scheduler_with_seg.add_task(task)
    
    results_with_seg = scheduler_with_seg.priority_aware_schedule_with_segmentation(time_window=300.0)
    print(f"Scheduled events: {len(results_with_seg)}")
    if results_with_seg:
        total_time_with_seg = results_with_seg[-1].end_time
        print(f"Total completion time: {total_time_with_seg:.1f}ms")
    
    # Reset tasks and test without segmentation
    print("\n--- Testing WITHOUT Segmentation ---")
    tasks_no_seg = create_sample_tasks_with_segmentation()
    # Force all tasks to use NO_SEGMENTATION strategy
    for task in tasks_no_seg:
        task.set_segmentation_strategy(SegmentationStrategy.NO_SEGMENTATION)
    
    scheduler_no_seg = setup_scheduler_with_segmentation(enable_segmentation=False)
    for task in tasks_no_seg:
        scheduler_no_seg.add_task(task)
    
    results_no_seg = scheduler_no_seg.priority_aware_schedule_with_segmentation(time_window=300.0)
    print(f"Scheduled events: {len(results_no_seg)}")
    if results_no_seg:
        total_time_no_seg = results_no_seg[-1].end_time
        print(f"Total completion time: {total_time_no_seg:.1f}ms")
    
    # Compare results
    if results_with_seg and results_no_seg:
        time_improvement = total_time_no_seg - total_time_with_seg
        improvement_percentage = (time_improvement / total_time_no_seg) * 100
        print(f"\nComparison Results:")
        print(f"  Time improvement with segmentation: {time_improvement:.1f}ms ({improvement_percentage:.1f}%)")
        
        total_overhead = sum(s.segmentation_overhead for s in results_with_seg)
        print(f"  Total segmentation overhead: {total_overhead:.1f}ms")
        print(f"  Net benefit: {time_improvement - total_overhead:.1f}ms")

def main():
    """Enhanced main application entry point with network segmentation"""
    print("=== Enhanced Neural Network Task Scheduler with Network Segmentation ===")
    print("New Features:")
    print("- 🔗 Network segmentation with configurable cut points")
    print("- ⚡ Adaptive segmentation strategies")
    print("- 📊 Overhead-aware scheduling decisions")
    print("- 🎯 Sub-segment granular resource allocation")
    print("- 📈 Segmentation impact analysis")
    
    # Create scheduler with segmentation enabled
    scheduler = setup_scheduler_with_segmentation(enable_segmentation=True, max_overhead_ratio=0.12)
    
    # Add sample tasks with segmentation support
    tasks = create_sample_tasks_with_segmentation()
    for task in tasks:
        scheduler.add_task(task)
    
    print(f"\nSetup complete:")
    print(f"- NPU Resources: {len(scheduler.resources[ResourceType.NPU])}")
    print(f"- DSP Resources: {len(scheduler.resources[ResourceType.DSP])}")
    print(f"- Total Tasks: {len(scheduler.tasks)}")
    print(f"- Segmentation enabled: {scheduler.enable_segmentation}")
    print(f"- Max overhead ratio: {scheduler.max_segmentation_overhead_ratio:.1%}")
    
    # Show task configuration with cut points
    print("\nTask Configuration with Cut Points:")
    for task in scheduler.tasks.values():
        available_cuts = task.get_all_available_cuts()
        total_cuts = sum(len(cuts) for cuts in available_cuts.values())
        print(f"  {task.task_id} ({task.name}) - {task.segmentation_strategy.value}")
        print(f"    Total cut points: {total_cuts}")
        for segment_id, cuts in available_cuts.items():
            if cuts:
                print(f"      {segment_id}: {cuts}")
    
    # Execute enhanced scheduling with segmentation
    print("\nStarting enhanced scheduling with network segmentation...")
    schedule_results = scheduler.priority_aware_schedule_with_segmentation(time_window=400.0)
    
    # Print enhanced results
    scheduler.print_schedule_summary()
    
    # Create enhanced visualizer
    visualizer = SchedulerVisualizer(scheduler)
    
    # Plot enhanced task overview with segmentation information
    print("\nGenerating enhanced task overview with segmentation...")
    try:
        visualizer.plot_task_overview_with_segmentation(selected_bw=4.0)
    except Exception as e:
        print(f"Could not generate task overview plot: {e}")
    
    # Plot enhanced scheduling Gantt chart with sub-segments
    print("\nGenerating enhanced Gantt chart with sub-segment visualization...")
    try:
        visualizer.plot_pipeline_schedule_with_segmentation(time_window=250.0)
    except Exception as e:
        print(f"Could not generate Gantt chart: {e}")
    
    # Print detailed scheduling events with segmentation info
    print("\nFirst 25 scheduling events with segmentation details:")
    for i, schedule in enumerate(schedule_results[:25]):
        task = scheduler.tasks[schedule.task_id]
        runtime_symbol = 'B' if task.runtime_type == RuntimeType.DSP_RUNTIME else 'P'
        seg_symbol = 'S' if task.is_segmented else 'N'  # S=Segmented, N=Normal
        
        cuts_info = ""
        if schedule.used_cuts:
            total_cuts = sum(len(cuts) for cuts in schedule.used_cuts.values())
            cuts_info = f", {total_cuts} cuts"
        
        overhead_info = f", +{schedule.segmentation_overhead:.1f}ms" if schedule.segmentation_overhead > 0 else ""
        
        print(f"{i+1:2d}. [{task.priority.name:8s}] {task.name:20s} ({runtime_symbol}{seg_symbol}) @ "
              f"{schedule.start_time:5.1f}-{schedule.end_time:5.1f}ms{cuts_info}{overhead_info}")
        
        # Show sub-segment details for highly segmented tasks
        if len(schedule.sub_segment_schedule) > 2:
            print(f"     Sub-segments: {len(schedule.sub_segment_schedule)} parts")
    
    # Analyze segmentation impact
    analyze_segmentation_impact(scheduler, schedule_results)
    
    # Run comparison demonstration
    demonstrate_segmentation_comparison()
    
    print("\nEnhanced demo with network segmentation completed successfully!")
    print("\nKey enhancements:")
    print("- 'S' indicates segmented tasks, 'N' indicates non-segmented")
    print("- Sub-segment scheduling enables finer resource utilization")
    print("- Adaptive strategies balance overhead vs. parallelism benefits")
    print("- Cut points are automatically selected based on resource availability")
    print("- Overhead tracking ensures segmentation remains beneficial")

if __name__ == "__main__":
    main()