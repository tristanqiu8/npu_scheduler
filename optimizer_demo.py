#!/usr/bin/env python3
"""
Demo of the scheduling optimizer that treats all scheduling aspects as variables
"""

from enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from task import NNTask
from scheduler import MultiResourceScheduler
from scheduling_optimizer import SchedulingOptimizer, SchedulingSearchSpace, SchedulingObjective
from complete_resource_fix import apply_complete_resource_fix, validate_fixed_schedule
from dragon4_segmentation_final_test import apply_simple_segmentation_patch



def create_optimization_scenario():
    """Create a scenario for optimization"""
    # Create scheduler
    scheduler = MultiResourceScheduler(enable_segmentation=True)
    # 应用资源冲突修复
    apply_complete_resource_fix(scheduler)
    
    # 如果启用了分段，应用分段补丁
    if scheduler.enable_segmentation:
        apply_simple_segmentation_patch(scheduler)
    # Add resources - 4 NPUs and 2 DSPs with different capabilities
    scheduler.add_npu("NPU_0", bandwidth=8.0)  # High-performance
    scheduler.add_npu("NPU_1", bandwidth=8.0)  # High-performance
    scheduler.add_npu("NPU_2", bandwidth=4.0)  # Medium-performance
    scheduler.add_npu("NPU_3", bandwidth=2.0)  # Low-performance
    scheduler.add_dsp("DSP_0", bandwidth=4.0)
    scheduler.add_dsp("DSP_1", bandwidth=4.0)
    
    # Create tasks with initial configurations (these will be optimized)
    tasks = []
    
    # Task 1: Vision processing (initially NORMAL priority)
    task1 = NNTask("T1", "VisionProcessing", 
                   priority=TaskPriority.NORMAL,  # Will be optimized
                   runtime_type=RuntimeType.ACPU_RUNTIME,  # Will be optimized
                   segmentation_strategy=SegmentationStrategy.CUSTOM_SEGMENTATION)
    task1.set_npu_only({2.0: 40, 4.0: 25, 8.0: 15}, "vision_seg")
    task1.add_cut_points_to_segment("vision_seg", [
        ("conv1", 0.2, 0.15),
        ("conv2", 0.4, 0.12),
        ("conv3", 0.6, 0.14),
        ("conv4", 0.8, 0.16)
    ])
    # Define multiple possible configurations
    task1.set_preset_cut_configurations("vision_seg", [
        [],                              # Config 0: No cuts
        ["conv2"],                       # Config 1: Single middle cut
        ["conv1", "conv3"],             # Config 2: Two cuts
        ["conv1", "conv2", "conv3"],    # Config 3: Three cuts
        ["conv1", "conv2", "conv3", "conv4"]  # Config 4: All cuts
    ])
    task1.set_performance_requirements(fps=30, latency=35)
    tasks.append(task1)
    
    # Task 2: Sensor fusion (initially LOW priority)
    task2 = NNTask("T2", "SensorFusion",
                   priority=TaskPriority.LOW,  # Will be optimized
                   runtime_type=RuntimeType.ACPU_RUNTIME,  # Will be optimized
                   segmentation_strategy=SegmentationStrategy.CUSTOM_SEGMENTATION)
    task2.set_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 10, 8.0: 7}, 0, "fusion_dsp"),
        (ResourceType.NPU, {2.0: 30, 4.0: 20, 8.0: 12}, 10, "fusion_npu"),
    ])
    task2.add_cut_points_to_segment("fusion_dsp", [
        ("preprocess", 0.5, 0.1)
    ])
    task2.add_cut_points_to_segment("fusion_npu", [
        ("layer1", 0.3, 0.14),
        ("layer2", 0.6, 0.13),
        ("layer3", 0.9, 0.15)
    ])
    task2.set_preset_cut_configurations("fusion_dsp", [
        [],                 # Config 0: No cuts
        ["preprocess"]      # Config 1: Cut
    ])
    task2.set_preset_cut_configurations("fusion_npu", [
        [],                         # Config 0: No cuts
        ["layer2"],                 # Config 1: Middle cut
        ["layer1", "layer3"],       # Config 2: Skip middle
        ["layer1", "layer2", "layer3"]  # Config 3: All cuts
    ])
    task2.set_performance_requirements(fps=20, latency=50)
    tasks.append(task2)
    
    # Task 3: Control loop (initially NORMAL priority)
    task3 = NNTask("T3", "ControlLoop",
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.CUSTOM_SEGMENTATION)
    task3.set_npu_only({2.0: 25, 4.0: 15, 8.0: 10}, "control_seg")
    task3.add_cut_points_to_segment("control_seg", [
        ("predict", 0.4, 0.11),
        ("optimize", 0.7, 0.13)
    ])
    task3.set_preset_cut_configurations("control_seg", [
        [],                      # Config 0: No cuts
        ["predict"],             # Config 1: Early cut
        ["optimize"],            # Config 2: Late cut
        ["predict", "optimize"]  # Config 3: Both cuts
    ])
    task3.set_performance_requirements(fps=50, latency=20)  # Tight requirements
    tasks.append(task3)
    
    # Task 4: Background analytics (initially HIGH priority - suboptimal)
    task4 = NNTask("T4", "Analytics",
                   priority=TaskPriority.HIGH,  # Probably too high
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.CUSTOM_SEGMENTATION)
    task4.set_npu_only({2.0: 60, 4.0: 40, 8.0: 25}, "analytics_seg")
    task4.add_cut_points_to_segment("analytics_seg", [
        ("stage1", 0.25, 0.18),
        ("stage2", 0.5, 0.16),
        ("stage3", 0.75, 0.17)
    ])
    task4.set_preset_cut_configurations("analytics_seg", [
        [],                           # Config 0: No cuts
        ["stage2"],                   # Config 1: Middle only
        ["stage1", "stage3"],         # Config 2: Edges only
        ["stage1", "stage2", "stage3"]  # Config 3: All cuts
    ])
    task4.set_performance_requirements(fps=5, latency=200)  # Relaxed requirements
    tasks.append(task4)
    
    # Add tasks to scheduler
    for task in tasks:
        scheduler.add_task(task)
    
    return scheduler, tasks


def demonstrate_full_optimization():
    """Demonstrate optimization of all variables"""
    print("=== Scheduling Optimization Demo ===")
    print("Optimizing: Priority, Runtime Type, Segmentation Config, and Core Assignment")
    print()
    
    # Create scenario
    scheduler, tasks = create_optimization_scenario()
    
    # Create optimizer
    optimizer = SchedulingOptimizer(scheduler)
    
    # Define search spaces for each task
    print("Defining search spaces for optimization...")
    
    # Task 1: Vision - could be any priority, but DSP_Runtime might not make sense
    optimizer.define_search_space("T1", SchedulingSearchSpace(
        task_id="T1",
        allowed_priorities=[TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.NORMAL],
        allowed_runtime_types=[RuntimeType.ACPU_RUNTIME],  # NPU-only task
        segmentation_options={"vision_seg": [0, 1, 2, 3, 4]},  # All configs
        available_cores={ResourceType.NPU: ["NPU_0", "NPU_1", "NPU_2", "NPU_3"]}
    ))
    
    # Task 2: Sensor Fusion - flexible task
    optimizer.define_search_space("T2", SchedulingSearchSpace(
        task_id="T2",
        allowed_priorities=list(TaskPriority),  # Any priority
        allowed_runtime_types=list(RuntimeType),  # Any runtime
        segmentation_options={
            "fusion_dsp": [0, 1],
            "fusion_npu": [0, 1, 2, 3]
        },
        available_cores={
            ResourceType.DSP: ["DSP_0", "DSP_1"],
            ResourceType.NPU: ["NPU_0", "NPU_1", "NPU_2", "NPU_3"]
        }
    ))
    
    # Task 3: Control Loop - likely needs high priority
    optimizer.define_search_space("T3", SchedulingSearchSpace(
        task_id="T3",
        allowed_priorities=[TaskPriority.CRITICAL, TaskPriority.HIGH],  # Needs high priority
        allowed_runtime_types=list(RuntimeType),  # Any runtime
        segmentation_options={"control_seg": [0, 1, 2, 3]},  # All configs
        available_cores={ResourceType.NPU: ["NPU_0", "NPU_1", "NPU_2", "NPU_3"]}
    ))
    
    # Task 4: Analytics - likely low priority
    optimizer.define_search_space("T4", SchedulingSearchSpace(
        task_id="T4",
        allowed_priorities=[TaskPriority.NORMAL, TaskPriority.LOW],  # Should be lower priority
        allowed_runtime_types=[RuntimeType.ACPU_RUNTIME],  # Can pipeline
        segmentation_options={"analytics_seg": [0, 1, 2, 3]},  # All configs
        available_cores={ResourceType.NPU: ["NPU_0", "NPU_1", "NPU_2", "NPU_3"]}
    ))
    
    # Set optimization objectives
    optimizer.objective = SchedulingObjective(
        latency_weight=1.0,
        throughput_weight=1.5,  # Emphasize meeting FPS requirements
        utilization_weight=0.8,
        priority_violation_weight=3.0,  # Heavy penalty for not meeting requirements
        overhead_weight=0.5
    )
    
    print("\nInitial Configuration:")
    print("-" * 80)
    for task in tasks:
        print(f"{task.task_id}: Priority={task.priority.name}, Runtime={task.runtime_type.value}")
        print(f"     Segmentation: {task.get_segmentation_summary()}")
    
    # Run initial scheduling to see baseline
    print("\n--- Baseline Performance ---")
    baseline_results = scheduler.priority_aware_schedule_with_segmentation(time_window=500.0)
    if baseline_results:
        print_performance_summary(scheduler, baseline_results)
    
    # Run greedy optimization
    print("\n--- Running Greedy Optimization ---")
    greedy_solution = optimizer.optimize_greedy(time_window=500.0, iterations=5)
    
    # Apply and evaluate greedy solution
    print("\n--- Greedy Solution Performance ---")
    apply_solution_to_scheduler(scheduler, greedy_solution)
    greedy_results = scheduler.priority_aware_schedule_with_segmentation(time_window=500.0)
    if greedy_results:
        print_performance_summary(scheduler, greedy_results)
    
    # Print optimized configuration
    optimizer.print_solution(greedy_solution)
    
    # Reset and try genetic algorithm
    print("\n\n--- Running Genetic Algorithm Optimization ---")
    reset_tasks(scheduler, tasks)
    genetic_solution = optimizer.optimize_genetic(
        population_size=30, 
        generations=10, 
        time_window=500.0
    )
    
    # Apply and evaluate genetic solution
    print("\n--- Genetic Algorithm Solution Performance ---")
    apply_solution_to_scheduler(scheduler, genetic_solution)
    genetic_results = scheduler.priority_aware_schedule_with_segmentation(time_window=500.0)
    if genetic_results:
        print_performance_summary(scheduler, genetic_results)
    
    # Print optimized configuration
    optimizer.print_solution(genetic_solution)
    
    # Compare solutions
    print("\n\n=== Solution Comparison ===")
    compare_solutions(scheduler, tasks, greedy_solution, genetic_solution)


def apply_solution_to_scheduler(scheduler, solution):
    """Apply optimization solution to scheduler tasks"""
    for task_id, decision in solution.items():
        task = scheduler.tasks[task_id]
        
        # Apply priority
        task.priority = decision.priority
        
        # Apply runtime type
        task.runtime_type = decision.runtime_type
        
        # Apply segmentation configurations
        for seg_id, config_idx in decision.segmentation_configs.items():
            if seg_id in task.preset_cut_configurations:
                task.select_cut_configuration(seg_id, config_idx)
        
        # Note: Core assignments would need scheduler modification to fully implement


def reset_tasks(scheduler, original_tasks):
    """Reset tasks to original configuration"""
    for i, task in enumerate(original_tasks):
        scheduler_task = scheduler.tasks[task.task_id]
        scheduler_task.priority = task.priority
        scheduler_task.runtime_type = task.runtime_type
        scheduler_task.current_segmentation = {}
        scheduler_task.selected_cut_config_index = {}


def print_performance_summary(scheduler, results):
    """Print performance summary of scheduling results"""
    if not results:
        print("No scheduling results to analyze")
        return
    
    # Task execution counts
    task_counts = {}
    task_latencies = {}
    
    for schedule in results:
        task_id = schedule.task_id
        if task_id not in task_counts:
            task_counts[task_id] = 0
            task_latencies[task_id] = []
        
        task_counts[task_id] += 1
        task_latencies[task_id].append(schedule.actual_latency)
    
    # Calculate metrics
    total_time = results[-1].end_time
    
    print(f"Total completion time: {total_time:.1f}ms")
    print(f"\nTask Performance:")
    print(f"{'Task':<8} {'Count':<8} {'FPS':<12} {'Req FPS':<12} {'Avg Latency':<12} {'Req Latency':<12} {'Status'}")
    print("-" * 90)
    
    total_violations = 0
    for task_id, task in scheduler.tasks.items():
        count = task_counts.get(task_id, 0)
        achieved_fps = count / (total_time / 1000.0) if total_time > 0 else 0
        avg_latency = sum(task_latencies.get(task_id, [0])) / len(task_latencies.get(task_id, [1]))
        
        fps_ok = achieved_fps >= task.fps_requirement * 0.95  # 5% tolerance
        latency_ok = avg_latency <= task.latency_requirement * 1.05  # 5% tolerance
        
        status = "OK" if fps_ok and latency_ok else "VIOLATION"
        if not fps_ok or not latency_ok:
            total_violations += 1
        
        print(f"{task_id:<8} {count:<8} {achieved_fps:<12.1f} {task.fps_requirement:<12.1f} "
              f"{avg_latency:<12.1f} {task.latency_requirement:<12.1f} {status}")
    
    # Resource utilization
    utilization = scheduler.get_resource_utilization(total_time)
    avg_util = sum(utilization.values()) / len(utilization) if utilization else 0
    
    print(f"\nResource Utilization: {avg_util:.1f}%")
    print(f"Total violations: {total_violations}")


def compare_solutions(scheduler, tasks, greedy_solution, genetic_solution):
    """Compare two optimization solutions"""
    print(f"{'Task':<8} {'Original Priority':<18} {'Greedy Priority':<18} {'Genetic Priority':<18}")
    print("-" * 70)
    
    for task in tasks:
        orig_priority = task.priority.name
        greedy_priority = greedy_solution[task.task_id].priority.name
        genetic_priority = genetic_solution[task.task_id].priority.name
        
        print(f"{task.task_id:<8} {orig_priority:<18} {greedy_priority:<18} {genetic_priority:<18}")
    
    print(f"\n{'Task':<8} {'Original Runtime':<18} {'Greedy Runtime':<18} {'Genetic Runtime':<18}")
    print("-" * 70)
    
    for task in tasks:
        orig_runtime = task.runtime_type.value
        greedy_runtime = greedy_solution[task.task_id].runtime_type.value
        genetic_runtime = genetic_solution[task.task_id].runtime_type.value
        
        print(f"{task.task_id:<8} {orig_runtime:<18} {greedy_runtime:<18} {genetic_runtime:<18}")
    
    print(f"\n{'Task':<8} {'Segment':<15} {'Original Cfg':<15} {'Greedy Cfg':<15} {'Genetic Cfg':<15}")
    print("-" * 75)
    
    for task in tasks:
        for seg_id in task.preset_cut_configurations.keys():
            orig_cfg = task.selected_cut_config_index.get(seg_id, 0)
            greedy_cfg = greedy_solution[task.task_id].segmentation_configs.get(seg_id, 0)
            genetic_cfg = genetic_solution[task.task_id].segmentation_configs.get(seg_id, 0)
            
            print(f"{task.task_id:<8} {seg_id:<15} {orig_cfg:<15} {greedy_cfg:<15} {genetic_cfg:<15}")


def demonstrate_constraint_based_optimization():
    """Demonstrate optimization with specific constraints"""
    print("\n\n=== Constraint-Based Optimization Demo ===")
    print("Adding constraints to the optimization problem")
    
    # Create scenario
    scheduler, tasks = create_optimization_scenario()
    optimizer = SchedulingOptimizer(scheduler)
    
    # Define constrained search spaces
    print("\nApplying constraints:")
    print("- T1 (Vision): Must be HIGH or CRITICAL priority")
    print("- T2 (Sensor Fusion): Must use DSP_Runtime if HIGH priority")
    print("- T3 (Control): Must be CRITICAL with minimal segmentation")
    print("- T4 (Analytics): Must be LOW priority with maximal segmentation")
    
    # Constrained search spaces
    optimizer.define_search_space("T1", SchedulingSearchSpace(
        task_id="T1",
        allowed_priorities=[TaskPriority.CRITICAL, TaskPriority.HIGH],  # Constrained
        allowed_runtime_types=[RuntimeType.ACPU_RUNTIME],
        segmentation_options={"vision_seg": [0, 1, 2]},  # Limited configs
        available_cores={ResourceType.NPU: ["NPU_0", "NPU_1"]}  # Only high-perf NPUs
    ))
    
    optimizer.define_search_space("T2", SchedulingSearchSpace(
        task_id="T2",
        allowed_priorities=[TaskPriority.HIGH, TaskPriority.NORMAL],
        allowed_runtime_types=[RuntimeType.DSP_RUNTIME],  # Forced DSP_Runtime
        segmentation_options={
            "fusion_dsp": [0, 1],
            "fusion_npu": [0, 1]  # Limited segmentation
        },
        available_cores={
            ResourceType.DSP: ["DSP_0", "DSP_1"],
            ResourceType.NPU: ["NPU_0", "NPU_1", "NPU_2", "NPU_3"]
        }
    ))
    
    optimizer.define_search_space("T3", SchedulingSearchSpace(
        task_id="T3",
        allowed_priorities=[TaskPriority.CRITICAL],  # Fixed priority
        allowed_runtime_types=list(RuntimeType),
        segmentation_options={"control_seg": [0, 1]},  # Minimal segmentation
        available_cores={ResourceType.NPU: ["NPU_0", "NPU_1"]}  # High-perf only
    ))
    
    optimizer.define_search_space("T4", SchedulingSearchSpace(
        task_id="T4",
        allowed_priorities=[TaskPriority.LOW],  # Fixed priority
        allowed_runtime_types=[RuntimeType.ACPU_RUNTIME],
        segmentation_options={"analytics_seg": [2, 3]},  # Max segmentation
        available_cores={ResourceType.NPU: ["NPU_2", "NPU_3"]}  # Low-perf NPUs
    ))
    
    # Run constrained optimization
    print("\n--- Running Constrained Optimization ---")
    constrained_solution = optimizer.optimize_greedy(time_window=500.0, iterations=3)
    
    # Show results
    optimizer.print_solution(constrained_solution)
    
    # Apply and evaluate
    apply_solution_to_scheduler(scheduler, constrained_solution)
    results = scheduler.priority_aware_schedule_with_segmentation(time_window=500.0)
    if results:
        print("\n--- Constrained Solution Performance ---")
        print_performance_summary(scheduler, results)


if __name__ == "__main__":
    # Run main optimization demo
    demonstrate_full_optimization()
    
    # Run constrained optimization demo
    # demonstrate_constraint_based_optimization()
    
    print("\n\n=== Key Insights ===")
    print("1. Priority assignment significantly affects scheduling order and resource allocation")
    print("2. Runtime type (DSP vs ACPU) determines resource binding behavior")
    print("3. Segmentation configuration trades off between parallelism and overhead")
    print("4. Core assignment can balance load across resources of different capabilities")
    print("5. Joint optimization of all variables can find non-obvious optimal configurations")