#!/usr/bin/env python3
"""
Dragon4系统优化测试
测试分段选择、优先级调整和运行时类型优化
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict, Optional, Tuple
from dragon4_system import Dragon4System, Dragon4Config
from dragon4_workload import Dragon4Workload, WorkloadConfig
from scheduling_optimizer import SchedulingOptimizer, SchedulingSearchSpace, SchedulingObjective
from scheduling_optimizer_fix import apply_scheduling_optimizer_fix
from validator_precision_fix import apply_validator_precision_fix
from schedule_validator import validate_schedule
from elegant_visualization import ElegantSchedulerVisualizer
from enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy


def run_baseline_test(system: Dragon4System, tasks: List, test_name: str = "Baseline") -> Tuple[List, Dict]:
    """运行基准测试（不分段）"""
    print(f"\n{'='*60}")
    print(f"{test_name} Test")
    print(f"{'='*60}")
    
    # 添加任务到系统
    system.reset()
    for task in tasks:
        system.scheduler.add_task(task)
    
    # 执行调度
    time_window = 500.0
    results = system.schedule(time_window)
    
    if results:
        print(f"✅ 调度成功: {len(results)} 个事件")
        
        # 验证调度
        is_valid, conflicts = validate_schedule(system.scheduler)
        if is_valid:
            print("✅ 调度验证通过，无资源冲突")
        else:
            print(f"❌ 发现 {len(conflicts)} 个资源冲突")
            for i, conflict in enumerate(conflicts[:3]):
                print(f"  {i+1}. {conflict}")
        
        # 计算性能指标
        metrics = calculate_performance_metrics(system.scheduler, results, time_window)
        print_performance_summary(metrics)
        
        return results, metrics
    else:
        print("❌ 调度失败")
        return None, {}


def calculate_performance_metrics(scheduler, results, time_window) -> Dict:
    """计算性能指标"""
    metrics = {
        'total_events': len(results),
        'time_window': time_window,
        'task_metrics': {},
        'resource_utilization': {},
        'total_violations': 0,
        'avg_latency': 0,
        'total_segmentation_overhead': 0
    }
    
    # 任务执行统计
    task_counts = {}
    task_latencies = {}
    
    for schedule in results:
        task_id = schedule.task_id
        if task_id not in task_counts:
            task_counts[task_id] = 0
            task_latencies[task_id] = []
        
        task_counts[task_id] += 1
        task_latencies[task_id].append(schedule.actual_latency)
    
    # 计算每个任务的指标
    total_latency = 0
    for task_id, task in scheduler.tasks.items():
        count = task_counts.get(task_id, 0)
        achieved_fps = count / (time_window / 1000.0) if time_window > 0 else 0
        latencies = task_latencies.get(task_id, [0])
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        fps_ok = achieved_fps >= task.fps_requirement * 0.95
        latency_ok = avg_latency <= task.latency_requirement * 1.05
        
        metrics['task_metrics'][task_id] = {
            'count': count,
            'achieved_fps': achieved_fps,
            'required_fps': task.fps_requirement,
            'avg_latency': avg_latency,
            'required_latency': task.latency_requirement,
            'fps_ok': fps_ok,
            'latency_ok': latency_ok,
            'violation': not (fps_ok and latency_ok)
        }
        
        if not (fps_ok and latency_ok):
            metrics['total_violations'] += 1
        
        total_latency += avg_latency
        metrics['total_segmentation_overhead'] += task.total_segmentation_overhead
    
    metrics['avg_latency'] = total_latency / len(scheduler.tasks) if scheduler.tasks else 0
    
    # 资源利用率
    metrics['resource_utilization'] = scheduler.get_resource_utilization(time_window)
    metrics['avg_utilization'] = sum(metrics['resource_utilization'].values()) / len(metrics['resource_utilization']) if metrics['resource_utilization'] else 0
    
    return metrics


def print_performance_summary(metrics: Dict):
    """打印性能摘要"""
    print("\n任务性能指标:")
    print(f"{'任务':<8} {'执行':<6} {'FPS':<12} {'延迟(ms)':<12} {'状态':<10}")
    print("-" * 50)
    
    for task_id, task_metrics in sorted(metrics['task_metrics'].items()):
        fps_str = f"{task_metrics['achieved_fps']:.1f}/{task_metrics['required_fps']:.0f}"
        latency_str = f"{task_metrics['avg_latency']:.1f}/{task_metrics['required_latency']:.0f}"
        status = "❌违反" if task_metrics['violation'] else "✅正常"
        
        print(f"{task_id:<8} {task_metrics['count']:<6} {fps_str:<12} {latency_str:<12} {status:<10}")
    
    print(f"\n性能摘要:")
    print(f"  总违反数: {metrics['total_violations']}")
    print(f"  平均延迟: {metrics['avg_latency']:.1f}ms")
    print(f"  平均资源利用率: {metrics['avg_utilization']:.1f}%")
    print(f"  分段开销: {metrics['total_segmentation_overhead']:.2f}ms")


def run_optimization_test(system: Dragon4System, base_tasks: List) -> Tuple[List, Dict, Dict]:
    """运行优化测试"""
    print(f"\n\n{'='*60}")
    print("Optimization Test")
    print(f"{'='*60}")
    
    # 重置系统并添加任务
    system.reset()
    for task in base_tasks:
        system.scheduler.add_task(task)
    
    # 创建优化器
    optimizer = SchedulingOptimizer(system.scheduler)
    
    # 获取硬件资源名称
    resources = system.get_resource_names()
    
    print("\n定义优化搜索空间...")
    
    # 为每个任务定义搜索空间
    for task in base_tasks:
        # 确定允许的优先级
        if task.task_id == "T1":  # 安全关键任务
            allowed_priorities = [TaskPriority.CRITICAL]
        elif task.priority == TaskPriority.HIGH:
            allowed_priorities = [TaskPriority.CRITICAL, TaskPriority.HIGH]
        elif task.priority == TaskPriority.NORMAL:
            allowed_priorities = [TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]
        else:
            allowed_priorities = [TaskPriority.NORMAL, TaskPriority.LOW]
        
        # 确定允许的运行时类型
        if task.uses_dsp and task.uses_npu:
            allowed_runtime_types = list(RuntimeType)  # 两种都可以
        else:
            allowed_runtime_types = [task.runtime_type]  # 保持原样
        
        # 收集分段选项
        segmentation_options = {}
        for segment in task.segments:
            if segment.segment_id in task.preset_cut_configurations:
                configs = task.preset_cut_configurations[segment.segment_id]
                segmentation_options[segment.segment_id] = list(range(len(configs)))
        
        # 确定可用核心
        available_cores = {}
        if task.uses_npu:
            available_cores[ResourceType.NPU] = resources["NPU"]
        if task.uses_dsp:
            available_cores[ResourceType.DSP] = resources["DSP"]
        
        # 定义搜索空间
        search_space = SchedulingSearchSpace(
            task_id=task.task_id,
            allowed_priorities=allowed_priorities,
            allowed_runtime_types=allowed_runtime_types,
            segmentation_options=segmentation_options,
            available_cores=available_cores
        )
        
        optimizer.define_search_space(task.task_id, search_space)
        
        print(f"\n{task.task_id} ({task.name}) 搜索空间:")
        print(f"  优先级选项: {[p.name for p in allowed_priorities]}")
        print(f"  运行时选项: {[r.value for r in allowed_runtime_types]}")
        print(f"  分段配置: {segmentation_options}")
    
    # 设置优化目标
    optimizer.objective = SchedulingObjective(
        latency_weight=2.0,      # 重视延迟
        throughput_weight=1.5,   # 重视吞吐量
        utilization_weight=1.0,  # 平衡资源利用
        priority_violation_weight=5.0,  # 严格惩罚违反要求
        overhead_weight=0.8      # 考虑分段开销
    )
    
    # 运行贪心优化
    print("\n运行贪心优化算法...")
    time_window = 500.0
    greedy_solution = optimizer.optimize_greedy(time_window=time_window, iterations=5)
    
    # 应用优化方案
    apply_optimization_solution(system.scheduler, greedy_solution)
    
    # 评估优化后的性能
    optimized_results = system.schedule(time_window)
    
    if optimized_results:
        print(f"\n✅ 优化后调度成功: {len(optimized_results)} 个事件")
        
        # 验证
        is_valid, conflicts = validate_schedule(system.scheduler)
        if is_valid:
            print("✅ 优化后调度验证通过")
        else:
            print(f"❌ 优化后发现 {len(conflicts)} 个冲突")
        
        # 计算优化后的指标
        optimized_metrics = calculate_performance_metrics(system.scheduler, optimized_results, time_window)
        print_performance_summary(optimized_metrics)
        
        # 打印优化决策
        print("\n优化决策:")
        optimizer.print_solution(greedy_solution)
        
        return optimized_results, optimized_metrics, greedy_solution
    
    return None, {}, {}


def apply_optimization_solution(scheduler, solution):
    """应用优化方案"""
    for task_id, decision in solution.items():
        if task_id in scheduler.tasks:
            task = scheduler.tasks[task_id]
            
            # 应用优先级
            task.priority = decision.priority
            
            # 应用运行时类型
            task.runtime_type = decision.runtime_type
            
            # 应用分段配置
            for seg_id, config_idx in decision.segmentation_configs.items():
                if seg_id in task.preset_cut_configurations:
                    task.select_cut_configuration(seg_id, config_idx)


def compare_results(baseline_metrics: Dict, optimized_metrics: Dict):
    """比较优化前后的结果"""
    print(f"\n\n{'='*60}")
    print("优化效果对比")
    print(f"{'='*60}")
    
    print(f"\n{'指标':<20} {'基准':<15} {'优化后':<15} {'改进':<15}")
    print("-" * 65)
    
    # 违反数对比
    baseline_violations = baseline_metrics.get('total_violations', 0)
    optimized_violations = optimized_metrics.get('total_violations', 0)
    violation_improvement = baseline_violations - optimized_violations
    print(f"{'任务违反数':<20} {baseline_violations:<15} {optimized_violations:<15} "
          f"{'-' + str(violation_improvement) if violation_improvement > 0 else '+' + str(abs(violation_improvement)):<15}")
    
    # 平均延迟对比
    baseline_latency = baseline_metrics.get('avg_latency', 0)
    optimized_latency = optimized_metrics.get('avg_latency', 0)
    latency_improvement = ((baseline_latency - optimized_latency) / baseline_latency * 100) if baseline_latency > 0 else 0
    print(f"{'平均延迟(ms)':<20} {baseline_latency:<15.1f} {optimized_latency:<15.1f} "
          f"{'-' + str(abs(latency_improvement)) + '%' if latency_improvement > 0 else '+' + str(abs(latency_improvement)) + '%':<15}")
    
    # 资源利用率对比
    baseline_util = baseline_metrics.get('avg_utilization', 0)
    optimized_util = optimized_metrics.get('avg_utilization', 0)
    util_improvement = optimized_util - baseline_util
    print(f"{'平均资源利用率(%)':<20} {baseline_util:<15.1f} {optimized_util:<15.1f} "
          f"{'+' + str(abs(util_improvement)) + '%' if util_improvement > 0 else '-' + str(abs(util_improvement)) + '%':<15}")
    
    # 分段开销对比
    baseline_overhead = baseline_metrics.get('total_segmentation_overhead', 0)
    optimized_overhead = optimized_metrics.get('total_segmentation_overhead', 0)
    overhead_diff = optimized_overhead - baseline_overhead
    print(f"{'分段开销(ms)':<20} {baseline_overhead:<15.2f} {optimized_overhead:<15.2f} "
          f"{'+' + str(abs(overhead_diff)) if overhead_diff > 0 else '-' + str(abs(overhead_diff)):<15}")
    
    # 各资源利用率详情
    print("\n资源利用率详情:")
    baseline_res_util = baseline_metrics.get('resource_utilization', {})
    optimized_res_util = optimized_metrics.get('resource_utilization', {})
    
    for resource in sorted(set(baseline_res_util.keys()) | set(optimized_res_util.keys())):
        baseline_val = baseline_res_util.get(resource, 0)
        optimized_val = optimized_res_util.get(resource, 0)
        improvement = optimized_val - baseline_val
        print(f"  {resource:<10}: {baseline_val:>6.1f}% → {optimized_val:>6.1f}% "
              f"({'↑' if improvement > 0 else '↓'}{abs(improvement):.1f}%)")


def main():
    """主测试函数"""
    print("=" * 80)
    print("Dragon4 系统分段优化测试")
    print("=" * 80)
    
    # 应用修复补丁
    apply_scheduling_optimizer_fix()
    apply_validator_precision_fix()
    
    # 创建Dragon4系统（按照simple_seg_test的配置）
    config = Dragon4Config(
        npu_bandwidth=120.0,
        dsp_count=2,
        dsp_bandwidth=40.0,
        enable_segmentation=True,
        enable_precision_scheduling=False  # 暂时禁用精度调度，避免冲突
    )
    system = Dragon4System(config)
    
    # 1. 简单基准测试（不分段）
    print("\n\n### 测试1: 简单基准测试（无分段）###")
    simple_tasks = Dragon4Workload.create_simple_workload()
    baseline_results, baseline_metrics = run_baseline_test(system, simple_tasks, "Simple Baseline")
    
    # 2. 汽车工作负载基准测试（不分段）
    print("\n\n### 测试2: 汽车工作负载基准测试（无分段）###")
    auto_config_no_seg = WorkloadConfig(name="automotive", enable_segmentation=False)
    auto_tasks_no_seg = Dragon4Workload.create_automotive_workload(auto_config_no_seg)
    auto_baseline_results, auto_baseline_metrics = run_baseline_test(system, auto_tasks_no_seg, "Automotive Baseline")
    
    # 3. 汽车工作负载优化测试（支持分段）
    print("\n\n### 测试3: 汽车工作负载优化测试（支持分段）###")
    auto_config_seg = WorkloadConfig(name="automotive", enable_segmentation=True)
    auto_tasks_seg = Dragon4Workload.create_automotive_workload(auto_config_seg)
    auto_opt_results, auto_opt_metrics, solution = run_optimization_test(system, auto_tasks_seg)
    
    # 4. 对比结果
    if auto_baseline_metrics and auto_opt_metrics:
        compare_results(auto_baseline_metrics, auto_opt_metrics)
    
    # 5. 压力测试
    print("\n\n### 测试4: 压力测试 ###")
    stress_tasks = Dragon4Workload.create_stress_workload(8)
    stress_results, stress_metrics = run_baseline_test(system, stress_tasks, "Stress Test")
    
    # 可视化（可选）
    if auto_opt_results:
        try:
            print("\n生成优化后的调度可视化...")
            visualizer = ElegantSchedulerVisualizer(system.scheduler)
            visualizer.plot_schedule_timeline(
                schedule_results=auto_opt_results[:50],
                time_window=200.0,
                title="Dragon4 Optimized Schedule"
            )
        except Exception as e:
            print(f"可视化失败: {e}")
    
    print("\n\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
