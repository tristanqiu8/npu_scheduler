#!/usr/bin/env python3
"""
Dragon4 调试脚本
诊断优化器和调度器的问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dragon4_system import Dragon4System, Dragon4Config
from dragon4_workload import Dragon4Workload, WorkloadConfig
from scheduling_optimizer import SchedulingOptimizer, SchedulingSearchSpace, SchedulingObjective
from schedule_validator import validate_schedule
from enums import ResourceType, TaskPriority, RuntimeType


def test_optimizer_behavior():
    """测试优化器行为"""
    print("=== 优化器行为测试 ===\n")
    
    # 创建简单系统
    config = Dragon4Config(
        npu_bandwidth=120.0,
        dsp_count=2,
        enable_segmentation=True,
        enable_precision_scheduling=False  # 先禁用精度调度
    )
    system = Dragon4System(config)
    
    # 使用简单工作负载
    tasks = Dragon4Workload.create_simple_workload()
    for task in tasks:
        system.scheduler.add_task(task)
    
    # 创建优化器
    optimizer = SchedulingOptimizer(system.scheduler)
    
    # 只为一个任务定义搜索空间
    print("定义T2的搜索空间...")
    optimizer.define_search_space("T2", SchedulingSearchSpace(
        task_id="T2",
        allowed_priorities=[TaskPriority.HIGH, TaskPriority.NORMAL],
        allowed_runtime_types=[RuntimeType.ACPU_RUNTIME],
        segmentation_options={},
        available_cores={ResourceType.NPU: ["NPU_0", "NPU_1"]}
    ))
    
    # 设置简单目标
    optimizer.objective = SchedulingObjective(
        latency_weight=1.0,
        throughput_weight=1.0,
        utilization_weight=0.0,
        priority_violation_weight=2.0,
        overhead_weight=0.0
    )
    
    # 记录调度次数
    original_schedule_method = system.scheduler.priority_aware_schedule_with_segmentation
    schedule_count = 0
    
    def counting_schedule_wrapper(time_window=1000.0):
        nonlocal schedule_count
        schedule_count += 1
        print(f"  [调度 #{schedule_count}] time_window={time_window}")
        return original_schedule_method(time_window)
    
    system.scheduler.priority_aware_schedule_with_segmentation = counting_schedule_wrapper
    
    # 运行优化（只1次迭代）
    print("\n运行优化器（1次迭代）...")
    schedule_count = 0
    solution = optimizer.optimize_greedy(time_window=100.0, iterations=1)
    
    print(f"\n总调度次数: {schedule_count}")
    print(f"期望调度次数: ~{len(optimizer.search_spaces) * 2} (每个任务2次)")
    
    # 检查解决方案
    print("\n优化解决方案:")
    for task_id, decision in solution.items():
        print(f"  {task_id}: Priority={decision.priority.name}")


def test_precision_scheduler_conflicts():
    """测试精度调度器的冲突问题"""
    print("\n\n=== 精度调度器冲突测试 ===\n")
    
    # 创建系统（启用精度调度）
    config = Dragon4Config(
        npu_bandwidth=120.0,
        dsp_count=1,
        enable_segmentation=False,
        enable_precision_scheduling=True
    )
    system = Dragon4System(config)
    
    # 创建会产生冲突的简单任务
    from task import NNTask
    
    # 任务1：每20ms执行一次，耗时15ms
    task1 = NNTask("T1", "Fast", priority=TaskPriority.HIGH, runtime_type=RuntimeType.ACPU_RUNTIME)
    task1.set_npu_only({120.0: 15.0}, "fast_seg")
    task1.set_performance_requirements(fps=50, latency=20)
    
    # 任务2：每30ms执行一次，耗时20ms
    task2 = NNTask("T2", "Slow", priority=TaskPriority.NORMAL, runtime_type=RuntimeType.ACPU_RUNTIME)
    task2.set_npu_only({120.0: 20.0}, "slow_seg")
    task2.set_performance_requirements(fps=33, latency=30)
    
    system.scheduler.add_task(task1)
    system.scheduler.add_task(task2)
    
    # 执行调度
    print("执行调度...")
    results = system.schedule(time_window=100.0)
    
    if results:
        print(f"调度了 {len(results)} 个事件")
        
        # 手动检查冲突
        print("\n检查NPU_0上的事件:")
        npu0_events = sorted(
            [(r.start_time, r.end_time, r.task_id) for r in results 
             if 'NPU_0' in r.assigned_resources.values()],
            key=lambda x: x[0]
        )
        
        print(f"NPU_0上有 {len(npu0_events)} 个事件:")
        for i, (start, end, task_id) in enumerate(npu0_events[:10]):  # 只显示前10个
            print(f"  {i+1}. {task_id}: {start:.1f} - {end:.1f} ms")
        
        # 检查冲突
        conflicts = 0
        for i in range(len(npu0_events) - 1):
            curr_end = npu0_events[i][1]
            next_start = npu0_events[i+1][0]
            if curr_end > next_start:
                conflicts += 1
                print(f"\n  ⚠️ 冲突: 事件{i+1}结束({curr_end:.1f}) > 事件{i+2}开始({next_start:.1f})")
        
        print(f"\n发现 {conflicts} 个时间冲突")
        
        # 使用验证器
        is_valid, validator_conflicts = validate_schedule(system.scheduler)
        print(f"\n验证器结果: {'通过' if is_valid else f'失败 ({len(validator_conflicts)} 个冲突)'}")


def test_simple_optimization():
    """测试简单的优化场景"""
    print("\n\n=== 简单优化场景测试 ===\n")
    
    # 禁用所有补丁，使用最简单的配置
    config = Dragon4Config(
        npu_bandwidth=120.0,
        dsp_count=2,
        enable_segmentation=False,
        enable_precision_scheduling=False
    )
    system = Dragon4System(config)
    
    # 创建两个简单任务
    from task import NNTask
    
    # 高优先级任务（可以降级）
    task1 = NNTask("T1", "HighTask", priority=TaskPriority.HIGH, runtime_type=RuntimeType.ACPU_RUNTIME)
    task1.set_npu_only({120.0: 10.0}, "t1_seg")
    task1.set_performance_requirements(fps=10, latency=100)  # 宽松要求
    
    # 低优先级任务（应该升级）
    task2 = NNTask("T2", "LowTask", priority=TaskPriority.LOW, runtime_type=RuntimeType.ACPU_RUNTIME)
    task2.set_npu_only({120.0: 5.0}, "t2_seg")
    task2.set_performance_requirements(fps=50, latency=20)  # 严格要求
    
    system.scheduler.add_task(task1)
    system.scheduler.add_task(task2)
    
    print("初始配置:")
    print(f"  T1: Priority={task1.priority.name}, FPS要求={task1.fps_requirement}")
    print(f"  T2: Priority={task2.priority.name}, FPS要求={task2.fps_requirement}")
    
    # 运行基准测试
    print("\n运行基准测试...")
    baseline_results = system.schedule(time_window=200.0)
    
    # 计算基准性能
    task_counts = {}
    for r in baseline_results:
        task_counts[r.task_id] = task_counts.get(r.task_id, 0) + 1
    
    print(f"\n基准结果:")
    for task_id, count in task_counts.items():
        task = system.scheduler.tasks[task_id]
        achieved_fps = count * 1000.0 / 200.0
        print(f"  {task_id}: 执行{count}次, FPS={achieved_fps:.1f}/{task.fps_requirement}")
    
    # 手动调整优先级
    print("\n手动调整优先级...")
    task1.priority = TaskPriority.NORMAL
    task2.priority = TaskPriority.HIGH
    
    print(f"  T1: Priority={task1.priority.name}")
    print(f"  T2: Priority={task2.priority.name}")
    
    # 重新调度
    system.scheduler.schedule_history.clear()
    for task in system.scheduler.tasks.values():
        task.last_execution_time = -float('inf')
    
    adjusted_results = system.schedule(time_window=200.0)
    
    # 计算调整后性能
    task_counts = {}
    for r in adjusted_results:
        task_counts[r.task_id] = task_counts.get(r.task_id, 0) + 1
    
    print(f"\n调整后结果:")
    for task_id, count in task_counts.items():
        task = system.scheduler.tasks[task_id]
        achieved_fps = count * 1000.0 / 200.0
        print(f"  {task_id}: 执行{count}次, FPS={achieved_fps:.1f}/{task.fps_requirement}")


def main():
    """主调试函数"""
    print("Dragon4 系统调试\n")
    
    # 1. 测试优化器行为
    test_optimizer_behavior()
    
    # 2. 测试精度调度器冲突
    test_precision_scheduler_conflicts()
    
    # 3. 测试简单优化
    test_simple_optimization()
    
    print("\n\n=== 诊断总结 ===")
    print("1. 优化器可能在每次评估时进行多次冗余调度")
    print("2. 精度调度器可能没有正确防止时间冲突")
    print("3. 优化器可能没有正确应用配置更改")
    print("4. 建议逐步调试每个组件")


if __name__ == "__main__":
    main()
