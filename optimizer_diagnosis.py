#!/usr/bin/env python3
"""
优化器诊断脚本
深入分析优化器为什么没有效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dragon4_system import Dragon4System, Dragon4Config
from dragon4_workload import Dragon4Workload
from scheduling_optimizer import SchedulingOptimizer, SchedulingSearchSpace, SchedulingObjective
from scheduling_optimizer_fix import apply_scheduling_optimizer_fix
from priority_scheduling_fix import apply_priority_scheduling_fix
from enums import ResourceType, TaskPriority, RuntimeType
from task import NNTask


def test_candidate_generation():
    """测试候选方案生成"""
    print("=== 测试候选方案生成 ===\n")
    
    # 创建简单系统
    config = Dragon4Config(
        npu_bandwidth=120.0,
        dsp_count=2,
        enable_segmentation=False,  # 简化测试
        enable_precision_scheduling=False
    )
    system = Dragon4System(config)
    
    # 创建简单任务
    task = NNTask("T1", "TestTask", 
                  priority=TaskPriority.NORMAL,
                  runtime_type=RuntimeType.ACPU_RUNTIME)
    task.set_npu_only({120.0: 10.0}, "test_seg")
    task.set_performance_requirements(fps=20, latency=50)
    system.scheduler.add_task(task)
    
    # 创建优化器
    apply_scheduling_optimizer_fix()
    optimizer = SchedulingOptimizer(system.scheduler)
    
    # 定义搜索空间
    optimizer.define_search_space("T1", SchedulingSearchSpace(
        task_id="T1",
        allowed_priorities=[TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW],
        allowed_runtime_types=[RuntimeType.ACPU_RUNTIME],
        segmentation_options={},
        available_cores={ResourceType.NPU: ["NPU_0", "NPU_1"]}
    ))
    
    # 测试候选生成
    print("生成候选方案...")
    
    # 手动调用内部方法
    if hasattr(optimizer, 'generate_candidate_solutions'):
        candidates = optimizer.generate_candidate_solutions("T1", max_candidates=5)
        print(f"生成了 {len(candidates)} 个候选方案:")
        for i, candidate in enumerate(candidates):
            print(f"  候选{i+1}: Priority={candidate.priority.name}, "
                  f"Runtime={candidate.runtime_type.value}")
    else:
        print("❌ 找不到generate_candidate_solutions方法")


def test_solution_application():
    """测试解决方案应用"""
    print("\n\n=== 测试解决方案应用 ===\n")
    
    # 创建系统
    config = Dragon4Config(
        npu_bandwidth=120.0,
        dsp_count=2,
        enable_segmentation=False,
        enable_precision_scheduling=False
    )
    system = Dragon4System(config)
    
    # 应用优先级修复
    apply_priority_scheduling_fix(system.scheduler)
    
    # 创建两个任务
    task1 = NNTask("T1", "HighTask", priority=TaskPriority.HIGH)
    task1.set_npu_only({120.0: 10.0}, "t1_seg")
    task1.set_performance_requirements(fps=10, latency=100)
    
    task2 = NNTask("T2", "LowTask", priority=TaskPriority.LOW)
    task2.set_npu_only({120.0: 5.0}, "t2_seg")
    task2.set_performance_requirements(fps=50, latency=20)
    
    system.scheduler.add_task(task1)
    system.scheduler.add_task(task2)
    
    print("初始状态:")
    print(f"  T1: Priority={task1.priority.name}")
    print(f"  T2: Priority={task2.priority.name}")
    
    # 运行调度
    results1 = system.schedule(time_window=100.0)
    count1 = {}
    for r in results1:
        count1[r.task_id] = count1.get(r.task_id, 0) + 1
    print(f"\n初始调度结果:")
    print(f"  T1: {count1.get('T1', 0)}次")
    print(f"  T2: {count1.get('T2', 0)}次")
    
    # 手动交换优先级
    print("\n交换优先级...")
    task1.priority = TaskPriority.LOW
    task2.priority = TaskPriority.HIGH
    
    print(f"  T1: Priority={task1.priority.name}")
    print(f"  T2: Priority={task2.priority.name}")
    
    # 清空调度历史
    system.scheduler.schedule_history.clear()
    for task in system.scheduler.tasks.values():
        task.last_execution_time = -float('inf')
        if hasattr(task, 'schedule_info'):
            task.schedule_info = None
    
    # 重置资源队列
    for queue in system.scheduler.resource_queues.values():
        queue.available_time = 0.0
        if hasattr(queue, 'queues'):
            for p in TaskPriority:
                queue.queues[p].clear()
    
    # 重新调度
    results2 = system.schedule(time_window=100.0)
    count2 = {}
    for r in results2:
        count2[r.task_id] = count2.get(r.task_id, 0) + 1
    
    print(f"\n调整后调度结果:")
    print(f"  T1: {count2.get('T1', 0)}次")
    print(f"  T2: {count2.get('T2', 0)}次")
    
    # 分析变化
    if count1 == count2:
        print("\n❌ 优先级调整没有效果！")
    else:
        print("\n✅ 优先级调整有效果")


def test_optimizer_evaluation():
    """测试优化器评估函数"""
    print("\n\n=== 测试优化器评估 ===\n")
    
    # 创建系统
    config = Dragon4Config(
        npu_bandwidth=120.0,
        dsp_count=1,
        enable_segmentation=False,
        enable_precision_scheduling=False
    )
    system = Dragon4System(config)
    
    # 创建会违反要求的任务
    task = NNTask("T1", "ImpossibleTask", priority=TaskPriority.NORMAL)
    task.set_npu_only({120.0: 50.0}, "test_seg")  # 50ms执行时间
    task.set_performance_requirements(fps=30, latency=35)  # 要求33ms间隔，35ms延迟
    system.scheduler.add_task(task)
    
    # 创建优化器
    apply_scheduling_optimizer_fix()
    optimizer = SchedulingOptimizer(system.scheduler)
    
    # 设置目标
    optimizer.objective = SchedulingObjective(
        latency_weight=1.0,
        throughput_weight=2.0,
        utilization_weight=0.0,
        priority_violation_weight=5.0,
        overhead_weight=0.0
    )
    
    # 手动创建解决方案
    from scheduling_optimizer import SchedulingDecisionVariable
    solution = {
        "T1": SchedulingDecisionVariable(
            task_id="T1",
            priority=TaskPriority.NORMAL,
            runtime_type=RuntimeType.ACPU_RUNTIME,
            segmentation_configs={},
            core_assignments={}
        )
    }
    
    # 评估
    print("评估解决方案...")
    score, metrics = optimizer.evaluate_solution(solution, time_window=100.0)
    
    print(f"\n评估结果:")
    print(f"  分数: {score:.2f}")
    print(f"  指标: {metrics}")
    
    # 尝试高优先级
    solution["T1"].priority = TaskPriority.CRITICAL
    score2, metrics2 = optimizer.evaluate_solution(solution, time_window=100.0)
    
    print(f"\n提高优先级后:")
    print(f"  分数: {score2:.2f} (变化: {score2 - score:.2f})")
    print(f"  指标: {metrics2}")


def test_scheduling_differences():
    """测试不同优先级的调度差异"""
    print("\n\n=== 测试调度差异 ===\n")
    
    for priority in [TaskPriority.LOW, TaskPriority.NORMAL, TaskPriority.HIGH]:
        # 创建新系统
        config = Dragon4Config(
            npu_bandwidth=120.0,
            dsp_count=1,
            enable_segmentation=False,
            enable_precision_scheduling=False
        )
        system = Dragon4System(config)
        
        # 创建任务
        task = NNTask("T1", f"Task_{priority.name}", priority=priority)
        task.set_npu_only({120.0: 10.0}, "test_seg")
        task.set_performance_requirements(fps=20, latency=50)
        system.scheduler.add_task(task)
        
        # 调度
        results = system.schedule(time_window=100.0)
        
        print(f"\n{priority.name} 优先级:")
        print(f"  调度事件数: {len(results)}")
        if results:
            print(f"  第一个事件: {results[0].start_time:.1f}ms")
            print(f"  最后事件: {results[-1].start_time:.1f}ms")


def main():
    """主函数"""
    print("优化器诊断\n")
    
    # 1. 测试候选生成
    test_candidate_generation()
    
    # 2. 测试解决方案应用
    test_solution_application()
    
    # 3. 测试优化器评估
    test_optimizer_evaluation()
    
    # 4. 测试调度差异
    test_scheduling_differences()
    
    print("\n\n=== 诊断结论 ===")
    print("请检查上述测试结果，特别关注:")
    print("1. 候选方案是否正确生成")
    print("2. 优先级改变是否影响调度")
    print("3. 评估函数是否正确计算分数")
    print("4. 不同优先级是否产生不同的调度结果")


if __name__ == "__main__":
    main()
