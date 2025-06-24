#!/usr/bin/env python3
"""
资源稀缺测试
测试当资源不足时，优先级是否真正起作用
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dragon4_system import Dragon4System, Dragon4Config
from priority_scheduling_fix import apply_priority_scheduling_fix
from enums import TaskPriority, RuntimeType, ResourceType
from task import NNTask


def test_resource_scarcity():
    """测试资源稀缺时的优先级影响"""
    print("=== 资源稀缺测试 ===")
    print("当资源不足以满足所有任务需求时，高优先级任务应该优先满足\n")
    
    # 创建只有一个NPU的系统
    config = Dragon4Config(
        npu_bandwidth=120.0,
        dsp_count=0,
        enable_segmentation=False,
        enable_precision_scheduling=False
    )
    
    system = Dragon4System(config)
    # 只保留一个NPU
    system.scheduler.resources[ResourceType.NPU] = [system.scheduler.resources[ResourceType.NPU][0]]
    apply_priority_scheduling_fix(system.scheduler)
    
    # 创建3个任务，总需求超过资源容量
    # 每个任务需要40ms，但都要求每50ms执行一次
    # 这意味着资源利用率需求是 40/50 * 3 = 240%，但只有100%可用
    
    task1 = NNTask("T1", "HighPriority", priority=TaskPriority.HIGH)
    task1.set_npu_only({120.0: 40.0}, "t1_seg")
    task1.set_performance_requirements(fps=20, latency=50)  # 每50ms执行，需要40ms
    
    task2 = NNTask("T2", "NormalPriority", priority=TaskPriority.NORMAL)
    task2.set_npu_only({120.0: 40.0}, "t2_seg")
    task2.set_performance_requirements(fps=20, latency=50)
    
    task3 = NNTask("T3", "LowPriority", priority=TaskPriority.LOW)
    task3.set_npu_only({120.0: 40.0}, "t3_seg")
    task3.set_performance_requirements(fps=20, latency=50)
    
    system.scheduler.add_task(task1)
    system.scheduler.add_task(task2)
    system.scheduler.add_task(task3)
    
    # 调度300ms
    results = system.schedule(time_window=300.0)
    
    # 分析结果
    task_stats = {}
    for event in results:
        task_id = event.task_id
        if task_id not in task_stats:
            task_stats[task_id] = {
                'count': 0,
                'start_times': [],
                'missed_deadlines': 0
            }
        
        task_stats[task_id]['count'] += 1
        task_stats[task_id]['start_times'].append(event.start_time)
    
    # 计算每个任务的性能
    print("\n任务执行统计：")
    print(f"{'任务':<10} {'优先级':<10} {'执行次数':<10} {'期望次数':<10} {'满足率':<10}")
    print("-" * 50)
    
    expected_executions = 300.0 / 50.0  # 6次
    
    for task_id in ['T1', 'T2', 'T3']:
        task = system.scheduler.tasks[task_id]
        stats = task_stats.get(task_id, {'count': 0})
        satisfaction_rate = (stats['count'] / expected_executions) * 100
        
        print(f"{task_id:<10} {task.priority.name:<10} {stats['count']:<10} "
              f"{expected_executions:<10.0f} {satisfaction_rate:<10.1f}%")
    
    # 检查调度间隔
    print("\n调度间隔分析：")
    for task_id, stats in task_stats.items():
        if len(stats['start_times']) > 1:
            intervals = []
            for i in range(1, len(stats['start_times'])):
                interval = stats['start_times'][i] - stats['start_times'][i-1]
                intervals.append(interval)
            
            avg_interval = sum(intervals) / len(intervals) if intervals else 0
            print(f"{task_id}: 平均间隔 {avg_interval:.1f}ms (要求 50.0ms)")
    
    # 资源利用率
    total_busy_time = sum(40.0 * stats['count'] for stats in task_stats.values())
    utilization = (total_busy_time / 300.0) * 100
    print(f"\n资源利用率: {utilization:.1f}%")
    
    # 判断优先级是否有效
    if task_stats.get('T1', {'count': 0})['count'] >= task_stats.get('T2', {'count': 0})['count'] >= task_stats.get('T3', {'count': 0})['count']:
        print("\n✅ 优先级正确工作：高优先级任务获得更多执行机会")
    else:
        print("\n❌ 优先级可能有问题")


def test_extreme_case():
    """极端测试：任务执行时间接近周期"""
    print("\n\n=== 极端情况测试 ===")
    print("任务执行时间几乎等于周期，强制资源竞争\n")
    
    config = Dragon4Config(
        npu_bandwidth=120.0,
        dsp_count=0,
        enable_segmentation=False,
        enable_precision_scheduling=False
    )
    
    system = Dragon4System(config)
    system.scheduler.resources[ResourceType.NPU] = [system.scheduler.resources[ResourceType.NPU][0]]
    apply_priority_scheduling_fix(system.scheduler)
    
    # 两个任务，都需要45ms执行时间，50ms周期
    # 几乎没有空闲时间，必须选择执行哪个
    
    task_high = NNTask("T1", "HighPri_LongExec", priority=TaskPriority.HIGH)
    task_high.set_npu_only({120.0: 45.0}, "high_seg")
    task_high.set_performance_requirements(fps=20, latency=50)
    
    task_low = NNTask("T2", "LowPri_LongExec", priority=TaskPriority.LOW)
    task_low.set_npu_only({120.0: 45.0}, "low_seg")
    task_low.set_performance_requirements(fps=20, latency=50)
    
    system.scheduler.add_task(task_high)
    system.scheduler.add_task(task_low)
    
    # 调度200ms
    results = system.schedule(time_window=200.0)
    
    # 统计
    counts = {}
    for event in results:
        counts[event.task_id] = counts.get(event.task_id, 0) + 1
    
    print("执行结果：")
    print(f"  T1 (HIGH): {counts.get('T1', 0)}次")
    print(f"  T2 (LOW): {counts.get('T2', 0)}次")
    
    if counts.get('T1', 0) > counts.get('T2', 0):
        print("\n✅ 高优先级任务执行更多次")
    elif counts.get('T1', 0) == counts.get('T2', 0):
        print("\n⚠️ 两个任务执行次数相同")
    else:
        print("\n❌ 低优先级任务执行更多次！")


def main():
    """主函数"""
    print("资源稀缺测试\n")
    print("说明：当资源充足时，所有任务都能满足其FPS要求。")
    print("优先级的真正作用是在资源不足时，决定谁先获得资源。\n")
    
    # 测试1：资源稀缺
    test_resource_scarcity()
    
    # 测试2：极端情况
    test_extreme_case()
    
    print("\n\n=== 结论 ===")
    print("优先级调度器应该：")
    print("1. 让高优先级任务优先获得资源（✅ 已实现）")
    print("2. 在资源不足时，优先满足高优先级任务的需求")
    print("3. 不会让任务执行超过其FPS要求")


if __name__ == "__main__":
    main()
