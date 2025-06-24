#!/usr/bin/env python3
"""
时间线分析器
详细分析调度时间线，查找资源冲突
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dragon4_system import Dragon4System, Dragon4Config
from priority_scheduling_fix import apply_priority_scheduling_fix
from enums import TaskPriority, RuntimeType, ResourceType
from task import NNTask


def print_timeline(results, scheduler, time_window):
    """打印详细的时间线"""
    print("\n=== 详细时间线 ===")
    
    # 收集所有事件
    events = []
    for r in results:
        task = scheduler.tasks[r.task_id]
        events.append({
            'time': r.start_time,
            'type': 'start',
            'task': r.task_id,
            'priority': task.priority.name,
            'event': r
        })
        events.append({
            'time': r.end_time,
            'type': 'end',
            'task': r.task_id,
            'priority': task.priority.name,
            'event': r
        })
    
    # 按时间排序
    events.sort(key=lambda x: (x['time'], x['type'] == 'end'))
    
    # 打印时间线
    active_tasks = set()
    for event in events:
        if event['type'] == 'start':
            active_tasks.add(event['task'])
            print(f"{event['time']:7.1f}ms: [{event['priority']:6}] {event['task']} 开始")
            if len(active_tasks) > 1:
                print(f"         ⚠️  同时运行: {active_tasks}")
        else:
            active_tasks.discard(event['task'])
            print(f"{event['time']:7.1f}ms: [{event['priority']:6}] {event['task']} 结束")


def check_resource_conflicts(results):
    """检查资源冲突"""
    print("\n=== 资源冲突检查 ===")
    
    # 按资源分组
    resource_timeline = {}
    
    for r in results:
        for res_type, res_id in r.assigned_resources.items():
            if res_id not in resource_timeline:
                resource_timeline[res_id] = []
            resource_timeline[res_id].append({
                'task': r.task_id,
                'start': r.start_time,
                'end': r.end_time
            })
    
    # 检查每个资源的冲突
    total_conflicts = 0
    for res_id, timeline in resource_timeline.items():
        print(f"\n{res_id}:")
        
        # 按开始时间排序
        timeline.sort(key=lambda x: x['start'])
        
        conflicts = []
        for i in range(len(timeline) - 1):
            curr = timeline[i]
            next_event = timeline[i + 1]
            
            if curr['end'] > next_event['start']:
                overlap = curr['end'] - next_event['start']
                conflicts.append({
                    'task1': curr['task'],
                    'task2': next_event['task'],
                    'overlap': overlap,
                    'time': next_event['start']
                })
                total_conflicts += 1
        
        if conflicts:
            print(f"  发现 {len(conflicts)} 个冲突:")
            for c in conflicts:
                print(f"    {c['task1']} 与 {c['task2']} 在 {c['time']:.1f}ms 重叠 {c['overlap']:.1f}ms")
        else:
            print(f"  无冲突")
    
    return total_conflicts


def analyze_scarcity_scenario():
    """分析资源稀缺场景"""
    print("=== 资源稀缺场景分析 ===\n")
    
    # 创建只有一个NPU的系统
    config = Dragon4Config(
        npu_bandwidth=120.0,
        dsp_count=0,
        enable_segmentation=False,
        enable_precision_scheduling=False
    )
    
    system = Dragon4System(config)
    
    # 确保只有一个NPU
    print(f"系统资源: {len(system.scheduler.resources[ResourceType.NPU])} 个NPU")
    
    # 手动设置只有一个NPU
    single_npu = system.scheduler.resources[ResourceType.NPU][0]
    system.scheduler.resources[ResourceType.NPU] = [single_npu]
    
    print(f"调整后: {len(system.scheduler.resources[ResourceType.NPU])} 个NPU")
    print(f"NPU ID: {single_npu.unit_id}")
    
    apply_priority_scheduling_fix(system.scheduler)
    
    # 创建3个任务
    task1 = NNTask("T1", "HighPriority", priority=TaskPriority.HIGH)
    task1.set_npu_only({120.0: 40.0}, "t1_seg")
    task1.set_performance_requirements(fps=20, latency=50)
    
    task2 = NNTask("T2", "NormalPriority", priority=TaskPriority.NORMAL)
    task2.set_npu_only({120.0: 40.0}, "t2_seg")
    task2.set_performance_requirements(fps=20, latency=50)
    
    task3 = NNTask("T3", "LowPriority", priority=TaskPriority.LOW)
    task3.set_npu_only({120.0: 40.0}, "t3_seg")
    task3.set_performance_requirements(fps=20, latency=50)
    
    system.scheduler.add_task(task1)
    system.scheduler.add_task(task2)
    system.scheduler.add_task(task3)
    
    # 调度150ms（足够看3个周期）
    print("\n调度150ms...")
    results = system.schedule(time_window=150.0)
    
    print(f"\n总共调度了 {len(results)} 个事件")
    
    # 打印时间线
    print_timeline(results, system.scheduler, 150.0)
    
    # 检查冲突
    conflicts = check_resource_conflicts(results)
    
    if conflicts > 0:
        print(f"\n❌ 发现 {conflicts} 个资源冲突！")
    else:
        print(f"\n✅ 没有资源冲突")
    
    # 统计每个任务的执行
    task_stats = {}
    for r in results:
        task_id = r.task_id
        if task_id not in task_stats:
            task_stats[task_id] = []
        task_stats[task_id].append(r.start_time)
    
    print("\n任务执行统计:")
    for task_id in ['T1', 'T2', 'T3']:
        times = task_stats.get(task_id, [])
        print(f"{task_id}: {len(times)}次 - 开始时间: {times}")
    
    # 计算理论上的最大可能执行次数
    print("\n理论分析:")
    print("每个任务需要40ms，周期50ms")
    print("3个任务完全执行需要120ms，但周期只有50ms")
    print("因此在每个50ms周期内，只能执行1.25个任务")
    print("150ms内最多执行 150/40 = 3.75 ≈ 3个任务")


def test_simple_conflict():
    """测试简单的冲突场景"""
    print("\n\n=== 简单冲突测试 ===")
    
    config = Dragon4Config(
        npu_bandwidth=120.0,
        dsp_count=0,
        enable_segmentation=False,
        enable_precision_scheduling=False
    )
    
    system = Dragon4System(config)
    system.scheduler.resources[ResourceType.NPU] = [system.scheduler.resources[ResourceType.NPU][0]]
    
    # 不应用优先级修复，看原始行为
    print("\n1. 原始调度器（无优先级修复）:")
    
    # 两个任务，执行时间之和超过周期
    task1 = NNTask("T1", "Task1", priority=TaskPriority.HIGH)
    task1.set_npu_only({120.0: 30.0}, "t1_seg")
    task1.set_performance_requirements(fps=20, latency=50)
    
    task2 = NNTask("T2", "Task2", priority=TaskPriority.LOW)
    task2.set_npu_only({120.0: 30.0}, "t2_seg")
    task2.set_performance_requirements(fps=20, latency=50)
    
    system.scheduler.add_task(task1)
    system.scheduler.add_task(task2)
    
    results = system.schedule(time_window=100.0)
    
    print(f"调度了 {len(results)} 个事件")
    
    # 检查前几个事件
    print("\n前10个事件:")
    for i, r in enumerate(results[:10]):
        print(f"{i+1}. {r.task_id}: {r.start_time:.1f} - {r.end_time:.1f}ms on {list(r.assigned_resources.values())}")


def main():
    """主函数"""
    print("时间线分析器\n")
    
    # 分析资源稀缺场景
    analyze_scarcity_scenario()
    
    # 测试简单冲突
    test_simple_conflict()
    
    print("\n\n=== 关键问题 ===")
    print("1. 检查是否真的只有一个NPU在工作")
    print("2. 检查是否存在资源冲突（任务重叠执行）")
    print("3. 理解为什么3个80%需求的任务都能100%满足")


if __name__ == "__main__":
    main()
