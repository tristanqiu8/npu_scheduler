#!/usr/bin/env python3
"""
简单的FIFO测试脚本
快速验证FIFO修复是否有效
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scheduler import MultiResourceScheduler
from minimal_fifo_fix import apply_minimal_fifo_fix
from priority_scheduling_fix import apply_priority_scheduling_fix
from comprehensive_segmentation_patch import apply_comprehensive_segmentation_patch
from real_task import create_real_tasks
from schedule_validator import validate_schedule


def simple_test():
    """简单测试FIFO功能"""
    
    print("=== 简单FIFO测试 ===\n")
    
    # 1. 创建调度器
    print("1. 创建调度器...")
    scheduler = MultiResourceScheduler(
        enable_segmentation=True,
        max_segmentation_overhead_ratio=0.5
    )
    
    # 2. 添加资源
    print("2. 添加资源...")
    scheduler.add_npu("NPU_0", bandwidth=40)
    scheduler.add_npu("NPU_1", bandwidth=40)
    scheduler.add_dsp("DSP_0", bandwidth=40)
    scheduler.add_dsp("DSP_1", bandwidth=40)
    
    # 3. 应用基础补丁
    print("3. 应用基础补丁...")
    apply_comprehensive_segmentation_patch(scheduler)
    apply_priority_scheduling_fix(scheduler)
    
    # 4. 创建任务
    print("4. 创建任务...")
    tasks = create_real_tasks()
    for task in tasks:
        scheduler.add_task(task)
    
    print(f"   添加了 {len(tasks)} 个任务")
    
    # 5. 应用FIFO修复
    print("5. 应用FIFO修复...")
    apply_minimal_fifo_fix(scheduler)
    
    # 6. 运行调度
    print("\n6. 运行调度 (100ms)...")
    results = scheduler.priority_aware_schedule_with_segmentation(time_window=100.0)
    print(f"   生成了 {len(results)} 个调度事件")
    
    # 7. 验证结果
    print("\n7. 验证结果...")
    is_valid, errors = validate_schedule(scheduler)
    
    if is_valid:
        print("   ✅ 没有资源冲突！")
    else:
        print(f"   ❌ 发现 {len(errors)} 个冲突:")
        for i, error in enumerate(errors[:3]):
            print(f"      {i+1}. {error}")
        if len(errors) > 3:
            print(f"      ... 还有 {len(errors)-3} 个冲突")
    
    # 8. 显示任务执行统计
    print("\n8. 任务执行统计:")
    task_stats = {}
    for event in results:
        if event.task_id not in task_stats:
            task_stats[event.task_id] = {
                'count': 0,
                'first': float('inf'),
                'priority': scheduler.tasks[event.task_id].priority.name,
                'order': getattr(scheduler.tasks[event.task_id], '_fifo_order', 999)
            }
        task_stats[event.task_id]['count'] += 1
        task_stats[event.task_id]['first'] = min(task_stats[event.task_id]['first'], event.start_time)
    
    # 按优先级分组显示
    from collections import defaultdict
    by_priority = defaultdict(list)
    for task_id, stats in task_stats.items():
        by_priority[stats['priority']].append((task_id, stats))
    
    for priority in ['CRITICAL', 'HIGH', 'NORMAL', 'LOW']:
        if priority in by_priority:
            print(f"\n   {priority}:")
            tasks_list = sorted(by_priority[priority], key=lambda x: x[1]['order'])
            for task_id, stats in tasks_list:
                print(f"     {task_id} (顺序={stats['order']}): "
                      f"执行{stats['count']}次, 首次@{stats['first']:.1f}ms")
    
    return is_valid, len(errors)


def test_execution_order():
    """测试同优先级任务的执行顺序"""
    
    print("\n\n=== 测试执行顺序 ===\n")
    
    # 创建简单场景
    scheduler = MultiResourceScheduler(enable_segmentation=False)
    scheduler.add_npu("NPU_0", bandwidth=120.0)
    
    apply_comprehensive_segmentation_patch(scheduler)
    apply_priority_scheduling_fix(scheduler)
    
    # 创建3个相同优先级的任务
    from task import NNTask
    from enums import TaskPriority
    
    for i in [3, 1, 2]:  # 故意乱序创建
        task = NNTask(f"T{i}", f"Task{i}", priority=TaskPriority.NORMAL)
        task.set_npu_only({120.0: 10.0}, f"seg{i}")
        task.set_performance_requirements(fps=10, latency=100)
        scheduler.add_task(task)
    
    # 应用FIFO修复
    apply_minimal_fifo_fix(scheduler)
    
    # 运行调度
    results = scheduler.priority_aware_schedule_with_segmentation(time_window=50.0)
    
    # 检查执行顺序
    execution_order = []
    for event in results:
        if event.task_id not in execution_order:
            execution_order.append(event.task_id)
    
    print(f"执行顺序: {' -> '.join(execution_order)}")
    
    if execution_order == ['T1', 'T2', 'T3']:
        print("✅ FIFO顺序正确！")
        return True
    else:
        print("❌ FIFO顺序不正确")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("简单FIFO测试")
    print("=" * 50)
    
    # 运行基本测试
    is_valid, conflicts = simple_test()
    
    # 运行顺序测试
    order_correct = test_execution_order()
    
    print("\n" + "=" * 50)
    print("测试总结:")
    print(f"  - 资源冲突: {'无' if is_valid else f'{conflicts}个'}")
    print(f"  - FIFO顺序: {'正确' if order_correct else '错误'}")
    
    if is_valid and order_correct:
        print("\n✅ 所有测试通过！")
    else:
        print("\n⚠️ 部分测试失败，需要进一步调试")
