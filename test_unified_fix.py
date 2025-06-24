#!/usr/bin/env python3
"""
测试统一Dragon4修复方案
替代所有其他补丁，确保零资源冲突
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict
from collections import defaultdict

# 核心导入
from scheduler import MultiResourceScheduler
from task import NNTask
from enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from unified_dragon4_fix import apply_unified_dragon4_fix, validate_unified_schedule

# 尝试导入Dragon4系统（如果可用）
try:
    from dragon4_workload import Dragon4Workload
    HAS_DRAGON4_WORKLOAD = True
except ImportError:
    HAS_DRAGON4_WORKLOAD = False


def create_test_scheduler():
    """创建测试调度器，应用统一修复"""
    
    print("🔧 创建测试调度器...")
    
    # 创建基础调度器（不启用其他修复，避免冲突）
    scheduler = MultiResourceScheduler(
        enable_segmentation=False,  # 先禁用分段，专注解决基础冲突
        max_segmentation_overhead_ratio=0.15
    )
    
    # 添加Dragon4硬件配置
    scheduler.add_npu("NPU_0", bandwidth=120.0)
    scheduler.add_npu("NPU_1", bandwidth=120.0)
    scheduler.add_dsp("DSP_0", bandwidth=40.0)
    scheduler.add_dsp("DSP_1", bandwidth=40.0)
    
    print("  ✓ 硬件配置: 2xNPU + 2xDSP")
    
    # 应用统一修复（这会替代所有其他补丁）
    apply_unified_dragon4_fix(scheduler)
    
    return scheduler


def create_test_workload() -> List[NNTask]:
    """创建测试工作负载"""
    
    if HAS_DRAGON4_WORKLOAD:
        # 使用完整的Dragon4工作负载
        return Dragon4Workload.create_simple_workload()
    else:
        # 使用简化的测试工作负载
        return create_simple_test_workload()


def create_simple_test_workload() -> List[NNTask]:
    """创建简化的测试工作负载"""
    
    tasks = []
    
    # 任务1: 高优先级，快速NPU任务
    task1 = NNTask("T1", "HighPriorityTask", 
                   priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task1.set_npu_only({120.0: 10.0}, "t1_seg")
    task1.set_performance_requirements(fps=30, latency=35)
    tasks.append(task1)
    
    # 任务2: 中等优先级，NPU任务
    task2 = NNTask("T2", "NormalPriorityTask", 
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task2.set_npu_only({120.0: 15.0}, "t2_seg")
    task2.set_performance_requirements(fps=20, latency=50)
    tasks.append(task2)
    
    # 任务3: DSP-NPU序列任务
    task3 = NNTask("T3", "DSPSequenceTask", 
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task3.set_dsp_npu_sequence([
        (ResourceType.DSP, {40.0: 5.0}, 0, "preprocess_seg"),
        (ResourceType.NPU, {120.0: 10.0}, 5, "inference_seg"),
    ])
    task3.set_performance_requirements(fps=15, latency=80)
    tasks.append(task3)
    
    # 任务4: 低优先级，长时间NPU任务
    task4 = NNTask("T4", "LowPriorityTask", 
                   priority=TaskPriority.LOW,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task4.set_npu_only({120.0: 20.0}, "t4_seg")
    task4.set_performance_requirements(fps=10, latency=100)
    tasks.append(task4)
    
    return tasks


def analyze_schedule_results(scheduler, results, time_window):
    """分析调度结果"""
    
    print(f"\n📊 调度结果分析 (时间窗口: {time_window}ms)")
    print("=" * 50)
    
    if not results:
        print("❌ 没有调度事件")
        return
    
    print(f"总调度事件: {len(results)}")
    
    # 1. 验证冲突
    is_valid, conflicts = validate_unified_schedule(scheduler)
    if is_valid:
        print("✅ 验证通过: 无资源冲突")
    else:
        print(f"❌ 发现 {len(conflicts)} 个冲突:")
        for i, conflict in enumerate(conflicts[:3]):
            print(f"  {i+1}. {conflict}")
    
    # 2. 资源利用率分析
    print(f"\n📈 资源利用率:")
    resource_busy_time = defaultdict(float)
    
    for result in results:
        duration = result.end_time - result.start_time
        for res_type, res_id in result.assigned_resources.items():
            resource_busy_time[res_id] += duration
    
    for res_id, busy_time in resource_busy_time.items():
        utilization = (busy_time / time_window) * 100
        print(f"  {res_id}: {utilization:.1f}% ({busy_time:.1f}ms)")
    
    # 3. 任务执行统计
    print(f"\n📋 任务执行统计:")
    task_counts = defaultdict(int)
    task_fps_achieved = {}
    
    for result in results:
        task_counts[result.task_id] += 1
    
    for task_id, count in task_counts.items():
        if task_id in scheduler.tasks:
            task = scheduler.tasks[task_id]
            fps_achieved = (count * 1000.0) / time_window
            fps_required = task.fps_requirement
            status = "✅" if fps_achieved >= fps_required * 0.9 else "❌"
            
            print(f"  {task_id}: {count}次 | {fps_achieved:.1f}/{fps_required:.1f} FPS {status}")
            task_fps_achieved[task_id] = fps_achieved
    
    # 4. 时间线分析
    print(f"\n🕒 资源时间线 (前10个事件):")
    
    # 按资源分组
    by_resource = defaultdict(list)
    for result in results:
        for res_type, res_id in result.assigned_resources.items():
            by_resource[res_id].append({
                'start': result.start_time,
                'end': result.end_time,
                'task': result.task_id
            })
    
    for res_id in sorted(by_resource.keys()):
        events = sorted(by_resource[res_id], key=lambda x: x['start'])
        print(f"\n  {res_id}:")
        
        for i, event in enumerate(events[:10]):  # 只显示前10个
            print(f"    {event['start']:6.1f} - {event['end']:6.1f} ms: {event['task']}")
        
        if len(events) > 10:
            print(f"    ... 还有 {len(events) - 10} 个事件")
        
        # 检查时间间隙
        gaps = []
        for i in range(len(events) - 1):
            gap = events[i+1]['start'] - events[i]['end']
            if gap > 0.1:  # 大于0.1ms的间隙
                gaps.append(gap)
        
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            print(f"    平均时间间隙: {avg_gap:.2f}ms")


def test_unified_fix():
    """测试统一修复方案"""
    
    print("🧪 测试统一Dragon4修复方案")
    print("=" * 60)
    
    # 1. 创建调度器
    scheduler = create_test_scheduler()
    
    # 2. 创建工作负载
    tasks = create_test_workload()
    print(f"\n📦 加载工作负载: {len(tasks)} 个任务")
    
    for task in tasks:
        scheduler.add_task(task)
        print(f"  + {task.task_id}: {task.priority.name} 优先级, {task.fps_requirement} FPS")
    
    # 3. 运行调度
    print(f"\n🚀 开始调度...")
    time_window = 500.0
    
    try:
        results = scheduler.priority_aware_schedule_with_segmentation(time_window)
        print(f"✅ 调度完成: {len(results)} 个事件")
    except Exception as e:
        print(f"❌ 调度失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 分析结果
    analyze_schedule_results(scheduler, results, time_window)
    
    # 5. 验证是否真正解决了冲突
    is_valid, conflicts = validate_unified_schedule(scheduler)
    
    print(f"\n🎯 最终验证结果:")
    if is_valid:
        print("✅ 成功! 统一修复方案有效，无资源冲突")
        return True
    else:
        print(f"❌ 失败! 仍有 {len(conflicts)} 个冲突")
        for conflict in conflicts:
            print(f"  - {conflict}")
        return False


def compare_with_original():
    """与原始调度器对比"""
    
    print(f"\n🔄 对比测试: 统一修复 vs 原始调度器")
    print("=" * 60)
    
    # 创建原始调度器（不应用修复）
    print("\n1️⃣ 测试原始调度器...")
    original_scheduler = MultiResourceScheduler(enable_segmentation=False)
    original_scheduler.add_npu("NPU_0", bandwidth=120.0)
    original_scheduler.add_npu("NPU_1", bandwidth=120.0)
    original_scheduler.add_dsp("DSP_0", bandwidth=40.0)
    original_scheduler.add_dsp("DSP_1", bandwidth=40.0)
    
    tasks = create_test_workload()
    for task in tasks:
        original_scheduler.add_task(task)
    
    try:
        original_results = original_scheduler.priority_aware_schedule_with_segmentation(500.0)
        print(f"  原始调度: {len(original_results)} 个事件")
        
        # 检查原始调度的冲突
        original_conflicts = check_basic_conflicts(original_scheduler)
        print(f"  原始冲突: {len(original_conflicts)} 个")
        
    except Exception as e:
        print(f"  原始调度失败: {e}")
        original_results = []
        original_conflicts = []
    
    # 测试统一修复调度器
    print("\n2️⃣ 测试统一修复调度器...")
    success = test_unified_fix()
    
    # 总结对比
    print(f"\n📊 对比总结:")
    print(f"  原始调度器: {len(original_results)} 事件, {len(original_conflicts)} 冲突")
    print(f"  统一修复版: {'成功' if success else '失败'}")
    
    if success and len(original_conflicts) > 0:
        print(f"✅ 统一修复有效: 从 {len(original_conflicts)} 个冲突降到 0 个")
    elif not success:
        print(f"❌ 统一修复需要进一步调优")


def check_basic_conflicts(scheduler):
    """基础冲突检查"""
    
    conflicts = []
    resource_timeline = defaultdict(list)
    
    # 构建时间线
    for schedule in scheduler.schedule_history:
        for res_type, res_id in schedule.assigned_resources.items():
            resource_timeline[res_id].append({
                'start': schedule.start_time,
                'end': schedule.end_time,
                'task': schedule.task_id
            })
    
    # 检查冲突
    for res_id, timeline in resource_timeline.items():
        timeline.sort(key=lambda x: x['start'])
        
        for i in range(len(timeline) - 1):
            curr = timeline[i]
            next_event = timeline[i + 1]
            
            if curr['end'] > next_event['start'] + 0.001:
                conflicts.append(
                    f"Resource {res_id}: {curr['task']} overlaps {next_event['task']}"
                )
    
    return conflicts


if __name__ == "__main__":
    print("统一Dragon4修复测试")
    print("=" * 80)
    
    # 运行对比测试
    compare_with_original()
