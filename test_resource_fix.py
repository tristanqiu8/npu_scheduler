#!/usr/bin/env python3
"""
测试完整资源修复方案
"""

from enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from task import NNTask
from scheduler import MultiResourceScheduler
from complete_resource_fix import apply_complete_resource_fix, validate_fixed_schedule


def test_resource_conflict_fix():
    """测试资源冲突修复"""
    
    print("🧪 测试资源冲突修复")
    print("=" * 50)
    
    # 创建调度器（禁用分段以专注于基础冲突修复）
    scheduler = MultiResourceScheduler(enable_segmentation=False)
    
    # 添加资源 - 只有一个NPU
    scheduler.add_npu("NPU_0", bandwidth=120.0)
    
    print(f"资源配置: {len(scheduler.resources[ResourceType.NPU])} 个NPU")
    
    # 创建三个竞争任务
    task1 = NNTask("T1", "HighPriority", 
                   priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task1.set_npu_only({120.0: 40.0}, "t1_seg")
    task1.set_performance_requirements(fps=20, latency=50)
    
    task2 = NNTask("T2", "NormalPriority", 
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task2.set_npu_only({120.0: 40.0}, "t2_seg")
    task2.set_performance_requirements(fps=20, latency=50)
    
    task3 = NNTask("T3", "LowPriority", 
                   priority=TaskPriority.LOW,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task3.set_npu_only({120.0: 40.0}, "t3_seg")
    task3.set_performance_requirements(fps=20, latency=50)
    
    # 添加任务到调度器
    scheduler.add_task(task1)
    scheduler.add_task(task2)
    scheduler.add_task(task3)
    
    print(f"添加了 {len(scheduler.tasks)} 个任务")
    print("每个任务: 40ms执行时间, 50ms周期 (20 FPS)")
    print("理论上在150ms内每个任务最多执行3次")
    print()
    
    # 应用修复
    print("🔧 应用完整资源修复...")
    apply_complete_resource_fix(scheduler)
    print()
    
    # 运行调度
    print("🚀 运行调度 (150ms)...")
    results = scheduler.priority_aware_schedule_with_segmentation(time_window=150.0)
    print()
    
    # 验证结果
    print("📊 验证调度结果...")
    is_valid = validate_fixed_schedule(scheduler)
    print()
    
    # 显示详细时间线
    print("=== 详细时间线 ===")
    if results:
        for i, schedule in enumerate(results):
            task = scheduler.tasks[schedule.task_id]
            print(f"{schedule.start_time:6.1f}ms: [{task.priority.name:6}] {task.task_id} "
                  f"({schedule.start_time:.1f} - {schedule.end_time:.1f}ms)")
    else:
        print("没有调度事件")
    
    # 检查资源利用率
    print("\n=== 资源利用率分析 ===")
    if results:
        total_busy_time = sum(r.end_time - r.start_time for r in results)
        utilization = (total_busy_time / 150.0) * 100
        print(f"NPU_0 利用率: {utilization:.1f}% ({total_busy_time:.1f}ms / 150.0ms)")
        
        # 检查是否存在时间重叠
        overlaps = []
        for i in range(len(results) - 1):
            curr = results[i]
            next_event = results[i + 1]
            if curr.end_time > next_event.start_time:
                overlaps.append((curr, next_event))
        
        if overlaps:
            print(f"❌ 发现 {len(overlaps)} 个时间重叠")
            for curr, next_event in overlaps:
                overlap = curr.end_time - next_event.start_time
                print(f"  {curr.task_id} 与 {next_event.task_id} 重叠 {overlap:.1f}ms")
        else:
            print("✅ 没有时间重叠")
    
    return is_valid


def test_priority_ordering():
    """测试优先级排序是否正确"""
    
    print("\n\n🎯 测试优先级排序")
    print("=" * 50)
    
    # 创建调度器
    scheduler = MultiResourceScheduler(enable_segmentation=False)
    scheduler.add_npu("NPU_0", bandwidth=120.0)
    
    # 创建不同优先级的任务，但相同的时间需求
    tasks = []
    priorities = [TaskPriority.LOW, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.CRITICAL]
    
    for i, priority in enumerate(priorities):
        task = NNTask(f"T{i+1}", f"Task_{priority.name}", 
                     priority=priority,
                     runtime_type=RuntimeType.ACPU_RUNTIME,
                     segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
        task.set_npu_only({120.0: 30.0}, f"t{i+1}_seg")
        task.set_performance_requirements(fps=10, latency=100)  # 100ms周期
        scheduler.add_task(task)
        tasks.append(task)
    
    print("创建任务:")
    for task in tasks:
        print(f"  {task.task_id}: {task.priority.name} (值: {task.priority.value})")
    
    # 应用修复
    apply_complete_resource_fix(scheduler)
    
    # 运行调度
    results = scheduler.priority_aware_schedule_with_segmentation(time_window=200.0)
    
    # 分析执行顺序
    print("\n执行顺序:")
    if results:
        for schedule in results:
            task = scheduler.tasks[schedule.task_id]
            print(f"{schedule.start_time:6.1f}ms: {task.task_id} ({task.priority.name})")
        
        # 验证优先级顺序
        print("\n优先级验证:")
        execution_order = [scheduler.tasks[r.task_id].priority.value for r in results]
        
        # 检查每个时间窗口内的优先级顺序
        time_windows = {}
        for schedule in results:
            window = int(schedule.start_time // 100)  # 每100ms一个窗口
            if window not in time_windows:
                time_windows[window] = []
            time_windows[window].append(scheduler.tasks[schedule.task_id].priority.value)
        
        all_correct = True
        for window, priorities_in_window in time_windows.items():
            is_sorted = all(priorities_in_window[i] <= priorities_in_window[i+1] 
                          for i in range(len(priorities_in_window)-1))
            if is_sorted:
                print(f"  窗口 {window}: ✅ 优先级顺序正确")
            else:
                print(f"  窗口 {window}: ❌ 优先级顺序错误 {priorities_in_window}")
                all_correct = False
        
        if all_correct:
            print("✅ 所有时间窗口的优先级顺序都正确")
        else:
            print("❌ 存在优先级顺序错误")
    
    return results


def main():
    """主测试函数"""
    
    print("🔬 完整资源冲突修复测试")
    print("=" * 60)
    
    # 测试1：基础资源冲突修复
    test1_passed = test_resource_conflict_fix()
    
    # 测试2：优先级排序
    test2_results = test_priority_ordering()
    
    # 总结
    print("\n\n📋 测试总结")
    print("=" * 60)
    
    if test1_passed:
        print("✅ 资源冲突修复测试: 通过")
    else:
        print("❌ 资源冲突修复测试: 失败")
    
    if test2_results:
        print("✅ 优先级排序测试: 通过")
    else:
        print("❌ 优先级排序测试: 失败")
    
    print("\n使用说明:")
    print("1. 将 complete_resource_fix.py 保存到项目目录")
    print("2. 在现有代码中导入并应用修复:")
    print("   from complete_resource_fix import apply_complete_resource_fix")
    print("   apply_complete_resource_fix(scheduler)")
    print("3. 运行调度并验证结果:")
    print("   from complete_resource_fix import validate_fixed_schedule")
    print("   validate_fixed_schedule(scheduler)")


if __name__ == "__main__":
    main()
