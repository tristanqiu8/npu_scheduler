#!/usr/bin/env python3
"""
Dragon4 分段功能最终测试
修复所有已知问题的版本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict, Optional
from collections import defaultdict

# 核心导入
from scheduler import MultiResourceScheduler
from task import NNTask
from enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy

# 导入修复
from complete_resource_fix import apply_complete_resource_fix, validate_fixed_schedule


def apply_simple_segmentation_patch(scheduler):
    """简化的分段补丁，确保分段决策被执行"""
    
    print("🔧 应用简化分段补丁...")
    
    # 保存原始方法
    original_find_resources = scheduler.find_available_resources_for_task_with_segmentation
    
    def patched_find_resources(task, current_time):
        """修补的资源查找方法"""
        
        # 确保分段决策被执行
        if scheduler.enable_segmentation and hasattr(scheduler, 'make_segmentation_decision'):
            try:
                # 调用分段决策
                segmentation_decision = scheduler.make_segmentation_decision(task, current_time)
                
                # 应用分段决策
                task.apply_segmentation_decision(segmentation_decision)
                
                # 打印分段信息
                if any(len(cuts) > 0 for cuts in segmentation_decision.values()):
                    print(f"  ✓ {task.task_id} 应用分段: {segmentation_decision}")
                
            except Exception as e:
                print(f"  ⚠️ {task.task_id} 分段决策失败: {e}")
        
        # 调用原始方法
        return original_find_resources(task, current_time)
    
    # 替换方法
    scheduler.find_available_resources_for_task_with_segmentation = patched_find_resources
    
    print("✅ 简化分段补丁应用成功")


def create_test_system():
    """创建测试系统"""
    
    print("🔧 创建测试系统...")
    
    # 创建调度器
    scheduler = MultiResourceScheduler(
        enable_segmentation=True,
        max_segmentation_overhead_ratio=0.2
    )
    
    # 添加资源
    scheduler.add_npu("NPU_0", bandwidth=100.0)
    scheduler.add_npu("NPU_1", bandwidth=100.0)
    scheduler.add_dsp("DSP_0", bandwidth=50.0)
    
    # 应用资源修复
    apply_complete_resource_fix(scheduler)
    
    # 应用分段补丁
    apply_simple_segmentation_patch(scheduler)
    
    print("✅ 系统创建完成")
    
    return scheduler


def create_test_tasks():
    """创建测试任务"""
    
    tasks = []
    
    print("\n📋 创建测试任务:")
    
    # 任务1: 强制分段
    task1 = NNTask("T1", "ForcedSeg",
                   priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION)
    task1.set_npu_only({100.0: 30.0}, "seg1")
    task1.add_cut_points_to_segment("seg1", [
        ("cut1", 0.5, 0.5)
    ])
    task1.set_preset_cut_configurations("seg1", [
        [],        # Config 0: 不分段
        ["cut1"]   # Config 1: 分段
    ])
    # 选择分段配置
    task1.selected_cut_config_index["seg1"] = 1
    task1.set_performance_requirements(fps=20, latency=50)
    tasks.append(task1)
    print("  ✓ T1: FORCED_SEGMENTATION (配置1)")
    
    # 任务2: 自定义分段
    task2 = NNTask("T2", "CustomSeg",
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.CUSTOM_SEGMENTATION)
    task2.set_npu_only({100.0: 25.0}, "seg2")
    task2.add_cut_points_to_segment("seg2", [
        ("cut_a", 0.33, 0.3),
        ("cut_b", 0.67, 0.3)
    ])
    task2.set_preset_cut_configurations("seg2", [
        [],
        ["cut_a"],
        ["cut_b"],
        ["cut_a", "cut_b"]
    ])
    # 选择双切分
    task2.selected_cut_config_index["seg2"] = 3
    task2.set_performance_requirements(fps=15, latency=70)
    tasks.append(task2)
    print("  ✓ T2: CUSTOM_SEGMENTATION (配置3: 双切分)")
    
    # 任务3: 自适应分段
    task3 = NNTask("T3", "AdaptiveSeg",
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION)
    task3.set_npu_only({100.0: 20.0}, "seg3")
    task3.add_cut_points_to_segment("seg3", [
        ("adapt", 0.5, 0.4)
    ])
    task3.set_performance_requirements(fps=25, latency=40)
    tasks.append(task3)
    print("  ✓ T3: ADAPTIVE_SEGMENTATION")
    
    # 任务4: 不分段（对比）
    task4 = NNTask("T4", "NoSeg",
                   priority=TaskPriority.LOW,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task4.set_npu_only({100.0: 15.0}, "seg4")
    task4.set_performance_requirements(fps=10, latency=100)
    tasks.append(task4)
    print("  ✓ T4: NO_SEGMENTATION (基准)")
    
    return tasks


def analyze_results(scheduler, results):
    """分析调度结果"""
    
    print(f"\n📊 调度结果分析:")
    print("=" * 80)
    
    # 任务统计
    task_stats = defaultdict(lambda: {
        'count': 0,
        'segmented_count': 0,
        'total_duration': 0.0,
        'segments_info': []
    })
    
    # 分析每个调度事件
    for i, schedule in enumerate(results[:20]):  # 只看前20个
        task = scheduler.tasks.get(schedule.task_id)
        if not task:
            continue
        
        stats = task_stats[schedule.task_id]
        stats['count'] += 1
        
        duration = schedule.end_time - schedule.start_time
        stats['total_duration'] += duration
        
        # 检查任务是否被分段
        is_task_segmented = task.is_segmented
        
        # 检查调度结果是否包含分段信息
        has_sub_segments = hasattr(schedule, 'sub_segment_schedule') and schedule.sub_segment_schedule
        
        if is_task_segmented or has_sub_segments:
            stats['segmented_count'] += 1
            
            # 收集段信息
            segment_info = {
                'event_index': i,
                'start_time': schedule.start_time,
                'duration': duration,
                'is_task_segmented': is_task_segmented,
                'has_sub_segments': has_sub_segments
            }
            
            if has_sub_segments:
                segment_info['num_sub_segments'] = len(schedule.sub_segment_schedule)
            
            stats['segments_info'].append(segment_info)
        
        # 打印前几个事件
        if i < 10:
            print(f"\n事件 {i+1}: {schedule.task_id}")
            print(f"  时间: {schedule.start_time:.1f} - {schedule.end_time:.1f}ms")
            print(f"  任务分段状态: {'是' if is_task_segmented else '否'}")
            print(f"  包含子段信息: {'是' if has_sub_segments else '否'}")
            
            if has_sub_segments:
                print(f"  子段数: {len(schedule.sub_segment_schedule)}")
    
    # 打印汇总
    print(f"\n📈 任务执行汇总:")
    print(f"{'任务':<6} {'策略':<25} {'执行次数':<10} {'分段次数':<10} {'分段率':<10}")
    print("-" * 70)
    
    total_segmented = 0
    for task_id in sorted(task_stats.keys()):
        task = scheduler.tasks.get(task_id)
        stats = task_stats[task_id]
        
        strategy = task.segmentation_strategy.name if task else "UNKNOWN"
        seg_rate = stats['segmented_count'] / stats['count'] * 100 if stats['count'] > 0 else 0
        
        print(f"{task_id:<6} {strategy:<25} {stats['count']:<10} "
              f"{stats['segmented_count']:<10} {seg_rate:<10.1f}%")
        
        total_segmented += stats['segmented_count']
    
    # 检查分段配置是否生效
    print(f"\n🔍 分段配置检查:")
    for task in scheduler.tasks.values():
        print(f"\n{task.task_id} ({task.segmentation_strategy.name}):")
        
        # 检查当前分段状态
        if task.current_segmentation:
            print(f"  当前分段配置: {task.current_segmentation}")
        
        # 检查段的分段状态
        for segment in task.segments:
            if segment.is_segmented:
                print(f"  段 {segment.segment_id}: 已分段")
                if hasattr(segment, 'sub_segments') and segment.sub_segments:
                    print(f"    子段数: {len(segment.sub_segments)}")
    
    return total_segmented > 0


def main():
    """主测试函数"""
    
    print("=" * 80)
    print("Dragon4 分段功能最终测试")
    print("=" * 80)
    
    # 1. 创建系统
    scheduler = create_test_system()
    
    # 2. 创建任务
    tasks = create_test_tasks()
    
    # 3. 添加任务
    for task in tasks:
        scheduler.add_task(task)
    
    # 4. 执行调度
    print(f"\n🚀 执行调度...")
    
    try:
        results = scheduler.priority_aware_schedule_with_segmentation(200.0)
        print(f"✅ 调度成功: {len(results)} 个事件")
    except Exception as e:
        print(f"❌ 调度失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 验证结果
    is_valid = validate_fixed_schedule(scheduler)
    print(f"\n验证结果: {'✅ 通过' if is_valid else '❌ 失败'}")
    
    # 6. 分析分段
    has_segmentation = analyze_results(scheduler, results)
    
    # 7. 总结
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")
    
    if has_segmentation:
        print("✅ 分段功能工作正常")
    else:
        print("⚠️  未检测到分段执行")
        print("\n可能的原因:")
        print("  1. make_segmentation_decision 未正确实现分段逻辑")
        print("  2. 分段条件过于严格")
        print("  3. 任务的分段配置未正确应用")
    
    if is_valid:
        print("✅ 调度结果无冲突")
    
    print("\n建议:")
    print("  - 检查 scheduler.make_segmentation_decision 的实现")
    print("  - 确认 task.apply_segmentation_decision 正确应用了分段")
    print("  - 验证 segment.apply_segmentation 生成了子段")


if __name__ == "__main__":
    main()
