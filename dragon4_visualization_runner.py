#!/usr/bin/env python3
"""
运行 dragon4_segmentation_final_test.py 并生成可视化
确保子段命名格式为 XX_S1, XX_S2 等
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
from models import SubSegment

# 导入修复
from complete_resource_fix import apply_complete_resource_fix, validate_fixed_schedule

# 导入可视化
from elegant_visualization import ElegantSchedulerVisualizer

# 导入子段调度信息补丁
from fix_sub_segment_schedule import apply_sub_segment_schedule_patch, enhance_visualization_for_sub_segments


def patch_sub_segment_naming(scheduler):
    """修补子段命名格式为 XX_S1, XX_S2 等"""
    
    print("🔧 应用子段命名格式补丁...")
    
    # 遍历所有任务
    for task in scheduler.tasks.values():
        if hasattr(task, 'segments'):
            for segment in task.segments:
                if hasattr(segment, 'sub_segments') and segment.sub_segments:
                    # 重命名子段
                    for i, sub_seg in enumerate(segment.sub_segments):
                        # 获取原始段ID的基础部分
                        base_id = segment.segment_id.split('_')[0] if segment.segment_id else task.task_id
                        # 使用 XX_S1, XX_S2 格式
                        new_sub_id = f"{base_id}_S{i+1}"
                        sub_seg.sub_id = new_sub_id
                        print(f"  ✓ 重命名子段: {sub_seg.sub_id}")


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
        result = original_find_resources(task, current_time)
        
        # 应用命名补丁
        if task.is_segmented:
            patch_task_sub_segments(task)
        
        return result
    
    # 替换方法
    scheduler.find_available_resources_for_task_with_segmentation = patched_find_resources
    
    print("✅ 简化分段补丁应用成功")


def patch_task_sub_segments(task):
    """为单个任务的子段应用命名格式"""
    if hasattr(task, 'segments'):
        for segment in task.segments:
            if hasattr(segment, 'sub_segments') and segment.sub_segments:
                for i, sub_seg in enumerate(segment.sub_segments):
                    # 获取任务ID作为基础
                    base_id = task.task_id
                    # 使用 XX_S1, XX_S2 格式
                    new_sub_id = f"{base_id}_S{i+1}"
                    sub_seg.sub_id = new_sub_id


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
    
    # 应用子段调度信息补丁
    apply_sub_segment_schedule_patch(scheduler)
    
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
    task3.set_npu_only({100.0: 35.0}, "seg3")
    task3.add_cut_points_to_segment("seg3", [
        ("adaptive_cut", 0.4, 0.4)
    ])
    task3.set_preset_cut_configurations("seg3", [
        [],
        ["adaptive_cut"]
    ])
    task3.set_performance_requirements(fps=12, latency=90)
    tasks.append(task3)
    print("  ✓ T3: ADAPTIVE_SEGMENTATION")
    
    # 任务4: 无分段
    task4 = NNTask("T4", "NoSeg",
                   priority=TaskPriority.LOW,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task4.set_npu_only({100.0: 20.0}, "seg4")
    task4.set_performance_requirements(fps=10, latency=100)
    tasks.append(task4)
    print("  ✓ T4: NO_SEGMENTATION")
    
    return tasks


def analyze_results(scheduler, results):
    """分析调度结果，检查分段情况"""
    
    print(f"\n📊 分析结果:")
    print(f"  - 总事件数: {len(results)}")
    
    segmented_events = 0
    task_segments = defaultdict(list)
    
    for result in results:
        if hasattr(result, 'sub_segment_schedule') and result.sub_segment_schedule:
            segmented_events += 1
            task_segments[result.task_id].extend(result.sub_segment_schedule)
    
    print(f"  - 分段事件数: {segmented_events}")
    
    # 打印每个任务的分段情况
    print(f"\n📋 任务分段详情:")
    for task_id, segments in task_segments.items():
        print(f"  {task_id}: {len(segments)} 个子段")
        for i, (sub_id, start, end) in enumerate(segments):
            print(f"    - {sub_id}: {start:.1f} - {end:.1f} ms")
    
    return segmented_events > 0


def generate_visualization(scheduler, results):
    """生成可视化"""
    
    print(f"\n🎨 生成可视化...")
    
    # 增强子段信息以便可视化
    enhance_visualization_for_sub_segments(scheduler)
    
    try:
        # 创建可视化器
        visualizer = ElegantSchedulerVisualizer(scheduler)
        
        # 生成甘特图
        print("  📊 生成甘特图...")
        visualizer.plot_elegant_gantt(
            bar_height=0.35,
            spacing=0.8,
            use_alt_colors=False
        )
        
        # 生成Chrome Tracing JSON
        trace_filename = "dragon4_segmentation_trace.json"
        print(f"  📄 生成Chrome Tracing JSON: {trace_filename}")
        visualizer.export_chrome_tracing(trace_filename)
        
        print(f"\n✅ 可视化生成成功!")
        print(f"  - 甘特图已显示")
        print(f"  - Chrome Tracing文件: {trace_filename}")
        print(f"  - 使用 chrome://tracing 加载JSON文件查看")
        
        # 检查子段命名
        print(f"\n📝 检查子段命名格式:")
        for task_id, task in scheduler.tasks.items():
            if task.is_segmented:
                sub_segments = task.get_sub_segments_for_scheduling()
                for sub_seg in sub_segments:
                    print(f"  - {task_id}: {sub_seg.sub_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ 可视化生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    
    print("=" * 60)
    print("Dragon4 分段测试与可视化")
    print("=" * 60)
    
    # 1. 创建系统
    scheduler = create_test_system()
    
    # 2. 创建任务
    tasks = create_test_tasks()
    
    # 3. 添加任务到调度器
    for task in tasks:
        scheduler.add_task(task)
    
    # 4. 应用命名补丁
    patch_sub_segment_naming(scheduler)
    
    # 5. 运行调度
    print(f"\n🚀 运行调度...")
    time_window = 200.0
    
    try:
        results = scheduler.priority_aware_schedule_with_segmentation(time_window)
        print(f"✅ 调度成功: {len(results)} 个事件")
    except Exception as e:
        print(f"❌ 调度失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. 验证结果
    is_valid = validate_fixed_schedule(scheduler)
    print(f"\n验证结果: {'✅ 通过' if is_valid else '❌ 失败'}")
    
    # 7. 分析分段
    has_segmentation = analyze_results(scheduler, results)
    
    # 8. 生成可视化
    visualization_success = generate_visualization(scheduler, results)
    
    # 9. 总结
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")
    
    if has_segmentation:
        print("✅ 分段功能工作正常")
    else:
        print("⚠️  未检测到分段执行")
    
    if is_valid:
        print("✅ 调度结果无冲突")
    
    if visualization_success:
        print("✅ 可视化生成成功")
        print("✅ 子段命名格式已更新为 XX_S1, XX_S2 形式")
    
    print("\n建议:")
    print("  - 查看生成的甘特图了解调度情况")
    print("  - 使用 Chrome Tracing 查看详细时间线")
    print("  - 检查子段命名是否符合要求")


if __name__ == "__main__":
    main()
