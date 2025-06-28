#!/usr/bin/env python3
"""
运行 dragon4_segmentation_final_test.py 并生成可视化
修复版本：解决NPU资源冲突问题
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
from real_task import create_real_tasks
from models import SubSegment

# 导入修复
from dragon4_single_core_fix import apply_single_core_dragon4_fix

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
        max_segmentation_overhead_ratio=0.5
    )
    
    # 添加资源 - 保持原始bandwidth设置
    scheduler.add_npu("NPU_0", bandwidth=40)
    scheduler.add_dsp("DSP_0", bandwidth=40)
    
    # 应用RuntimeType修复 - 将T1改为DSP_Runtime实现绑定执行
    from runtime_type_fix import apply_runtime_type_fix
    apply_runtime_type_fix(scheduler)
    
    # 应用分段补丁
    apply_simple_segmentation_patch(scheduler)
    
    # 应用子段调度信息补丁
    apply_sub_segment_schedule_patch(scheduler)
    
    print("✅ 系统创建完成")
    
    return scheduler


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
        if segments:
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
            use_alt_colors=True
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
                if sub_segments:
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
    print("Dragon4 分段测试与可视化 (修复版)")
    print("=" * 60)
    
    # 1. 创建系统
    scheduler = create_test_system()
    
    # 2. 创建任务
    tasks = create_real_tasks()
    
    print(f"\n📋 创建测试任务:")
    for task in tasks:
        scheduler.add_task(task)
        # 修复：使用正确的属性名
        seg_strategy = task.segmentation_strategy.name if hasattr(task, 'segmentation_strategy') else "UNKNOWN"
        seg_info = "SEG" if seg_strategy != "NO_SEGMENTATION" else "NO SEG"
        print(f"  ✓ {task.task_id} {task.name}: {seg_info}")
    
    # 3. 应用RuntimeType修复 - 将T1改为DSP_Runtime实现绑定执行（必须在添加任务后）
    from runtime_type_fix import apply_runtime_type_fix
    apply_runtime_type_fix(scheduler)
    
    # 4. 应用命名补丁
    patch_sub_segment_naming(scheduler)
    
    # 5. 运行调度
    print(f"\n🚀 运行调度...")
    time_window = 200.0
    
    try:
        results = scheduler.priority_aware_schedule_with_segmentation(time_window)
        print(f"✅ 调度成功: {len(results)} 个事件")
        
        # 显示调度事件
        print(f"调度{time_window}ms...")
        for i, event in enumerate(results[:25]):  # 显示前25个事件
            task = scheduler.tasks[event.task_id]
            print(f"  {event.start_time:6.1f}ms: [{task.priority.name:8}] {event.task_id} 开始")
            
    except Exception as e:
        print(f"❌ 调度失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. 使用单核系统验证
    is_valid = scheduler.validate_schedule() if hasattr(scheduler, 'validate_schedule') else False
    
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
    else:
        print("❌ 调度结果存在资源冲突")
    
    if visualization_success:
        print("✅ 可视化生成成功")
        print("✅ 子段命名格式已更新为 XX_S1, XX_S2 形式")
    
    print("\n建议:")
    print("  - 查看生成的甘特图了解调度情况")
    print("  - 使用 Chrome Tracing 查看详细时间线")
    print("  - 检查资源时间线调试信息")
    
    if not is_valid:
        print("\n⚠️  重要提示:")
        print("  资源冲突问题需要进一步调试")
        print("  请检查dragon4_conflict_fix.py中的修复是否正确应用")


if __name__ == "__main__":
    main()
