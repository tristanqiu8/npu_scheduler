#!/usr/bin/env python3
"""
修复子段调度信息传递问题
确保分段任务的子段调度信息被正确记录和显示
"""

import sys
import os
from typing import List, Dict, Optional, Tuple


def apply_sub_segment_schedule_patch(scheduler):
    """
    应用子段调度信息补丁
    确保分段任务的子段调度详情被记录到调度结果中
    """
    
    print("🔧 应用子段调度信息补丁...")
    
    # 保存原始的调度方法
    original_schedule = scheduler.priority_aware_schedule_with_segmentation
    
    def patched_schedule(time_window):
        """增强的调度方法，确保子段信息被记录"""
        
        # 调用原始调度方法
        results = original_schedule(time_window)
        
        # 后处理：为分段任务添加子段调度信息
        for i, schedule_info in enumerate(results):
            task = scheduler.tasks.get(schedule_info.task_id)
            
            if task and task.is_segmented:
                # 生成子段调度信息
                sub_segment_schedule = create_sub_segment_schedule_info(
                    task, schedule_info, scheduler
                )
                
                # 更新调度信息
                if sub_segment_schedule:
                    schedule_info.sub_segment_schedule = sub_segment_schedule
                    print(f"  ✓ 为 {task.task_id} 添加了 {len(sub_segment_schedule)} 个子段调度信息")
        
        return results
    
    # 替换调度方法
    scheduler.priority_aware_schedule_with_segmentation = patched_schedule
    
    # 同时补丁 schedule_task_at_time 方法
    if hasattr(scheduler, 'schedule_task_at_time'):
        original_schedule_task = scheduler.schedule_task_at_time
        
        def patched_schedule_task(task, current_time, assigned_resources):
            """增强的任务调度方法"""
            
            # 调用原始方法
            schedule_info = original_schedule_task(task, current_time, assigned_resources)
            
            # 如果是分段任务，确保子段信息
            if task.is_segmented and schedule_info:
                if not hasattr(schedule_info, 'sub_segment_schedule') or not schedule_info.sub_segment_schedule:
                    sub_segment_schedule = create_sub_segment_schedule_info(
                        task, schedule_info, scheduler
                    )
                    schedule_info.sub_segment_schedule = sub_segment_schedule
            
            return schedule_info
        
        scheduler.schedule_task_at_time = patched_schedule_task
    
    print("✅ 子段调度信息补丁应用成功")


def create_sub_segment_schedule_info(task, schedule_info, scheduler):
    """
    为分段任务创建子段调度信息
    使用正确的命名格式 XX_S1, XX_S2 等
    """
    
    sub_segment_schedule = []
    
    # 获取任务的所有子段
    sub_segments = task.get_sub_segments_for_scheduling()
    
    if not sub_segments:
        return sub_segment_schedule
    
    # 计算每个子段的时间
    current_time = schedule_info.start_time
    
    for i, sub_seg in enumerate(sub_segments):
        # 确保使用正确的命名格式
        sub_seg_id = f"{task.task_id}_S{i+1}"
        
        # 获取子段在对应资源上的执行时间
        if sub_seg.resource_type in schedule_info.assigned_resources:
            resource_id = schedule_info.assigned_resources[sub_seg.resource_type]
            
            # 查找资源带宽
            resource = None
            for res in scheduler.resources.get(sub_seg.resource_type, []):
                if res.unit_id == resource_id:
                    resource = res
                    break
            
            if resource:
                # 计算子段持续时间
                duration = sub_seg.get_duration(resource.bandwidth)
                
                # 添加段间缓冲（如果不是第一段）
                if i > 0:
                    current_time += 0.2  # 段间缓冲
                
                # 记录子段调度
                end_time = current_time + duration
                sub_segment_schedule.append((sub_seg_id, current_time, end_time))
                
                # 更新时间
                current_time = end_time
    
    return sub_segment_schedule


def enhance_visualization_for_sub_segments(scheduler):
    """
    增强可视化以更好地显示子段信息
    """
    
    print("🎨 增强子段可视化...")
    
    # 为调度历史中的每个事件检查并修复子段信息
    for schedule in scheduler.schedule_history:
        task = scheduler.tasks.get(schedule.task_id)
        
        if task and task.is_segmented:
            # 如果缺少子段信息，生成它
            if not hasattr(schedule, 'sub_segment_schedule') or not schedule.sub_segment_schedule:
                sub_segment_schedule = create_sub_segment_schedule_info(
                    task, schedule, scheduler
                )
                schedule.sub_segment_schedule = sub_segment_schedule
                print(f"  ✓ 修复 {task.task_id} 的子段可视化信息")


def test_sub_segment_patch():
    """测试子段补丁"""
    
    print("\n🧪 测试子段调度信息补丁...")
    
    from scheduler import MultiResourceScheduler
    from task import NNTask
    from enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
    
    # 创建测试调度器
    scheduler = MultiResourceScheduler(enable_segmentation=True)
    scheduler.add_npu("NPU_0", bandwidth=100.0)
    
    # 应用补丁
    apply_sub_segment_schedule_patch(scheduler)
    
    # 创建测试任务
    task = NNTask("TEST", "TestTask",
                  priority=TaskPriority.HIGH,
                  runtime_type=RuntimeType.ACPU_RUNTIME,
                  segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION)
    
    task.set_npu_only({100.0: 30.0}, "seg1")
    task.add_cut_points_to_segment("seg1", [("cut1", 0.5, 0.5)])
    task.set_preset_cut_configurations("seg1", [[], ["cut1"]])
    task.selected_cut_config_index["seg1"] = 1
    task.set_performance_requirements(fps=20, latency=50)
    
    # 应用分段
    task.apply_segmentation_decision({"seg1": ["cut1"]})
    
    # 添加任务并调度
    scheduler.add_task(task)
    
    try:
        results = scheduler.priority_aware_schedule_with_segmentation(100.0)
        
        # 检查结果
        if results:
            first_result = results[0]
            if hasattr(first_result, 'sub_segment_schedule') and first_result.sub_segment_schedule:
                print(f"✅ 测试通过: 子段调度信息已记录")
                for sub_id, start, end in first_result.sub_segment_schedule:
                    print(f"   - {sub_id}: {start:.1f} - {end:.1f} ms")
            else:
                print(f"❌ 测试失败: 缺少子段调度信息")
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数 - 可以单独运行测试"""
    print("=" * 60)
    print("子段调度信息补丁")
    print("=" * 60)
    
    test_sub_segment_patch()
    
    print("\n使用方法:")
    print("1. 在创建调度器后应用补丁:")
    print("   apply_sub_segment_schedule_patch(scheduler)")
    print("\n2. 在生成可视化前增强子段信息:")
    print("   enhance_visualization_for_sub_segments(scheduler)")


if __name__ == "__main__":
    main()
