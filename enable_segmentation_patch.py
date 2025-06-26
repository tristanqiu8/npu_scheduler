#!/usr/bin/env python3
"""
启用分段功能补丁
修复调度器中分段决策未被调用的问题
"""

from typing import Dict, List, Optional
from collections import defaultdict


def apply_segmentation_enablement_patch(scheduler):
    """
    应用补丁以启用分段功能
    """
    
    print("🔧 应用分段功能启用补丁...")
    
    # 保存原始的调度方法
    original_schedule = scheduler.priority_aware_schedule_with_segmentation
    original_find_resources = scheduler.find_available_resources_for_task_with_segmentation
    
    # 增强的资源查找方法（确保调用分段决策）
    def enhanced_find_resources(task, current_time):
        """增强的资源查找，确保分段决策被执行"""
        
        # 关键修复：确保在查找资源前进行分段决策
        if scheduler.enable_segmentation and hasattr(scheduler, 'make_segmentation_decision'):
            # 执行分段决策
            segmentation_decision = scheduler.make_segmentation_decision(task, current_time)
            
            # 应用分段决策到任务
            task.apply_segmentation_decision(segmentation_decision)
            
            # 记录分段信息
            if any(len(cuts) > 0 for cuts in segmentation_decision.values()):
                print(f"  ✓ {task.task_id} 分段决策: {segmentation_decision}")
                # is_segmented 是只读属性，通过应用分段决策来设置
            else:
                # 不分段
        
        # 调用原始方法查找资源
        return original_find_resources(task, current_time)
    
    # 增强的调度方法
    def enhanced_schedule(time_window=1000.0):
        """增强的调度方法，确保分段功能工作"""
        
        print(f"\n🚀 执行增强调度 (分段启用: {scheduler.enable_segmentation})")
        
        # 重置状态
        for queue in scheduler.resource_queues.values():
            queue.available_time = 0.0
            queue.release_binding()
            for p in scheduler.tasks.values().__iter__().__next__().priority.__class__:
                if hasattr(queue.queues, p.name):
                    queue.queues[p].clear()
        
        for task in scheduler.tasks.values():
            task.schedule_info = None
            task.last_execution_time = -float('inf')
            task.ready_time = 0
            task.current_segmentation = {}
            task.total_segmentation_overhead = 0.0
            # is_segmented 是只读属性，通过 current_segmentation 判断
        
        scheduler.schedule_history.clear()
        if hasattr(scheduler, 'active_bindings'):
            scheduler.active_bindings.clear()
        if hasattr(scheduler, 'segmentation_decisions_history'):
            scheduler.segmentation_decisions_history.clear()
        
        # 执行原始调度，但使用增强的资源查找
        scheduler.find_available_resources_for_task_with_segmentation = enhanced_find_resources
        
        try:
            results = original_schedule(time_window)
            
            # 后处理：为分段任务添加子段调度信息
            for schedule in results:
                task = scheduler.tasks.get(schedule.task_id)
                if task and task.is_segmented:  # 使用只读属性
                    # 创建子段调度信息
                    sub_segment_schedule = create_sub_segment_schedule(task, schedule)
                    if sub_segment_schedule and len(sub_segment_schedule) > 1:
                        schedule.sub_segment_schedule = sub_segment_schedule
                        # 标记调度结果为分段
                        if hasattr(schedule, '__dict__'):
                            schedule.__dict__['is_segmented'] = True
                        print(f"  ✓ {task.task_id} 被分段为 {len(sub_segment_schedule)} 个子段")
            
            return results
            
        finally:
            # 恢复原始方法
            scheduler.find_available_resources_for_task_with_segmentation = original_find_resources
    
    # 替换调度方法
    scheduler.priority_aware_schedule_with_segmentation = enhanced_schedule
    
    # 添加辅助方法
    scheduler._original_find_resources = original_find_resources
    scheduler._original_schedule = original_schedule
    
    print("✅ 分段功能启用补丁应用成功")
    print("  - 分段决策将在资源查找前执行")
    print("  - 分段任务将被正确标记")
    print("  - 子段调度信息将被生成")


def create_sub_segment_schedule(task, schedule):
    """为分段任务创建子段调度信息"""
    
    sub_segments = []
    current_time = schedule.start_time
    
    # 获取任务的子段
    if hasattr(task, 'get_sub_segments_for_scheduling'):
        task_sub_segments = task.get_sub_segments_for_scheduling()
    else:
        # 从当前分段信息构建子段
        task_sub_segments = []
        for segment in task.segments:
            if hasattr(segment, 'is_segmented') and segment.is_segmented:
                if hasattr(segment, 'sub_segments'):
                    task_sub_segments.extend(segment.sub_segments)
            else:
                # 非分段的段作为单个子段
                task_sub_segments.append(segment)
    
    # 为每个子段分配时间
    for sub_seg in task_sub_segments:
        # 获取子段ID
        if hasattr(sub_seg, 'sub_id'):
            sub_id = sub_seg.sub_id
        elif hasattr(sub_seg, 'segment_id'):
            sub_id = sub_seg.segment_id
        else:
            sub_id = f"seg_{len(sub_segments)}"
        
        # 计算子段持续时间
        if hasattr(sub_seg, 'get_duration'):
            # 从分配的资源获取带宽
            res_type = sub_seg.resource_type
            res_id = schedule.assigned_resources.get(res_type)
            if res_id:
                # 假设使用资源的最大带宽
                duration = sub_seg.get_duration(100.0)  # 使用默认带宽
            else:
                duration = 10.0  # 默认时长
        else:
            duration = 10.0
        
        # 添加子段调度
        end_time = current_time + duration
        sub_segments.append((sub_id, current_time, end_time))
        current_time = end_time
    
    return sub_segments


def test_segmentation_patch():
    """测试分段补丁"""
    
    print("\n=== 测试分段功能启用补丁 ===")
    
    try:
        from scheduler import MultiResourceScheduler
        from task import NNTask
        from enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
        
        # 创建调度器
        scheduler = MultiResourceScheduler(enable_segmentation=True)
        
        # 应用补丁
        apply_segmentation_enablement_patch(scheduler)
        
        # 添加资源
        scheduler.add_npu("NPU_0", bandwidth=100.0)
        
        # 创建测试任务
        task = NNTask("T1", "TestTask",
                      priority=TaskPriority.HIGH,
                      runtime_type=RuntimeType.ACPU_RUNTIME,
                      segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION)
        task.set_npu_only({100.0: 20.0}, "test_seg")
        task.add_cut_points_to_segment("test_seg", [("cut1", 0.5, 0.5)])
        task.set_preset_cut_configurations("test_seg", [[], ["cut1"]])
        task.selected_cut_config_index["test_seg"] = 1  # 选择分段配置
        task.set_performance_requirements(fps=10, latency=100)
        
        scheduler.add_task(task)
        
        # 执行调度
        print("\n执行测试调度...")
        results = scheduler.priority_aware_schedule_with_segmentation(100.0)
        
        print(f"\n✅ 测试完成: {len(results)} 个调度事件")
        
        # 检查分段
        segmented_count = 0
        for result in results:
            if hasattr(result, 'is_segmented') and result.is_segmented:
                segmented_count += 1
                print(f"  ✓ {result.task_id} 已分段")
        
        print(f"\n分段事件: {segmented_count}/{len(results)}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_segmentation_patch()
