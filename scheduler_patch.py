#!/usr/bin/env python3
"""
Scheduler Patch - 快速修复资源冲突问题
"""

def patch_scheduler(scheduler):
    """
    给调度器打补丁，修复资源冲突问题
    """
    # 保存原始方法
    original_find_pipelined = scheduler.find_pipelined_resources_with_segmentation
    original_find_bound = scheduler.find_bound_resources_with_segmentation
    
    def find_pipelined_resources_fixed(task, current_time):
        """修复版：检查资源可用性"""
        # 对于非分段任务，使用简单逻辑
        if not task.is_segmented:
            # 获取任务的资源需求
            resource_requirements = {}
            for seg in task.segments:
                resource_requirements[seg.resource_type] = seg
            
            assigned_resources = {}
            
            for res_type, segment in resource_requirements.items():
                best_resource = None
                earliest_available = float('inf')
                
                # 遍历该类型的所有资源
                for resource in scheduler.resources[res_type]:
                    queue = scheduler.resource_queues[resource.unit_id]
                    
                    # 检查资源可用时间
                    if queue.available_time <= current_time:
                        # 检查是否有更高优先级任务等待
                        if not queue.has_higher_priority_tasks(task.priority, current_time, task.task_id):
                            # 选择这个资源
                            best_resource = resource
                            earliest_available = queue.available_time
                            break
                    elif queue.available_time < earliest_available:
                        # 记录最早可用的资源，但不立即分配
                        best_resource = resource
                        earliest_available = queue.available_time
                
                if best_resource and earliest_available <= current_time:
                    assigned_resources[res_type] = best_resource.unit_id
                else:
                    # 如果没有可用资源，返回None
                    return None
            
            return assigned_resources if len(assigned_resources) == len(resource_requirements) else None
        else:
            # 对于分段任务，调用原始方法
            return original_find_pipelined(task, current_time)
    
    def find_bound_resources_fixed(task, current_time):
        """修复版：确保绑定资源的正确性"""
        # 类似的修复逻辑
        result = original_find_bound(task, current_time)
        
        if result:
            # 验证所有资源确实可用
            for res_type, res_id in result.items():
                queue = scheduler.resource_queues[res_id]
                if queue.available_time > current_time or queue.is_bound_to_other_task(task.task_id, current_time):
                    return None
        
        return result
    
    # 应用补丁
    scheduler.find_pipelined_resources_with_segmentation = find_pipelined_resources_fixed
    scheduler.find_bound_resources_with_segmentation = find_bound_resources_fixed
    
    print("✅ 调度器补丁已应用")
    return scheduler


def test_patched_scheduler():
    """测试打补丁后的调度器"""
    from scheduling_fix_simple import create_conflict_free_scenario
    from schedule_validator import validate_schedule
    from elegant_visualization import ElegantSchedulerVisualizer
    
    print("=== 测试补丁后的调度器 ===\n")
    
    # 创建场景
    scheduler, tasks = create_conflict_free_scenario()
    
    # 应用补丁
    patch_scheduler(scheduler)
    
    # 运行调度
    print("运行调度...")
    results = scheduler.priority_aware_schedule_with_segmentation(time_window=200.0)
    print(f"调度了 {len(results)} 个事件")
    
    # 验证结果
    is_valid, errors = validate_schedule(scheduler)
    
    if is_valid:
        print("\n✅ 补丁成功！没有资源冲突。")
        
        # 生成可视化
        viz = ElegantSchedulerVisualizer(scheduler)
        viz.plot_elegant_gantt(bar_height=0.4, spacing=1.0)
        viz.export_chrome_tracing("patched_schedule_trace.json")
        
        # 打印调度摘要
        print("\n调度摘要:")
        task_counts = {}
        for schedule in results:
            task_id = schedule.task_id
            task_counts[task_id] = task_counts.get(task_id, 0) + 1
        
        for task_id, count in sorted(task_counts.items()):
            task = scheduler.tasks[task_id]
            print(f"  {task_id}: {count} 次执行 (要求 FPS={task.fps_requirement})")
        
    else:
        print(f"\n❌ 仍有 {len(errors)} 个错误:")
        for i, error in enumerate(errors[:5]):
            print(f"  {i+1}. {error}")
    
    return is_valid


def test_with_segmentation():
    """测试启用分段功能"""
    from clean_viz_demo import create_realistic_scenario
    from schedule_validator import validate_schedule
    from elegant_visualization import ElegantSchedulerVisualizer
    
    print("\n\n=== 测试启用分段功能 ===\n")
    
    # 创建完整场景
    scheduler, tasks = create_realistic_scenario()
    
    # 应用补丁
    patch_scheduler(scheduler)
    
    # 限制任务数量进行测试
    print("使用前3个任务进行测试...")
    task_ids_to_remove = [t.task_id for t in tasks[3:]]
    for task_id in task_ids_to_remove:
        del scheduler.tasks[task_id]
    
    # 运行调度
    print("运行调度...")
    results = scheduler.priority_aware_schedule_with_segmentation(time_window=200.0)
    print(f"调度了 {len(results)} 个事件")
    
    # 验证结果
    is_valid, errors = validate_schedule(scheduler)
    
    if is_valid:
        print("\n✅ 分段功能也正常工作！")
        
        # 生成可视化
        viz = ElegantSchedulerVisualizer(scheduler)
        viz.plot_elegant_gantt(bar_height=0.35, spacing=0.8)
        viz.export_chrome_tracing("segmented_patched_trace.json")
        
    else:
        print(f"\n❌ 分段功能仍有问题: {len(errors)} 个错误")
        print("建议继续使用非分段模式")
    
    return is_valid


if __name__ == "__main__":
    # 测试基础补丁
    basic_success = test_patched_scheduler()
    
    if basic_success:
        # 如果基础测试成功，尝试分段功能
        segment_success = test_with_segmentation()
        
        if segment_success:
            print("\n🎉 所有功能都正常工作！")
        else:
            print("\n⚠️ 基础调度正常，但分段功能需要进一步修复")
    else:
        print("\n❌ 基础调度仍有问题，需要更深入的修复")
