#!/usr/bin/env python3
"""
教程演示 - 新手入门指导
"""

from core import NNTask, SchedulerFactory
from core.enums import TaskPriority, RuntimeType, ResourceType
from config import SchedulerConfig
from utils import validate_schedule


def tutorial_step_by_step():
    """逐步教程演示"""
    print("🎓 NPU调度器入门教程")
    print("=" * 40)
    
    # 步骤1: 基本概念
    print("\n📚 步骤1: 基本概念")
    print("   • NPU调度器管理多个计算资源")
    print("   • 任务有不同优先级: CRITICAL > HIGH > NORMAL > LOW")
    print("   • 两种运行时: DSP_Runtime(绑定) vs ACPU_Runtime(流水线)")
    
    input("按Enter继续...")
    
    # 步骤2: 创建调度器
    print("\n🔧 步骤2: 创建调度器")
    config = SchedulerConfig.for_testing()
    scheduler = SchedulerFactory.create_scheduler(config)
    
    print(f"   ✅ 调度器已创建")
    print(f"   • NPU资源: {len(scheduler.resources[ResourceType.NPU])} 个")
    print(f"   • DSP资源: {len(scheduler.resources[ResourceType.DSP])} 个")
    
    input("按Enter继续...")
    
    # 步骤3: 创建任务
    print("\n📝 步骤3: 创建第一个任务")
    
    # 创建简单任务
    task = NNTask("TUTORIAL_T1", "MyFirstTask", priority=TaskPriority.HIGH)
    task.set_npu_only({4.0: 20}, "my_segment")
    task.set_performance_requirements(fps=30, latency=50)
    
    scheduler.add_task(task)
    
    print("   ✅ 任务已创建和添加")
    print(f"   • 任务ID: {task.task_id}")
    print(f"   • 任务名称: {task.name}")
    print(f"   • 优先级: {task.priority.name}")
    print(f"   • FPS需求: {task.fps_requirement}")
    print(f"   • 延迟需求: {task.latency_requirement}ms")
    
    input("按Enter继续...")
    
    # 步骤4: 运行调度
    print("\n⚡ 步骤4: 运行调度")
    print("   正在执行调度算法...")
    
    results = scheduler.priority_aware_schedule_with_segmentation(100.0)
    
    if results:
        print("   ✅ 调度成功!")
        
        # 显示结果
        for schedule in scheduler.schedule_history:
            print(f"   • 任务 {schedule.task_id}: {schedule.start_time:.1f}-{schedule.end_time:.1f}ms")
            print(f"     使用资源: {list(schedule.assigned_resources.values())}")
    else:
        print("   ❌ 调度失败")
        return
    
    input("按Enter继续...")
    
    # 步骤5: 结果分析
    print("\n📊 步骤5: 分析结果")
    
    is_valid, errors = validate_schedule(scheduler, verbose=False)
    
    print(f"   • 调度有效性: {'✅ 有效' if is_valid else '❌ 无效'}")
    print(f"   • 发现错误: {len(errors)} 个")
    
    # 性能统计
    if scheduler.schedule_history:
        total_time = max(s.end_time for s in scheduler.schedule_history)
        print(f"   • 总执行时间: {total_time:.1f}ms")
        print(f"   • 任务延迟: {total_time:.1f}ms")
    
    input("按Enter继续...")
    
    # 步骤6: 进阶学习
    print("\n🎯 步骤6: 下一步学习")
    print("   恭喜完成入门教程！")
    print("   📖 建议继续学习:")
    print("   • python main.py --mode basic      # 基础演示")
    print("   • python main.py --mode optimization # 优化演示")
    print("   • python main.py --verbose          # 详细输出")
    
    print("\n🎉 教程完成！")
    return scheduler


def quick_tutorial():
    """快速教程（非交互式）"""
    print("⚡ 快速教程演示")
    print("=" * 30)
    
    # 快速演示所有步骤
    config = SchedulerConfig.for_testing()
    scheduler = SchedulerFactory.create_scheduler(config)
    
    # 创建多个任务演示不同特性
    tasks = []
    
    # 关键任务
    critical_task = NNTask("QUICK_CRITICAL", "CriticalTask", TaskPriority.CRITICAL)
    critical_task.set_npu_only({8.0: 5}, "critical_seg")
    critical_task.set_performance_requirements(fps=100, latency=10)
    tasks.append(critical_task)
    
    # 高优先级任务
    high_task = NNTask("QUICK_HIGH", "HighPriorityTask", TaskPriority.HIGH) 
    high_task.set_dsp_npu_sequence([
        (ResourceType.DSP, {8.0: 3}, 0, "high_dsp"),
        (ResourceType.NPU, {4.0: 12}, 3, "high_npu")
    ])
    high_task.set_performance_requirements(fps=50, latency=20)
    tasks.append(high_task)
    
    # 普通任务
    normal_task = NNTask("QUICK_NORMAL", "NormalTask", TaskPriority.NORMAL)
    normal_task.set_npu_only({2.0: 30}, "normal_seg")
    normal_task.set_performance_requirements(fps=20, latency=50)
    tasks.append(normal_task)
    
    # 添加所有任务
    for task in tasks:
        scheduler.add_task(task)
    
    print(f"✅ 创建了 {len(tasks)} 个不同优先级的任务")
    
    # 运行调度
    results = scheduler.priority_aware_schedule_with_segmentation(150.0)
    
    if results:
        print(f"✅ 成功调度 {len(scheduler.schedule_history)} 个任务实例")
        
        # 显示执行顺序（验证优先级）
        sorted_schedules = sorted(scheduler.schedule_history, key=lambda s: s.start_time)
        print("\n📋 执行顺序 (验证优先级):")
        for i, schedule in enumerate(sorted_schedules, 1):
            task = scheduler.tasks[schedule.task_id]
            print(f"   {i}. {task.task_id} ({task.priority.name}) - "
                  f"{schedule.start_time:.1f}ms")
        
        # 验证调度
        is_valid, errors = validate_schedule(scheduler, verbose=False)
        conflict_errors = [e for e in errors if e.error_type == "RESOURCE_CONFLICT"]
        
        print(f"\n🔍 验证结果:")
        print(f"   • 资源冲突: {len(conflict_errors)} 个")
        print(f"   • 总错误: {len(errors)} 个")
        
        if len(conflict_errors) == 0:
            print("   ✅ 无资源冲突，调度正确！")
    
    return scheduler


def run_tutorial_demo(interactive=False):
    """运行教程演示"""
    if interactive:
        return tutorial_step_by_step()
    else:
        return quick_tutorial()


if __name__ == "__main__":
    import sys
    interactive = "--interactive" in sys.argv
    run_tutorial_demo(interactive)