#!/usr/bin/env python3
"""
基础演示 - 展示核心调度功能
"""

from core import NNTask, SchedulerFactory
from core.enums import TaskPriority, RuntimeType, ResourceType
from config import SchedulerConfig
from utils import validate_schedule
from visualization import SchedulerVisualizer


def create_basic_tasks():
    """创建基础演示任务"""
    tasks = []
    
    # 关键优先级任务
    task1 = NNTask("T1", "SafetyMonitor", priority=TaskPriority.CRITICAL, 
                   runtime_type=RuntimeType.DSP_RUNTIME)
    task1.set_npu_only({4.0: 12, 8.0: 8}, "safety_segment")
    task1.set_performance_requirements(fps=60, latency=16)
    tasks.append(task1)
    
    # 高优先级任务
    task2 = NNTask("T2", "ObstacleDetection", priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.DSP_RUNTIME)
    task2.set_dsp_npu_sequence([
        (ResourceType.DSP, {8.0: 5}, 0, "detection_dsp"),
        (ResourceType.NPU, {4.0: 15}, 5, "detection_npu")
    ])
    task2.set_performance_requirements(fps=30, latency=33)
    tasks.append(task2)
    
    # 普通优先级任务
    task3 = NNTask("T3", "LaneDetection", priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME)
    task3.set_npu_only({2.0: 30, 4.0: 18}, "lane_segment")
    task3.set_performance_requirements(fps=20, latency=50)
    tasks.append(task3)
    
    return tasks


def run_basic_demo(config=None):
    """运行基础演示"""
    print("🚀 NPU调度器基础演示")
    print("=" * 40)
    
    # 使用配置
    if config is None:
        config = SchedulerConfig.for_development()
    
    # 创建调度器
    print("📋 初始化调度器...")
    scheduler = SchedulerFactory.create_scheduler(config)
    
    # 创建和添加任务
    print("📝 创建任务...")
    tasks = create_basic_tasks()
    for task in tasks:
        scheduler.add_task(task)
    
    print(f"   • 已添加 {len(tasks)} 个任务")
    print(f"   • NPU资源: {len(scheduler.resources[ResourceType.NPU])} 个")
    print(f"   • DSP资源: {len(scheduler.resources[ResourceType.DSP])} 个")
    
    # 运行调度
    print("\n⚡ 执行调度...")
    results = scheduler.priority_aware_schedule_with_segmentation(300.0)
    
    if results:
        print(f"   ✅ 调度成功，完成 {len(scheduler.schedule_history)} 个任务实例")
        
        # 显示调度结果
        print("\n📊 调度结果:")
        for schedule in scheduler.schedule_history[:5]:  # 显示前5个
            task = scheduler.tasks[schedule.task_id]
            print(f"   • {task.task_id} ({task.priority.name}): "
                  f"{schedule.start_time:.1f}-{schedule.end_time:.1f}ms")
        
        # 性能统计
        total_time = max(s.end_time for s in scheduler.schedule_history)
        avg_latency = sum(s.end_time - s.start_time for s in scheduler.schedule_history) / len(scheduler.schedule_history)
        utilization = scheduler.get_resource_utilization(total_time)
        avg_util = sum(utilization.values()) / len(utilization) if utilization else 0
        
        print(f"\n📈 性能统计:")
        print(f"   • 总完成时间: {total_time:.1f}ms")
        print(f"   • 平均任务延迟: {avg_latency:.1f}ms")
        print(f"   • 平均资源利用率: {avg_util:.1f}%")
        
        # 验证结果
        print("\n🔍 验证调度结果...")
        is_valid, errors = validate_schedule(scheduler, verbose=False)
        
        if is_valid:
            print("   ✅ 验证通过，没有发现错误")
        else:
            conflict_errors = [e for e in errors if e.error_type == "RESOURCE_CONFLICT"]
            if len(conflict_errors) == 0:
                print("   ✅ 没有资源冲突")
            else:
                print(f"   ⚠️ 发现 {len(conflict_errors)} 个资源冲突")
        
        # 可视化
        print("\n🎨 生成可视化...")
        try:
            visualizer = SchedulerVisualizer(scheduler)
            visualizer.plot_elegant_gantt()
            print("   ✅ 可视化图表已生成")
        except Exception as e:
            print(f"   ⚠️ 可视化失败: {e}")
    
    else:
        print("   ❌ 调度失败")
    
    return scheduler


if __name__ == "__main__":
    run_basic_demo()
