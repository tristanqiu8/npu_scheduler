#!/usr/bin/env python3
"""
优化演示 - 展示调度优化算法
"""

from core import NNTask, SchedulerFactory
from core.enums import TaskPriority, RuntimeType, ResourceType
from config import SchedulerConfig
try:
    from optimization import TaskSchedulerOptimizer, SchedulingSearchSpace
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False


def create_optimization_tasks():
    """创建优化演示任务"""
    tasks = []
    
    # 视觉处理任务
    task1 = NNTask("OPT_T1", "VisionProcessing", priority=TaskPriority.NORMAL)
    task1.set_npu_only({2.0: 40, 4.0: 25, 8.0: 15}, "vision_seg")
    task1.set_performance_requirements(fps=30, latency=35)
    tasks.append(task1)
    
    # 感知融合任务
    task2 = NNTask("OPT_T2", "SensorFusion", priority=TaskPriority.HIGH)
    task2.set_dsp_npu_sequence([
        (ResourceType.DSP, {8.0: 8}, 0, "fusion_dsp"),
        (ResourceType.NPU, {4.0: 18}, 8, "fusion_npu")
    ])
    task2.set_performance_requirements(fps=25, latency=40)
    tasks.append(task2)
    
    # 控制算法任务
    task3 = NNTask("OPT_T3", "ControlAlgorithm", priority=TaskPriority.CRITICAL)
    task3.set_npu_only({4.0: 10, 8.0: 6}, "control_seg")
    task3.set_performance_requirements(fps=100, latency=10)
    tasks.append(task3)
    
    return tasks


def run_optimization_demo(config=None):
    """运行优化演示"""
    print("🎯 NPU调度器优化演示")
    print("=" * 40)
    
    if not OPTIMIZATION_AVAILABLE:
        print("❌ 优化模块不可用，跳过优化演示")
        return None
    
    # 配置
    if config is None:
        config = SchedulerConfig.for_development()
    
    # 创建调度器和任务
    print("📋 初始化优化场景...")
    scheduler = SchedulerFactory.create_scheduler(config)
    tasks = create_optimization_tasks()
    
    for task in tasks:
        scheduler.add_task(task)
    
    print(f"   • 已添加 {len(tasks)} 个任务")
    
    # 运行基线调度
    print("\n📊 运行基线调度...")
    baseline_results = scheduler.priority_aware_schedule_with_segmentation(200.0)
    
    if not baseline_results:
        print("❌ 基线调度失败")
        return None
    
    baseline_metrics = scheduler.get_performance_metrics(200.0)
    print(f"   • 基线完成时间: {baseline_metrics.makespan:.1f}ms")
    print(f"   • 基线平均延迟: {baseline_metrics.average_latency:.1f}ms")
    print(f"   • 基线资源利用率: {baseline_metrics.average_utilization:.1f}%")
    
    # 创建优化器
    print("\n🔍 启动优化算法...")
    optimizer = TaskSchedulerOptimizer(scheduler)
    
    # 定义搜索空间
    for task in tasks:
        search_space = SchedulingSearchSpace(
            task_id=task.task_id,
            allowed_priorities=[TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.NORMAL],
            allowed_runtime_types=[RuntimeType.DSP_RUNTIME, RuntimeType.ACPU_RUNTIME],
            segmentation_options={},
            available_cores={}
        )
        optimizer.define_search_space(task.task_id, search_space)
    
    # 运行优化
    print("   🧬 执行贪心优化...")
    solution = optimizer.optimize_greedy(time_window=200.0, iterations=5)
    
    if solution:
        print(f"   ✅ 优化完成，评分: {solution.objective_value:.2f}")
        
        # 显示优化结果
        print("\n📈 优化配置:")
        for task_id, config in solution.task_configs.items():
            print(f"   • {task_id}: {config.priority.name} 优先级, {config.runtime_type.value}")
        
        # 性能对比
        if solution.metrics:
            improvement_makespan = ((baseline_metrics.makespan - solution.metrics.makespan) / baseline_metrics.makespan) * 100
            improvement_latency = ((baseline_metrics.average_latency - solution.metrics.average_latency) / baseline_metrics.average_latency) * 100
            
            print(f"\n🚀 性能改进:")
            print(f"   • 完成时间改进: {improvement_makespan:.1f}%")
            print(f"   • 延迟改进: {improvement_latency:.1f}%")
            print(f"   • 利用率: {solution.metrics.average_utilization:.1f}%")
    
    else:
        print("❌ 优化失败")
    
    return scheduler, solution


if __name__ == "__main__":
    run_optimization_demo()