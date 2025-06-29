#!/usr/bin/env python3
"""
简单优化完整测试用例
基于real_task的任务，使用单NPU+单DSP资源
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scheduler import MultiResourceScheduler
from real_task import create_real_tasks
from modular_scheduler_fixes import apply_basic_fixes
from simple_constraint_optimizer import run_simple_optimization
from elegant_visualization import ElegantSchedulerVisualizer
from fixed_validation_and_metrics import validate_schedule_correctly
from collections import defaultdict


def create_single_resource_system():
    """创建单NPU+单DSP的系统"""
    
    print("🔧 创建单资源系统...")
    
    # 创建调度器
    scheduler = MultiResourceScheduler(enable_segmentation=True)
    
    # 添加单个NPU和DSP
    scheduler.add_npu("NPU_0", bandwidth=120.0)  # 使用较高带宽
    scheduler.add_dsp("DSP_0", bandwidth=40.0)
    
    print("  ✓ 添加 NPU_0 (120MHz)")
    print("  ✓ 添加 DSP_0 (40MHz)")
    
    return scheduler


def print_task_summary(tasks):
    """打印任务摘要"""
    
    print("\n📋 任务摘要:")
    print("-" * 80)
    print(f"{'任务ID':<8} {'名称':<15} {'优先级':<10} {'FPS要求':<10} {'资源需求':<15} {'依赖':<10}")
    print("-" * 80)
    
    for task in tasks:
        # 分析资源需求
        resources = []
        for seg in task.segments:
            res_type = seg.resource_type.value
            if res_type not in resources:
                resources.append(res_type)
        
        resources_str = "+".join(resources)
        deps_str = ",".join(task.dependencies) if task.dependencies else "无"
        
        print(f"{task.task_id:<8} {task.name:<15} {task.priority.name:<10} "
              f"{task.fps_requirement:<10.0f} {resources_str:<15} {deps_str:<10}")


def analyze_scheduling_results(scheduler, time_window):
    """分析调度结果"""
    
    print("\n📊 调度结果分析:")
    print("=" * 60)
    
    # 统计任务执行
    task_counts = defaultdict(int)
    for event in scheduler.schedule_history:
        task_counts[event.task_id] += 1
    
    # 计算FPS满足情况
    print("\n任务执行统计:")
    print(f"{'任务ID':<8} {'名称':<15} {'要求FPS':<10} {'实际次数':<10} {'期望次数':<10} {'满足率':<10} {'状态':<8}")
    print("-" * 80)
    
    total_tasks = len(scheduler.tasks)
    satisfied_tasks = 0
    
    for task_id, task in sorted(scheduler.tasks.items()):
        expected = int((time_window / 1000.0) * task.fps_requirement)
        actual = task_counts[task_id]
        rate = (actual / expected * 100) if expected > 0 else 0
        
        if rate >= 95:
            satisfied_tasks += 1
            status = "✅"
        elif rate >= 80:
            status = "⚠️"
        else:
            status = "❌"
        
        print(f"{task_id:<8} {task.name:<15} {task.fps_requirement:<10.0f} "
              f"{actual:<10} {expected:<10} {rate:<9.1f}% {status:<8}")
    
    fps_satisfaction_rate = (satisfied_tasks / total_tasks * 100) if total_tasks > 0 else 0
    print(f"\n总体FPS满足率: {fps_satisfaction_rate:.1f}% ({satisfied_tasks}/{total_tasks} 任务)")
    
    # 资源利用率
    resource_busy = defaultdict(float)
    for event in scheduler.schedule_history:
        duration = event.end_time - event.start_time
        for res_type, res_id in event.assigned_resources.items():
            resource_busy[res_id] += duration
    
    print("\n资源利用率:")
    for res_id in ["NPU_0", "DSP_0"]:
        busy_time = resource_busy.get(res_id, 0)
        utilization = (busy_time / time_window * 100)
        idle_time = time_window - busy_time
        print(f"  {res_id}: {utilization:.1f}% (忙碌: {busy_time:.1f}ms, 空闲: {idle_time:.1f}ms)")
    
    return fps_satisfaction_rate


def run_baseline_test(scheduler, tasks, time_window):
    """运行基线测试（无优化）"""
    
    print("\n" + "=" * 80)
    print("🏃 运行基线测试（无优化）")
    print("=" * 80)
    
    # 清空调度历史
    scheduler.schedule_history.clear()
    
    # 运行调度
    results = scheduler.priority_aware_schedule_with_segmentation(time_window)
    print(f"\n调度完成: {len(results)} 个事件")
    
    # 验证
    is_valid, conflicts = validate_schedule_correctly(scheduler)
    if is_valid:
        print("✅ 无资源冲突")
    else:
        print(f"❌ 发现 {len(conflicts)} 个资源冲突")
        for i, conflict in enumerate(conflicts[:3]):
            print(f"  - {conflict}")
    
    # 分析结果
    baseline_fps = analyze_scheduling_results(scheduler, time_window)
    
    return baseline_fps, len(conflicts)


def run_optimized_test(scheduler, tasks, time_window):
    """运行优化测试"""
    
    print("\n" + "=" * 80)
    print("🚀 运行简单约束优化")
    print("=" * 80)
    
    # 运行优化
    optimizer, best_config = run_simple_optimization(scheduler, time_window)
    
    # 应用最佳配置
    optimizer._apply_configuration(best_config)
    
    # 重新运行调度
    scheduler.schedule_history.clear()
    results = scheduler.priority_aware_schedule_with_segmentation(time_window)
    print(f"\n优化后调度完成: {len(results)} 个事件")
    
    # 验证
    is_valid, conflicts = validate_schedule_correctly(scheduler)
    if is_valid:
        print("✅ 无资源冲突")
    else:
        print(f"❌ 发现 {len(conflicts)} 个资源冲突")
    
    # 分析结果
    optimized_fps = analyze_scheduling_results(scheduler, time_window)
    
    return optimized_fps, len(conflicts)


def generate_visualization(scheduler, filename_prefix):
    """生成可视化"""
    
    try:
        viz = ElegantSchedulerVisualizer(scheduler)
        
        # 生成甘特图
        viz.plot_elegant_gantt(
            bar_height=0.4,
            spacing=1.0,
            save_filename=f"{filename_prefix}_gantt.png"
        )
        
        # 生成Chrome追踪
        viz.export_chrome_tracing(f"{filename_prefix}_trace.json")
        
        print(f"\n✅ 可视化已生成:")
        print(f"  - {filename_prefix}_gantt.png")
        print(f"  - {filename_prefix}_trace.json")
        
    except Exception as e:
        print(f"\n⚠️ 可视化生成失败: {e}")


def main():
    """主测试函数"""
    
    print("=" * 80)
    print("🧪 简单优化完整测试用例")
    print("=" * 80)
    
    # 1. 创建系统
    scheduler = create_single_resource_system()
    
    # 2. 应用基础修复
    fix_manager = apply_basic_fixes(scheduler)
    
    # 3. 创建任务
    tasks = create_real_tasks()
    for task in tasks:
        scheduler.add_task(task)
    
    print(f"\n✅ 添加了 {len(tasks)} 个任务")
    
    # 4. 打印任务摘要
    print_task_summary(tasks)
    
    # 5. 设置时间窗口
    time_window = 200.0
    print(f"\n⏱️ 时间窗口: {time_window}ms")
    
    # 6. 运行基线测试
    baseline_fps, baseline_conflicts = run_baseline_test(scheduler, tasks, time_window)
    
    # 生成基线可视化
    generate_visualization(scheduler, "baseline")
    
    # 7. 运行优化测试
    optimized_fps, optimized_conflicts = run_optimized_test(scheduler, tasks, time_window)
    
    # 生成优化后的可视化
    generate_visualization(scheduler, "optimized")
    
    # 8. 对比结果
    print("\n" + "=" * 80)
    print("📊 优化效果对比")
    print("=" * 80)
    
    print(f"\nFPS满足率:")
    print(f"  基线: {baseline_fps:.1f}%")
    print(f"  优化后: {optimized_fps:.1f}%")
    print(f"  提升: {optimized_fps - baseline_fps:+.1f}%")
    
    print(f"\n资源冲突:")
    print(f"  基线: {baseline_conflicts} 个")
    print(f"  优化后: {optimized_conflicts} 个")
    
    if optimized_conflicts == 0 and optimized_fps > baseline_fps:
        print("\n🎉 优化成功！在保证无冲突的前提下提升了FPS满足率")
    elif optimized_conflicts == 0:
        print("\n✅ 优化成功消除了所有冲突")
    else:
        print("\n⚠️ 优化未能完全消除冲突")
    
    print("\n💡 建议:")
    print("1. 查看生成的甘特图对比优化前后的调度")
    print("2. 在Chrome中打开trace文件查看详细时间线")
    print("3. 根据任务特性进一步调整优化策略")


if __name__ == "__main__":
    main()
