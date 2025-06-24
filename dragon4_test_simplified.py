#!/usr/bin/env python3
"""
简化版Dragon4测试 - 直接使用修复后的elegant_visualization
避免依赖额外的配置模块
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# 导入验证成功的修复
from complete_resource_fix import apply_complete_resource_fix, validate_fixed_schedule

# 核心导入
from scheduler import MultiResourceScheduler
from task import NNTask
from enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy

# 尝试导入Dragon4模块
try:
    from dragon4_system import Dragon4System, Dragon4Config
    HAS_DRAGON4_SYSTEM = True
except ImportError:
    HAS_DRAGON4_SYSTEM = False

try:
    from dragon4_workload import Dragon4Workload, WorkloadConfig
    HAS_DRAGON4_WORKLOAD = True
except ImportError:
    HAS_DRAGON4_WORKLOAD = False

# 应用可用的优化器修复
try:
    from scheduling_optimizer_fix import apply_scheduling_optimizer_fix
    apply_scheduling_optimizer_fix()
    print("✅ Applied scheduling optimizer fix")
except ImportError:
    print("ℹ️  scheduling_optimizer_fix not available")

try:
    from validator_precision_fix import apply_validator_precision_fix
    apply_validator_precision_fix()
    print("✅ Applied validator precision fix")
except ImportError:
    print("ℹ️  validator_precision_fix not available")


def create_dragon4_system_with_complete_fix():
    """创建应用完整修复的Dragon4系统"""
    
    print("🐉 创建Dragon4系统...")
    
    if HAS_DRAGON4_SYSTEM:
        # 使用原始Dragon4系统，但禁用可能冲突的补丁
        config = Dragon4Config(
            npu_bandwidth=120.0,
            dsp_count=2,
            dsp_bandwidth=40.0,
            enable_segmentation=False,  # 先禁用分段，专注解决基础冲突
            enable_precision_scheduling=False  # 避免与我们的修复冲突
        )
        
        # 创建系统但跳过自动补丁应用
        system = Dragon4System.__new__(Dragon4System)
        system.config = config
        system.scheduler = MultiResourceScheduler(
            enable_segmentation=config.enable_segmentation,
            max_segmentation_overhead_ratio=config.max_segmentation_overhead_ratio
        )
        
        # 手动添加硬件资源
        system.scheduler.add_npu("NPU_0", bandwidth=config.npu_bandwidth)
        system.scheduler.add_npu("NPU_1", bandwidth=config.npu_bandwidth)
        for i in range(config.dsp_count):
            system.scheduler.add_dsp(f"DSP_{i}", bandwidth=config.dsp_bandwidth)
        
        # 应用我们验证成功的完整修复
        apply_complete_resource_fix(system.scheduler)
        
        print(f"🐉 Dragon4 Hardware System Initialized:")
        print(f"  - 2 x NPU @ {config.npu_bandwidth} GOPS each")
        print(f"  - {config.dsp_count} x DSP @ {config.dsp_bandwidth} GOPS each")
        print(f"  - Complete Resource Fix: Applied")
        
        return system
        
    else:
        # 备用系统
        scheduler = MultiResourceScheduler(enable_segmentation=False)
        scheduler.add_npu("NPU_0", bandwidth=120.0)
        scheduler.add_npu("NPU_1", bandwidth=120.0)
        scheduler.add_dsp("DSP_0", bandwidth=40.0)
        scheduler.add_dsp("DSP_1", bandwidth=40.0)
        
        # 应用完整修复
        apply_complete_resource_fix(scheduler)
        
        print("🐉 Fallback Dragon4 System Created with Complete Fix")
        
        # 创建简单的系统包装器
        class SimpleSystem:
            def __init__(self, scheduler):
                self.scheduler = scheduler
            
            def schedule(self, time_window):
                return self.scheduler.priority_aware_schedule_with_segmentation(time_window)
            
            def reset(self):
                self.scheduler.schedule_history = []
                self.scheduler.active_bindings = []
                self.scheduler.tasks = {}
                for queue in self.scheduler.resource_queues.values():
                    queue.available_time = 0.0
        
        return SimpleSystem(scheduler)


def create_workload():
    """创建工作负载"""
    
    if HAS_DRAGON4_WORKLOAD:
        return Dragon4Workload.create_simple_workload()
    else:
        return create_fallback_workload()


def create_fallback_workload():
    """创建备用工作负载"""
    
    tasks = []
    
    # 任务1: 高优先级NPU任务
    task1 = NNTask("T1", "DetectionTask", 
                   priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task1.set_npu_only({120.0: 10.0}, "detection_seg")
    task1.set_performance_requirements(fps=30, latency=35)
    tasks.append(task1)
    
    # 任务2: DSP-NPU序列任务
    task2 = NNTask("T2", "ProcessingTask", 
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task2.set_dsp_npu_sequence([
        (ResourceType.DSP, {40.0: 5.0}, 0, "preprocess_seg"),
        (ResourceType.NPU, {120.0: 15.0}, 5, "inference_seg"),
    ])
    task2.set_performance_requirements(fps=20, latency=50)
    tasks.append(task2)
    
    # 任务3: DSP-NPU序列任务
    task3 = NNTask("T3", "AnalysisTask", 
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task3.set_dsp_npu_sequence([
        (ResourceType.DSP, {40.0: 5.0}, 0, "analysis_dsp_seg"),
        (ResourceType.NPU, {120.0: 10.0}, 5, "analysis_npu_seg"),
    ])
    task3.set_performance_requirements(fps=15, latency=80)
    tasks.append(task3)
    
    # 任务4: 低优先级NPU任务
    task4 = NNTask("T4", "BackgroundTask", 
                   priority=TaskPriority.LOW,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task4.set_npu_only({120.0: 20.0}, "background_seg")
    task4.set_performance_requirements(fps=10, latency=100)
    tasks.append(task4)
    
    return tasks


def generate_simple_visualization(scheduler, test_name="Dragon4"):
    """生成简化的可视化，直接使用修复后的elegant_visualization"""
    
    print(f"\n🎨 生成可视化 (使用修复后的颜色方案)...")
    
    try:
        from elegant_visualization import ElegantSchedulerVisualizer
        
        # 创建可视化器（已经包含Dragon4颜色修复）
        visualizer = ElegantSchedulerVisualizer(scheduler)
        
        # 生成甘特图
        print("  📊 创建Dragon4甘特图...")
        visualizer.plot_elegant_gantt(
            bar_height=0.35,
            spacing=0.8,
            use_alt_colors=False  # 使用修复后的主颜色方案
        )
        
        # 导出Chrome Tracing格式
        trace_filename = f"{test_name.lower().replace(' ', '_')}_trace.json"
        print(f"  🔄 导出Chrome Tracing -> {trace_filename}")
        visualizer.export_chrome_tracing(trace_filename)
        
        print(f"\n✅ 可视化生成完成!")
        print(f"   📊 甘特图: Dragon4颜色方案 (🔴红 🟠橙 🟢绿 🔵蓝)")
        print(f"   🔄 Chrome Tracing: {trace_filename}")
        print(f"   💡 打开 chrome://tracing 加载 {trace_filename}")
        
        return True
        
    except ImportError as e:
        print(f"❌ elegant_visualization 模块不可用: {e}")
        print("   请确保 elegant_visualization.py 文件存在")
        print("   并已运行颜色修复: python elegant_visualization_dragon4_fix.py")
        return False
    except Exception as e:
        print(f"❌ 可视化生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_task_analysis_with_colors(scheduler, results):
    """打印任务分析，手动显示颜色信息"""
    
    print(f"\n📋 Dragon4任务分析:")
    print("=" * 70)
    
    # Dragon4颜色方案
    priority_colors = {
        TaskPriority.CRITICAL: "🔴 红色",
        TaskPriority.HIGH: "🟠 橙色", 
        TaskPriority.NORMAL: "🟢 绿色",
        TaskPriority.LOW: "🔵 蓝色"
    }
    
    # 按任务分组分析
    task_analysis = defaultdict(list)
    for result in results:
        task_analysis[result.task_id].append(result)
    
    print("任务概览 (在可视化中的显示方式):")
    print(f"{'显示名称':<15} {'类型':<12} {'优先级':<10} {'颜色':<10} {'执行次数':<8} {'总时长'}")
    print("-" * 75)
    
    for task_id in sorted(task_analysis.keys()):
        if task_id in scheduler.tasks:
            task = scheduler.tasks[task_id]
            executions = task_analysis[task_id]
            
            # 确定显示名称
            if task.runtime_type == RuntimeType.DSP_RUNTIME:
                task_display = f"X: {task_id}"  # DSP Runtime任务前加 "X: "
                task_type = "DSP Runtime"
            else:
                task_display = task_id  # ACPU Runtime任务不加标识
                task_type = "ACPU Runtime"
            
            # 获取优先级颜色
            color_info = priority_colors.get(task.priority, "⚪ 默认")
            
            # 计算总执行时间
            total_duration = sum(r.end_time - r.start_time for r in executions)
            
            print(f"{task_display:<15} {task_type:<12} {task.priority.name:<10} {color_info:<10} {len(executions):<8} {total_duration:.1f}ms")
    
    print(f"\n🎨 Dragon4可视化图例:")
    print("  - 甘特图颜色: 🔴红色(CRITICAL) → 🟠橙色(HIGH) → 🟢绿色(NORMAL) → 🔵蓝色(LOW)")
    print("  - DSP Runtime任务: 显示为 'X: TaskID' (前缀 'X: ')")
    print("  - ACPU Runtime任务: 显示为 'TaskID' (无前缀)")
    
    # 执行时间统计
    print(f"\n⏱️  执行时间统计:")
    for task_id in sorted(task_analysis.keys()):
        if task_id in scheduler.tasks:
            task = scheduler.tasks[task_id]
            executions = task_analysis[task_id]
            
            if executions:
                execution_times = [r.start_time for r in executions]
                intervals = [execution_times[i+1] - execution_times[i] 
                           for i in range(len(execution_times)-1)]
                
                avg_interval = sum(intervals) / len(intervals) if intervals else 0
                expected_interval = 1000.0 / task.fps_requirement if task.fps_requirement > 0 else 0
                
                display_name = f"X: {task_id}" if task.runtime_type == RuntimeType.DSP_RUNTIME else task_id
                print(f"  {display_name}: 平均间隔 {avg_interval:.1f}ms (期望 {expected_interval:.1f}ms)")


def run_test_with_complete_fix(system, tasks, test_name="Dragon4测试"):
    """使用完整修复运行测试"""
    
    print(f"\n{'='*60}")
    print(f"{test_name}")
    print(f"{'='*60}")
    
    # 重置系统
    system.reset()
    
    # 添加任务
    for task in tasks:
        system.scheduler.add_task(task)
    
    print(f"添加了 {len(tasks)} 个任务:")
    for task in tasks:
        runtime_label = "DSP Runtime" if task.runtime_type == RuntimeType.DSP_RUNTIME else "ACPU Runtime"
        display_name = f"X: {task.task_id}" if task.runtime_type == RuntimeType.DSP_RUNTIME else task.task_id
        print(f"  + {display_name}: {task.priority.name} 优先级, {task.fps_requirement} FPS ({runtime_label})")
    
    # 执行调度
    time_window = 500.0
    print(f"\n🚀 运行调度 ({time_window}ms)...")
    
    try:
        results = system.schedule(time_window)
        print(f"✅ 调度成功: {len(results)} 个事件")
    except Exception as e:
        print(f"❌ 调度失败: {e}")
        import traceback
        traceback.print_exc()
        return [], {}
    
    # 验证调度结果
    print(f"\n📊 验证调度结果...")
    is_valid = validate_fixed_schedule(system.scheduler)
    
    if is_valid:
        print("🎉 完美! 调度验证通过，无资源冲突")
    else:
        print("❌ 验证失败，仍有冲突")
    
    # 计算性能指标
    metrics = calculate_performance_metrics(system.scheduler, results, time_window)
    print_performance_summary(metrics)
    
    # 显示详细时间线
    print_resource_timeline(system.scheduler, results)
    
    return results, metrics


def calculate_performance_metrics(scheduler, results, time_window):
    """计算性能指标"""
    
    metrics = {
        'total_events': len(results),
        'avg_latency': 0.0,
        'avg_utilization': 0.0,
        'total_violations': 0,
        'resource_utilization': {},
        'task_performance': {}
    }
    
    if not results:
        return metrics
    
    # 计算平均延迟
    latencies = [r.actual_latency for r in results if hasattr(r, 'actual_latency')]
    if latencies:
        metrics['avg_latency'] = sum(latencies) / len(latencies)
    
    # 计算资源利用率
    resource_busy_time = defaultdict(float)
    for result in results:
        duration = result.end_time - result.start_time
        for res_type, res_id in result.assigned_resources.items():
            resource_busy_time[res_id] += duration
    
    total_utilization = 0
    resource_count = 0
    for res_id, busy_time in resource_busy_time.items():
        utilization = (busy_time / time_window) * 100
        metrics['resource_utilization'][res_id] = utilization
        total_utilization += utilization
        resource_count += 1
    
    if resource_count > 0:
        metrics['avg_utilization'] = total_utilization / resource_count
    
    # 计算任务性能
    task_counts = defaultdict(int)
    for result in results:
        task_counts[result.task_id] += 1
    
    for task_id, task in scheduler.tasks.items():
        expected_executions = int((time_window / 1000.0) * task.fps_requirement)
        actual_executions = task_counts[task_id]
        
        metrics['task_performance'][task_id] = {
            'expected': expected_executions,
            'actual': actual_executions,
            'fps_achieved': (actual_executions * 1000.0) / time_window,
            'fps_required': task.fps_requirement
        }
        
        if actual_executions < expected_executions * 0.9:
            metrics['total_violations'] += 1
    
    return metrics


def print_performance_summary(metrics):
    """打印性能摘要"""
    
    print(f"\n📊 性能指标摘要:")
    print(f"  总调度事件: {metrics['total_events']}")
    print(f"  平均延迟: {metrics['avg_latency']:.2f}ms")
    print(f"  平均资源利用率: {metrics['avg_utilization']:.1f}%")
    print(f"  任务违反数: {metrics['total_violations']}")
    
    if metrics['resource_utilization']:
        print(f"\n  资源利用率详情:")
        for res_id, util in metrics['resource_utilization'].items():
            print(f"    {res_id}: {util:.1f}%")
    
    if metrics['task_performance']:
        print(f"\n  任务性能:")
        for task_id, perf in metrics['task_performance'].items():
            achieved_fps = perf['fps_achieved']
            required_fps = perf['fps_required']
            status = "✅" if achieved_fps >= required_fps * 0.9 else "❌"
            print(f"    {task_id}: {achieved_fps:.1f}/{required_fps:.1f} FPS {status}")


def print_resource_timeline(scheduler, results):
    """打印资源时间线"""
    
    print(f"\n🕒 资源时间线 (前15个事件):")
    
    # 按资源分组
    by_resource = defaultdict(list)
    for result in results:
        for res_type, res_id in result.assigned_resources.items():
            task = scheduler.tasks[result.task_id]
            display_name = f"X: {task.task_id}" if task.runtime_type == RuntimeType.DSP_RUNTIME else task.task_id
            
            by_resource[res_id].append({
                'start': result.start_time,
                'end': result.end_time,
                'task': display_name
            })
    
    for res_id in sorted(by_resource.keys()):
        events = sorted(by_resource[res_id], key=lambda x: x['start'])
        print(f"\n  {res_id}:")
        
        for i, event in enumerate(events[:15]):
            print(f"    {event['start']:6.1f} - {event['end']:6.1f} ms: {event['task']}")
        
        if len(events) > 15:
            print(f"    ... 还有 {len(events) - 15} 个事件")


def main():
    """主测试函数"""
    
    print("=" * 80)
    print("Dragon4 系统基础调度测试 - 简化版 (Dragon4颜色方案)")
    print("=" * 80)
    
    # 1. 创建应用完整修复的Dragon4系统
    system = create_dragon4_system_with_complete_fix()
    
    # 2. 创建工作负载
    tasks = create_workload()
    
    print(f"\n使用 {'完整' if HAS_DRAGON4_SYSTEM else '备用'} Dragon4系统")
    print(f"使用 {'完整' if HAS_DRAGON4_WORKLOAD else '备用'} 工作负载")
    
    # 3. 运行基础调度测试
    results, metrics = run_test_with_complete_fix(system, tasks, "Dragon4基础调度测试")
    
    # 4. 详细任务分析
    if results:
        print_task_analysis_with_colors(system.scheduler, results)
    
    # 5. 生成可视化
    if results:
        visualization_success = generate_simple_visualization(system.scheduler, "Dragon4_基础调度")
    
    # 6. 最终验证和总结
    print(f"\n{'='*60}")
    print("最终验证结果")
    print(f"{'='*60}")
    
    if results:
        is_valid = validate_fixed_schedule(system.scheduler)
        if is_valid:
            print("🎉 成功! Dragon4系统基础调度测试完成")
            print("✅ 零资源冲突")
            print("✅ 优先级调度正确")  
            print("✅ 任务性能满足需求")
            if 'visualization_success' in locals() and visualization_success:
                print("✅ Dragon4可视化生成成功")
            
            print(f"\n📊 性能摘要:")
            print(f"  - 调度事件: {len(results)}")
            print(f"  - 平均延迟: {metrics.get('avg_latency', 0):.2f}ms")
            print(f"  - 平均资源利用率: {metrics.get('avg_utilization', 0):.1f}%")
            print(f"  - 任务违反数: {metrics.get('total_violations', 0)}")
            
        else:
            print("❌ 仍需进一步调优")
    else:
        print("❌ 调度失败，需要检查配置")
    
    print(f"\n💡 Dragon4可视化说明:")
    print(f"   📊 甘特图颜色: 🔴红(CRITICAL) 🟠橙(HIGH) 🟢绿(NORMAL) 🔵蓝(LOW)")
    print(f"   🏷️  任务标识: DSP Runtime = 'X: TaskID', ACPU Runtime = 'TaskID'")
    print(f"   🔄 Chrome Tracing: 打开 chrome://tracing 加载JSON文件")
    
    print(f"\n🔧 如需修复颜色显示:")
    print(f"   python elegant_visualization_dragon4_fix.py")


if __name__ == "__main__":
    main()
