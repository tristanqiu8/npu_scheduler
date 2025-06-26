#!/usr/bin/env python3
"""
Dragon4分段功能测试 - 基于dragon4_test_simplified.py
测试启用分段后的调度功能
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


def create_dragon4_system_with_segmentation():
    """创建启用分段功能的Dragon4系统"""
    
    print("🐉 创建启用分段的Dragon4系统...")
    
    if HAS_DRAGON4_SYSTEM:
        # 使用原始Dragon4系统，启用分段
        config = Dragon4Config(
            npu_bandwidth=120.0,
            dsp_count=2,
            dsp_bandwidth=40.0,
            enable_segmentation=True,  # 启用分段
            max_segmentation_overhead_ratio=0.15,  # 最大15%的分段开销
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
        
        print(f"🐉 Dragon4 Hardware System Initialized with Segmentation:")
        print(f"  - 2 x NPU @ {config.npu_bandwidth} GOPS each")
        print(f"  - {config.dsp_count} x DSP @ {config.dsp_bandwidth} GOPS each")
        print(f"  - Segmentation: ENABLED (max overhead: {config.max_segmentation_overhead_ratio * 100}%)")
        print(f"  - Complete Resource Fix: Applied")
        
        return system
        
    else:
        # 备用系统
        scheduler = MultiResourceScheduler(
            enable_segmentation=True,
            max_segmentation_overhead_ratio=0.15
        )
        scheduler.add_npu("NPU_0", bandwidth=120.0)
        scheduler.add_npu("NPU_1", bandwidth=120.0)
        scheduler.add_dsp("DSP_0", bandwidth=40.0)
        scheduler.add_dsp("DSP_1", bandwidth=40.0)
        
        # 应用完整修复
        apply_complete_resource_fix(scheduler)
        
        print("🐉 Fallback Dragon4 System Created with Segmentation Enabled")
        
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


def create_segmentation_workload():
    """创建包含分段策略的工作负载"""
    
    tasks = []
    
    # 任务1: 高优先级NPU任务，强制分段
    task1 = NNTask("T1", "DetectionTask_Segmented", 
                   priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION)
    task1.set_npu_only({120.0: 10.0}, "detection_seg")
    # 添加切分点
    task1.add_cut_points_to_segment("detection_seg", [
        ("cut1", 0.3, 0.05),  # 30%处切分，5%开销
        ("cut2", 0.7, 0.05)   # 70%处切分，5%开销
    ])
    # 设置预设切分配置
    task1.set_preset_cut_configurations("detection_seg", [
        [],                    # Config 0: 无切分
        ["cut1"],             # Config 1: 仅在30%处切分
        ["cut2"],             # Config 2: 仅在70%处切分
        ["cut1", "cut2"]      # Config 3: 两处都切分
    ])
    task1.set_performance_requirements(fps=30, latency=35)
    tasks.append(task1)
    
    # 任务2: DSP-NPU序列任务，自适应分段
    task2 = NNTask("T2", "ProcessingTask_Adaptive", 
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION)
    task2.set_dsp_npu_sequence([
        (ResourceType.DSP, {40.0: 5.0}, 0, "preprocess_seg"),
        (ResourceType.NPU, {120.0: 15.0}, 5, "inference_seg"),
    ])
    # NPU段添加切分点
    task2.add_cut_points_to_segment("inference_seg", [
        ("npu_cut1", 0.5, 0.1)  # 50%处切分，10%开销
    ])
    task2.set_preset_cut_configurations("inference_seg", [
        [],           # Config 0: 无切分
        ["npu_cut1"]  # Config 1: 中间切分
    ])
    task2.set_performance_requirements(fps=20, latency=50)
    tasks.append(task2)
    
    # 任务3: DSP-NPU序列任务，不分段
    task3 = NNTask("T3", "AnalysisTask_NoSeg", 
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task3.set_dsp_npu_sequence([
        (ResourceType.DSP, {40.0: 5.0}, 0, "analysis_dsp_seg"),
        (ResourceType.NPU, {120.0: 10.0}, 5, "analysis_npu_seg"),
    ])
    task3.set_performance_requirements(fps=15, latency=80)
    tasks.append(task3)
    
    # 任务4: 低优先级NPU任务，强制分段
    task4 = NNTask("T4", "BackgroundTask_Forced", 
                   priority=TaskPriority.LOW,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION)
    task4.set_npu_only({120.0: 20.0}, "background_seg")
    # 添加多个切分点
    task4.add_cut_points_to_segment("background_seg", [
        ("bg_cut1", 0.25, 0.08),
        ("bg_cut2", 0.5, 0.08),
        ("bg_cut3", 0.75, 0.08)
    ])
    task4.set_preset_cut_configurations("background_seg", [
        [],                              # Config 0: 无切分
        ["bg_cut2"],                     # Config 1: 仅中间切分
        ["bg_cut1", "bg_cut3"],         # Config 2: 两端切分
        ["bg_cut1", "bg_cut2", "bg_cut3"] # Config 3: 全部切分
    ])
    task4.set_performance_requirements(fps=10, latency=100)
    tasks.append(task4)
    
    # 任务5: 关键任务，自适应分段
    task5 = NNTask("T5", "CriticalTask_Adaptive",
                   priority=TaskPriority.CRITICAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION)
    task5.set_npu_only({120.0: 8.0}, "critical_seg")
    task5.add_cut_points_to_segment("critical_seg", [
        ("critical_cut", 0.6, 0.03)  # 60%处切分，3%开销
    ])
    task5.set_preset_cut_configurations("critical_seg", [
        [],
        ["critical_cut"]
    ])
    task5.set_performance_requirements(fps=60, latency=20)
    tasks.append(task5)
    
    return tasks


def analyze_segmentation_results(scheduler, results):
    """分析分段执行结果"""
    
    print(f"\n🔍 分段执行分析:")
    print("=" * 80)
    
    # 统计分段情况
    segmentation_stats = defaultdict(lambda: {
        'total_executions': 0,
        'segmented_executions': 0,
        'total_segments': 0,
        'overhead_time': 0.0,
        'strategies': defaultdict(int)
    })
    
    for result in results:
        task_id = result.task_id
        task = scheduler.tasks.get(task_id)
        if not task:
            continue
            
        stats = segmentation_stats[task_id]
        stats['total_executions'] += 1
        stats['strategies'][task.segmentation_strategy.name] += 1
        
        # 检查是否有分段
        if hasattr(result, 'segments') and result.segments:
            stats['segmented_executions'] += 1
            stats['total_segments'] += len(result.segments)
            
            # 计算分段开销
            if hasattr(result, 'segmentation_overhead'):
                stats['overhead_time'] += result.segmentation_overhead
    
    # 打印分段统计
    print(f"{'任务ID':<10} {'策略':<25} {'执行次数':<10} {'分段次数':<10} {'平均段数':<10} {'总开销(ms)':<12}")
    print("-" * 90)
    
    for task_id in sorted(segmentation_stats.keys()):
        stats = segmentation_stats[task_id]
        task = scheduler.tasks.get(task_id)
        
        if stats['segmented_executions'] > 0:
            avg_segments = stats['total_segments'] / stats['segmented_executions']
        else:
            avg_segments = 0
        
        strategy = task.segmentation_strategy.name if task else "UNKNOWN"
        
        print(f"{task_id:<10} {strategy:<25} {stats['total_executions']:<10} "
              f"{stats['segmented_executions']:<10} {avg_segments:<10.1f} "
              f"{stats['overhead_time']:<12.2f}")
    
    # 分段效果分析
    print(f"\n📊 分段效果分析:")
    total_tasks = len(segmentation_stats)
    segmented_tasks = sum(1 for stats in segmentation_stats.values() 
                         if stats['segmented_executions'] > 0)
    
    print(f"  - 总任务数: {total_tasks}")
    print(f"  - 实际分段的任务数: {segmented_tasks}")
    print(f"  - 分段比例: {segmented_tasks/total_tasks*100:.1f}%")
    
    # 按策略统计
    strategy_counts = defaultdict(int)
    for task in scheduler.tasks.values():
        strategy_counts[task.segmentation_strategy.name] += 1
    
    print(f"\n  策略分布:")
    for strategy, count in sorted(strategy_counts.items()):
        print(f"    - {strategy}: {count} 任务")


def print_segmentation_timeline(scheduler, results, max_events=20):
    """打印分段执行时间线"""
    
    print(f"\n⏱️  分段执行时间线 (前{max_events}个事件):")
    print("=" * 100)
    
    event_count = 0
    for result in results[:max_events]:
        task = scheduler.tasks.get(result.task_id)
        if not task:
            continue
        
        # 任务基本信息
        display_name = f"X: {task.task_id}" if task.runtime_type == RuntimeType.DSP_RUNTIME else task.task_id
        
        # 检查是否分段
        if hasattr(result, 'segments') and result.segments:
            seg_info = f"[分{len(result.segments)}段]"
        else:
            seg_info = "[未分段]"
        
        # 打印事件信息
        print(f"\n{event_count + 1:3d}. {display_name} {seg_info} @ {result.start_time:.1f}-{result.end_time:.1f}ms")
        
        # 如果有分段详情，打印每个子段
        if hasattr(result, 'segments') and result.segments:
            for i, seg in enumerate(result.segments):
                print(f"     段{i+1}: {seg.start_time:.1f}-{seg.end_time:.1f}ms "
                      f"在 {seg.resource_id} 上执行")
        
        event_count += 1


def main():
    """主测试函数"""
    
    print("=" * 80)
    print("Dragon4 分段功能测试")
    print("=" * 80)
    
    # 1. 创建启用分段的Dragon4系统
    system = create_dragon4_system_with_segmentation()
    
    # 2. 创建包含分段策略的工作负载
    tasks = create_segmentation_workload()
    
    print(f"\n📋 工作负载配置:")
    for task in tasks:
        seg_strategy = task.segmentation_strategy.name.replace('_SEGMENTATION', '')
        runtime_label = "DSP Runtime" if task.runtime_type == RuntimeType.DSP_RUNTIME else "ACPU Runtime"
        display_name = f"X: {task.task_id}" if task.runtime_type == RuntimeType.DSP_RUNTIME else task.task_id
        
        print(f"  + {display_name}: {task.priority.name} 优先级, {seg_strategy} 分段策略, "
              f"{task.fps_requirement} FPS ({runtime_label})")
        
        # 打印切分点信息（检查是否有切分点）
        if hasattr(task, 'segments') and task.segments:
            for segment in task.segments:
                if hasattr(segment, 'cut_points') and segment.cut_points:
                    print(f"    - 段 {segment.segment_id}: {len(segment.cut_points)} 个切分点")
    
    # 3. 运行分段调度测试
    print(f"\n🚀 运行分段调度测试...")
    time_window = 500.0
    
    # 重置系统
    system.reset()
    
    # 添加任务
    for task in tasks:
        system.scheduler.add_task(task)
    
    try:
        results = system.schedule(time_window)
        print(f"✅ 调度成功: {len(results)} 个事件")
    except Exception as e:
        print(f"❌ 调度失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 验证调度结果
    print(f"\n📊 验证调度结果...")
    is_valid = validate_fixed_schedule(system.scheduler)
    
    if is_valid:
        print("🎉 调度验证通过，无资源冲突")
    else:
        print("❌ 验证失败，存在资源冲突")
    
    # 5. 分析分段执行情况
    analyze_segmentation_results(system.scheduler, results)
    
    # 6. 打印分段时间线
    print_segmentation_timeline(system.scheduler, results)
    
    # 7. 计算性能指标
    from dragon4_test_simplified import calculate_performance_metrics, print_performance_summary
    metrics = calculate_performance_metrics(system.scheduler, results, time_window)
    print_performance_summary(metrics)
    
    # 8. 生成可视化
    print(f"\n🎨 生成分段可视化...")
    try:
        from dragon4_test_simplified import generate_simple_visualization
        visualization_success = generate_simple_visualization(system.scheduler, "Dragon4_分段调度")
    except Exception as e:
        print(f"❌ 可视化生成失败: {e}")
        visualization_success = False
    
    # 9. 最终总结
    print(f"\n{'='*60}")
    print("分段功能测试总结")
    print(f"{'='*60}")
    
    if is_valid:
        print("✅ 分段功能基本正常工作")
        print("✅ 资源调度无冲突")
        if 'segmented_tasks' in locals() and segmented_tasks > 0:
            print(f"✅ {segmented_tasks}/{total_tasks} 个任务成功执行分段")
        if visualization_success:
            print("✅ 分段可视化生成成功")
    else:
        print("⚠️  分段功能存在问题，需要进一步调试")
    
    print(f"\n💡 分段功能说明:")
    print(f"   - FORCED_SEGMENTATION: 强制分段")
    print(f"   - ADAPTIVE_SEGMENTATION: 自适应分段")
    print(f"   - NO_SEGMENTATION: 禁用分段")
    print(f"   - 分段开销会影响总执行时间")


if __name__ == "__main__":
    main()
