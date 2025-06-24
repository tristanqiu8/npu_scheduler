#!/usr/bin/env python3
"""
Dragon4 Workload Definition
定义适用于Dragon4系统的任务集
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from task import NNTask


@dataclass
class WorkloadConfig:
    """工作负载配置"""
    name: str = "default"
    enable_segmentation: bool = True
    complexity_level: str = "medium"  # low, medium, high


class Dragon4Workload:
    """Dragon4系统工作负载定义"""
    
    @staticmethod
    def create_automotive_workload(config: Optional[WorkloadConfig] = None) -> List[NNTask]:
        """创建汽车场景工作负载"""
        config = config or WorkloadConfig(name="automotive", enable_segmentation=True)
        tasks = []
        
        # 1. 安全监控任务 - 最高优先级
        task1 = NNTask("T1", "SafetyMonitor", 
                       priority=TaskPriority.CRITICAL,
                       runtime_type=RuntimeType.DSP_RUNTIME,
                       segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION if config.enable_segmentation else SegmentationStrategy.NO_SEGMENTATION)
        
        task1.set_dsp_npu_sequence([
            (ResourceType.DSP, {40.0: 5.0, 80.0: 3.0}, 0, "safety_preprocess"),
            (ResourceType.NPU, {60.0: 20.0, 120.0: 10.0, 240.0: 5.0}, 5, "safety_detect"),
            (ResourceType.DSP, {40.0: 2.0}, 15, "safety_postprocess")
        ])
        
        if config.enable_segmentation:
            task1.add_cut_points_to_segment("safety_detect", [
                ("layer1", 0.3, 0.1),
                ("layer2", 0.7, 0.1)
            ])
            task1.set_preset_cut_configurations("safety_detect", [
                [],                    # Config 0: 无切分
                ["layer1"],           # Config 1: 前切
                ["layer2"],           # Config 2: 后切
                ["layer1", "layer2"]  # Config 3: 全切
            ])
        
        task1.set_performance_requirements(fps=30, latency=35)
        tasks.append(task1)
        
        # 2. 目标检测任务 - 高优先级
        task2 = NNTask("T2", "ObjectDetection",
                       priority=TaskPriority.HIGH,
                       runtime_type=RuntimeType.ACPU_RUNTIME,
                       segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION if config.enable_segmentation else SegmentationStrategy.NO_SEGMENTATION)
        
        task2.set_npu_only({60.0: 40.0, 120.0: 20.0, 240.0: 10.0}, "detection_backbone")
        
        if config.enable_segmentation:
            task2.add_cut_points_to_segment("detection_backbone", [
                ("backbone_1", 0.25, 0.12),
                ("backbone_2", 0.5, 0.11),
                ("backbone_3", 0.75, 0.13)
            ])
            task2.set_preset_cut_configurations("detection_backbone", [
                [],                                      # Config 0: 无切分
                ["backbone_2"],                          # Config 1: 中间切
                ["backbone_1", "backbone_3"],            # Config 2: 边缘切
                ["backbone_1", "backbone_2", "backbone_3"] # Config 3: 全切
            ])
        
        task2.set_performance_requirements(fps=20, latency=50)
        tasks.append(task2)
        
        # 3. 车道检测任务 - 普通优先级
        task3 = NNTask("T3", "LaneDetection",
                       priority=TaskPriority.NORMAL,
                       runtime_type=RuntimeType.ACPU_RUNTIME,
                       segmentation_strategy=SegmentationStrategy.CUSTOM_SEGMENTATION if config.enable_segmentation else SegmentationStrategy.NO_SEGMENTATION)
        
        task3.set_npu_only({60.0: 30.0, 120.0: 15.0, 240.0: 8.0}, "lane_seg")
        
        if config.enable_segmentation:
            task3.add_cut_points_to_segment("lane_seg", [
                ("edge_detect", 0.4, 0.1),
                ("line_fit", 0.8, 0.1)
            ])
            task3.set_preset_cut_configurations("lane_seg", [
                [],                       # Config 0: 无切分
                ["edge_detect"],          # Config 1: 边缘检测后切
                ["line_fit"],             # Config 2: 线拟合前切
                ["edge_detect", "line_fit"] # Config 3: 全切
            ])
            task3.select_cut_configuration("lane_seg", 0)  # 默认不切分
        
        task3.set_performance_requirements(fps=15, latency=70)
        task3.add_dependency("T1")  # 依赖安全监控
        tasks.append(task3)
        
        # 4. 语义分割任务 - 低优先级但计算密集
        task4 = NNTask("T4", "SemanticSegmentation",
                       priority=TaskPriority.LOW,
                       runtime_type=RuntimeType.DSP_RUNTIME,
                       segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION if config.enable_segmentation else SegmentationStrategy.NO_SEGMENTATION)
        
        task4.set_dsp_npu_sequence([
            (ResourceType.DSP, {40.0: 8.0}, 0, "seg_preprocess"),
            (ResourceType.NPU, {60.0: 60.0, 120.0: 30.0, 240.0: 15.0}, 8, "seg_backbone"),
            (ResourceType.NPU, {60.0: 20.0, 120.0: 10.0}, 38, "seg_decoder"),
            (ResourceType.DSP, {40.0: 5.0}, 48, "seg_postprocess")
        ])
        
        if config.enable_segmentation:
            # 主干网络切分点
            task4.add_cut_points_to_segment("seg_backbone", [
                ("encoder1", 0.2, 0.15),
                ("encoder2", 0.4, 0.14),
                ("encoder3", 0.6, 0.13),
                ("encoder4", 0.8, 0.12)
            ])
            
            # 解码器切分点
            task4.add_cut_points_to_segment("seg_decoder", [
                ("decoder1", 0.5, 0.1)
            ])
        
        task4.set_performance_requirements(fps=5, latency=200)
        tasks.append(task4)
        
        # 5. 后台分析任务
        task5 = NNTask("T5", "Analytics",
                       priority=TaskPriority.LOW,
                       runtime_type=RuntimeType.ACPU_RUNTIME,
                       segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
        
        task5.set_npu_only({60.0: 25.0, 120.0: 12.5}, "analytics_seg")
        task5.set_performance_requirements(fps=2, latency=500)
        tasks.append(task5)
        
        return tasks
    
    @staticmethod
    def create_simple_workload() -> List[NNTask]:
        """创建简单工作负载（用于基准测试）"""
        tasks = []
        
        # 简单任务1 - 高优先级
        task1 = NNTask("T1", "SimpleHigh",
                       priority=TaskPriority.HIGH,
                       runtime_type=RuntimeType.ACPU_RUNTIME,
                       segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
        task1.set_npu_only({120.0: 10.0}, "simple_high")
        task1.set_performance_requirements(fps=30, latency=35)
        tasks.append(task1)
        
        # 简单任务2 - 中优先级
        task2 = NNTask("T2", "SimpleMedium",
                       priority=TaskPriority.NORMAL,
                       runtime_type=RuntimeType.ACPU_RUNTIME,
                       segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
        task2.set_npu_only({120.0: 15.0}, "simple_medium")
        task2.set_performance_requirements(fps=20, latency=50)
        tasks.append(task2)
        
        # 简单任务3 - DSP+NPU
        task3 = NNTask("T3", "SimpleMixed",
                       priority=TaskPriority.NORMAL,
                       runtime_type=RuntimeType.DSP_RUNTIME,
                       segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
        task3.set_dsp_npu_sequence([
            (ResourceType.DSP, {40.0: 5.0}, 0, "simple_dsp"),
            (ResourceType.NPU, {120.0: 10.0}, 5, "simple_npu")
        ])
        task3.set_performance_requirements(fps=15, latency=70)
        tasks.append(task3)
        
        # 简单任务4 - 低优先级
        task4 = NNTask("T4", "SimpleLow",
                       priority=TaskPriority.LOW,
                       runtime_type=RuntimeType.ACPU_RUNTIME,
                       segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
        task4.set_npu_only({120.0: 20.0}, "simple_low")
        task4.set_performance_requirements(fps=10, latency=100)
        tasks.append(task4)
        
        return tasks
    
    @staticmethod
    def create_stress_workload(base_count: int = 10) -> List[NNTask]:
        """创建压力测试工作负载"""
        tasks = []
        
        for i in range(base_count):
            # 交替优先级
            if i < base_count // 3:
                priority = TaskPriority.HIGH
            elif i < 2 * base_count // 3:
                priority = TaskPriority.NORMAL
            else:
                priority = TaskPriority.LOW
            
            # 交替运行时类型
            runtime = RuntimeType.DSP_RUNTIME if i % 3 == 0 else RuntimeType.ACPU_RUNTIME
            
            task = NNTask(f"T{i+1}", f"StressTask_{i+1}",
                         priority=priority,
                         runtime_type=runtime,
                         segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION)
            
            # 计算负载随任务编号变化
            base_load = 10 + (i % 5) * 5
            
            if runtime == RuntimeType.DSP_RUNTIME:
                task.set_dsp_npu_sequence([
                    (ResourceType.DSP, {40.0: base_load * 0.3}, 0, f"stress_dsp_{i}"),
                    (ResourceType.NPU, {120.0: base_load}, base_load * 0.3, f"stress_npu_{i}")
                ])
            else:
                task.set_npu_only({120.0: base_load}, f"stress_seg_{i}")
            
            # 为一些任务添加切分点
            if i % 2 == 0:
                seg_id = f"stress_npu_{i}" if runtime == RuntimeType.DSP_RUNTIME else f"stress_seg_{i}"
                task.add_cut_points_to_segment(seg_id, [
                    ("cut1", 0.5, 0.1)
                ])
                task.set_preset_cut_configurations(seg_id, [
                    [],        # Config 0: 无切分
                    ["cut1"]   # Config 1: 中间切分
                ])
            
            # 设置性能要求
            fps = 30 - i * 2
            latency = 50 + i * 10
            task.set_performance_requirements(fps=max(fps, 5), latency=min(latency, 200))
            
            # 添加一些依赖关系
            if i > 0 and i % 3 == 0:
                task.add_dependency(f"T{i}")  # 依赖前一个任务
            
            tasks.append(task)
        
        return tasks
    
    @staticmethod
    def print_workload_summary(tasks: List[NNTask], name: str = "Workload"):
        """打印工作负载摘要"""
        print(f"\n=== {name} Summary ===")
        print(f"Total Tasks: {len(tasks)}")
        
        # 按优先级统计
        priority_count = {}
        for task in tasks:
            priority = task.priority.name
            priority_count[priority] = priority_count.get(priority, 0) + 1
        
        print("\nPriority Distribution:")
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, 
                        TaskPriority.NORMAL, TaskPriority.LOW]:
            count = priority_count.get(priority.name, 0)
            if count > 0:
                print(f"  {priority.name}: {count} tasks")
        
        # 按运行时类型统计
        runtime_count = {}
        for task in tasks:
            runtime = task.runtime_type.value
            runtime_count[runtime] = runtime_count.get(runtime, 0) + 1
        
        print("\nRuntime Type Distribution:")
        for runtime, count in runtime_count.items():
            print(f"  {runtime}: {count} tasks")
        
        # 分段策略统计
        seg_count = {}
        for task in tasks:
            strategy = task.segmentation_strategy.value
            seg_count[strategy] = seg_count.get(strategy, 0) + 1
        
        print("\nSegmentation Strategy Distribution:")
        for strategy, count in seg_count.items():
            print(f"  {strategy}: {count} tasks")
        
        # 任务详情
        print("\nTask Details:")
        for task in tasks:
            deps = f", deps={list(task.dependencies)}" if task.dependencies else ""
            print(f"  {task.task_id}: {task.name} "
                  f"[{task.priority.name}, {task.runtime_type.value}] "
                  f"FPS={task.fps_requirement}, Latency={task.latency_requirement}ms{deps}")


if __name__ == "__main__":
    # 测试工作负载定义
    print("=== Dragon4 Workload Test ===")
    
    # 1. 简单工作负载
    print("\n1. Simple Workload:")
    simple_tasks = Dragon4Workload.create_simple_workload()
    Dragon4Workload.print_workload_summary(simple_tasks, "Simple Workload")
    
    # 2. 汽车工作负载（无分段）
    print("\n\n2. Automotive Workload (No Segmentation):")
    auto_config_no_seg = WorkloadConfig(
        name="automotive_no_seg",
        enable_segmentation=False
    )
    auto_tasks_no_seg = Dragon4Workload.create_automotive_workload(auto_config_no_seg)
    Dragon4Workload.print_workload_summary(auto_tasks_no_seg, "Automotive (No Seg)")
    
    # 3. 汽车工作负载（有分段）
    print("\n\n3. Automotive Workload (With Segmentation):")
    auto_config_seg = WorkloadConfig(
        name="automotive_seg",
        enable_segmentation=True
    )
    auto_tasks_seg = Dragon4Workload.create_automotive_workload(auto_config_seg)
    Dragon4Workload.print_workload_summary(auto_tasks_seg, "Automotive (Segmented)")
    
    # 4. 压力测试工作负载
    print("\n\n4. Stress Test Workload:")
    stress_tasks = Dragon4Workload.create_stress_workload(8)
    Dragon4Workload.print_workload_summary(stress_tasks, "Stress Test")
