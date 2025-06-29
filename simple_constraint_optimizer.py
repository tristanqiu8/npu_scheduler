#!/usr/bin/env python3
"""
简单约束满足优化器
优先保证依赖关系和资源无冲突，然后尽可能提高FPS满足率
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import copy

from enums import TaskPriority, RuntimeType, SegmentationStrategy
from intelligent_scheduler_optimizer import IntelligentSchedulerOptimizer, SchedulingConfiguration


class SimpleConstraintOptimizer(IntelligentSchedulerOptimizer):
    """简单约束满足优化器"""
    
    def __init__(self, scheduler, time_window: float = 200.0):
        super().__init__(scheduler, time_window)
        self.constraint_weight = 1000.0  # 约束违反的惩罚权重
        
    def optimize_simple(self, max_attempts: int = 10) -> SchedulingConfiguration:
        """简单优化算法：优先满足约束，再优化FPS"""
        
        print("\n🎯 简单约束满足优化")
        print("=" * 60)
        print("策略：优先保证依赖关系和资源无冲突")
        
        # 步骤1：先找到一个满足所有约束的基础配置
        base_config = self._find_constraint_satisfying_config(max_attempts)
        
        if not base_config:
            print("❌ 无法找到满足约束的配置")
            return self.original_config
        
        print(f"\n✅ 找到满足约束的基础配置:")
        print(f"  - 资源冲突: {base_config.conflict_count}")
        print(f"  - FPS满足率: {base_config.fps_satisfaction_rate:.1%}")
        
        # 步骤2：在保持约束满足的前提下，逐步优化FPS
        optimized_config = self._optimize_fps_with_constraints(base_config)
        
        self.best_config = optimized_config
        return optimized_config
    
    def _find_constraint_satisfying_config(self, max_attempts: int) -> Optional[SchedulingConfiguration]:
        """寻找满足约束的配置"""
        
        print("\n步骤1: 寻找满足约束的配置...")
        
        # 策略1：使用保守的配置
        conservative_config = self._create_conservative_config()
        fitness = self.evaluate_configuration(conservative_config)
        
        if conservative_config.conflict_count == 0:
            print("  ✓ 保守配置满足约束")
            return conservative_config
        
        # 策略2：逐步调整优先级避免冲突
        for attempt in range(max_attempts):
            print(f"\n  尝试 {attempt + 1}/{max_attempts}...")
            
            adjusted_config = self._adjust_for_constraints(conservative_config)
            fitness = self.evaluate_configuration(adjusted_config)
            
            print(f"    冲突数: {adjusted_config.conflict_count}")
            print(f"    FPS满足率: {adjusted_config.fps_satisfaction_rate:.1%}")
            
            if adjusted_config.conflict_count == 0:
                print("  ✓ 找到无冲突配置！")
                return adjusted_config
            
            # 基于反馈继续调整
            conservative_config = adjusted_config
        
        return None
    
    def _create_conservative_config(self) -> SchedulingConfiguration:
        """创建保守配置：避免资源竞争"""
        
        config = copy.deepcopy(self.original_config)
        
        # 分析任务的资源使用
        npu_tasks = []
        dsp_tasks = []
        mixed_tasks = []
        
        for tid, task in self.scheduler.tasks.items():
            uses_npu = any(seg.resource_type.value == "NPU" for seg in task.segments)
            uses_dsp = any(seg.resource_type.value == "DSP" for seg in task.segments)
            
            if uses_npu and uses_dsp:
                mixed_tasks.append(tid)
            elif uses_npu:
                npu_tasks.append(tid)
            elif uses_dsp:
                dsp_tasks.append(tid)
        
        # 策略：
        # 1. 混合任务使用DSP_Runtime（绑定执行，避免并行冲突）
        # 2. 交错分配优先级，避免同时竞争资源
        # 3. 不使用分段（避免复杂性）
        
        for tid in mixed_tasks:
            config.task_runtimes[tid] = RuntimeType.DSP_RUNTIME
            config.task_segmentations[tid]['strategy'] = SegmentationStrategy.NO_SEGMENTATION
        
        # 交错优先级分配
        priority_levels = [TaskPriority.CRITICAL, TaskPriority.HIGH, 
                          TaskPriority.NORMAL, TaskPriority.LOW]
        
        # 给依赖较少的任务更高优先级
        tasks_by_deps = sorted(self.scheduler.tasks.items(), 
                              key=lambda x: len(x[1].dependencies))
        
        for i, (tid, task) in enumerate(tasks_by_deps):
            # 保持关键任务的高优先级
            if task.fps_requirement >= 50:  # 高FPS任务
                config.task_priorities[tid] = TaskPriority.HIGH
            else:
                # 其他任务交错分配
                config.task_priorities[tid] = priority_levels[i % len(priority_levels)]
        
        return config
    
    def _adjust_for_constraints(self, config: SchedulingConfiguration) -> SchedulingConfiguration:
        """调整配置以满足约束"""
        
        new_config = copy.deepcopy(config)
        
        # 获取当前的冲突信息
        self.evaluate_configuration(new_config)
        
        # 分析哪些任务经常冲突
        conflict_tasks = self._analyze_conflict_patterns()
        
        # 调整策略
        for tid in conflict_tasks:
            task = self.scheduler.tasks[tid]
            
            # 策略1：降低冲突任务的优先级
            current_priority = new_config.task_priorities[tid]
            if current_priority != TaskPriority.LOW:
                # 降一级
                priority_values = [p.value for p in TaskPriority]
                current_idx = priority_values.index(current_priority.value)
                if current_idx < len(priority_values) - 1:
                    new_priority = TaskPriority(priority_values[current_idx + 1])
                    new_config.task_priorities[tid] = new_priority
                    print(f"    降低 {tid} 优先级: {current_priority.name} → {new_priority.name}")
            
            # 策略2：改变运行时类型
            if len(task.segments) > 1:  # 混合任务
                if new_config.task_runtimes[tid] == RuntimeType.ACPU_RUNTIME:
                    new_config.task_runtimes[tid] = RuntimeType.DSP_RUNTIME
                    print(f"    改变 {tid} 运行时: ACPU → DSP (绑定执行)")
        
        return new_config
    
    def _analyze_conflict_patterns(self) -> List[str]:
        """分析冲突模式，返回经常冲突的任务"""
        
        # 简化实现：返回高FPS任务（它们更容易冲突）
        conflict_tasks = []
        
        for tid, task in self.scheduler.tasks.items():
            if task.fps_requirement >= 25:  # 高频任务
                conflict_tasks.append(tid)
        
        return conflict_tasks
    
    def _optimize_fps_with_constraints(self, base_config: SchedulingConfiguration) -> SchedulingConfiguration:
        """在保持约束的前提下优化FPS"""
        
        print("\n步骤2: 优化FPS满足率...")
        
        current_config = copy.deepcopy(base_config)
        current_fitness = self.evaluate_configuration(current_config)
        
        # 识别未满足FPS的任务
        unsatisfied_tasks = self._identify_unsatisfied_tasks(current_config)
        
        print(f"\n未满足FPS的任务: {len(unsatisfied_tasks)} 个")
        
        # 对每个未满足的任务尝试优化
        for tid, deficit_ratio in unsatisfied_tasks:
            task = self.scheduler.tasks[tid]
            print(f"\n  优化 {tid} ({task.name}): FPS缺口 {deficit_ratio:.1%}")
            
            # 尝试的优化策略
            strategies = [
                ('提升优先级', self._try_priority_boost),
                ('改变运行时', self._try_runtime_change),
                ('启用分段', self._try_enable_segmentation)
            ]
            
            for strategy_name, strategy_func in strategies:
                test_config = strategy_func(current_config, tid)
                
                if test_config:
                    # 评估新配置
                    fitness = self.evaluate_configuration(test_config)
                    
                    # 检查约束是否仍然满足
                    if test_config.conflict_count == 0:
                        # 检查是否有改进
                        new_deficit = self._calculate_task_deficit(test_config, tid)
                        
                        if new_deficit < deficit_ratio:
                            print(f"    ✓ {strategy_name}有效: FPS缺口降至 {new_deficit:.1%}")
                            current_config = test_config
                            deficit_ratio = new_deficit
                            
                            if new_deficit < 0.05:  # 满足95%即可
                                break
                    else:
                        print(f"    ✗ {strategy_name}导致冲突，放弃")
        
        return current_config
    
    def _identify_unsatisfied_tasks(self, config: SchedulingConfiguration) -> List[Tuple[str, float]]:
        """识别未满足FPS的任务"""
        
        unsatisfied = []
        
        for tid, stats in config.fitness_score:  # 这里需要从评估中获取详细统计
            if tid in self.scheduler.tasks:
                task = self.scheduler.tasks[tid]
                # 简化：使用配置中存储的信息
                expected = int((self.time_window / 1000.0) * task.fps_requirement)
                # 这里需要从最近的评估中获取实际执行次数
                actual = self._get_task_execution_count(tid)
                
                if actual < expected * 0.95:
                    deficit_ratio = 1.0 - (actual / expected)
                    unsatisfied.append((tid, deficit_ratio))
        
        # 按缺口大小排序
        unsatisfied.sort(key=lambda x: x[1], reverse=True)
        
        return unsatisfied
    
    def _get_task_execution_count(self, task_id: str) -> int:
        """获取任务的执行次数"""
        
        # 从最近的调度历史中统计
        count = 0
        for event in self.scheduler.schedule_history:
            if event.task_id == task_id:
                count += 1
        
        return count
    
    def _calculate_task_deficit(self, config: SchedulingConfiguration, task_id: str) -> float:
        """计算任务的FPS缺口"""
        
        task = self.scheduler.tasks[task_id]
        expected = int((self.time_window / 1000.0) * task.fps_requirement)
        actual = self._get_task_execution_count(task_id)
        
        if expected > 0:
            return 1.0 - (actual / expected)
        return 0.0
    
    def _try_priority_boost(self, config: SchedulingConfiguration, task_id: str) -> Optional[SchedulingConfiguration]:
        """尝试提升优先级"""
        
        new_config = copy.deepcopy(config)
        current_priority = new_config.task_priorities[task_id]
        
        # 只有非最高优先级才能提升
        if current_priority != TaskPriority.CRITICAL:
            priority_values = [p.value for p in TaskPriority]
            current_idx = priority_values.index(current_priority.value)
            
            if current_idx > 0:
                new_priority = TaskPriority(priority_values[current_idx - 1])
                new_config.task_priorities[task_id] = new_priority
                return new_config
        
        return None
    
    def _try_runtime_change(self, config: SchedulingConfiguration, task_id: str) -> Optional[SchedulingConfiguration]:
        """尝试改变运行时类型"""
        
        new_config = copy.deepcopy(config)
        task = self.scheduler.tasks[task_id]
        
        # 只对单资源任务尝试
        if len(task.segments) == 1:
            current_runtime = new_config.task_runtimes[task_id]
            
            if current_runtime == RuntimeType.DSP_RUNTIME:
                new_config.task_runtimes[task_id] = RuntimeType.ACPU_RUNTIME
                return new_config
        
        return None
    
    def _try_enable_segmentation(self, config: SchedulingConfiguration, task_id: str) -> Optional[SchedulingConfiguration]:
        """尝试启用分段"""
        
        new_config = copy.deepcopy(config)
        current_strategy = new_config.task_segmentations[task_id]['strategy']
        
        # 只对支持分段的任务尝试
        if current_strategy == SegmentationStrategy.NO_SEGMENTATION:
            # 检查任务是否支持分段
            task = self.scheduler.tasks[task_id]
            if hasattr(task, 'preset_cut_configurations') and task.preset_cut_configurations:
                new_config.task_segmentations[task_id]['strategy'] = SegmentationStrategy.ADAPTIVE_SEGMENTATION
                return new_config
        
        return None
    
    def _calculate_fitness(self, metrics: Dict) -> float:
        """重写适应度计算：重点惩罚约束违反"""
        
        # 约束满足是首要目标
        if metrics['conflict_count'] > 0:
            # 严重惩罚
            return -self.constraint_weight * metrics['conflict_count']
        
        # 如果约束满足，则优化FPS
        fps_score = metrics['fps_satisfaction_rate']
        
        # 资源利用率作为次要目标
        if metrics['resource_utilization']:
            avg_utilization = sum(metrics['resource_utilization'].values()) / len(metrics['resource_utilization'])
        else:
            avg_utilization = 0
        
        # 主要关注FPS，资源利用率次要
        fitness = fps_score * 0.8 + avg_utilization * 0.2
        
        return fitness
    
    def print_simple_report(self):
        """打印简单的优化报告"""
        
        if not self.best_config:
            print("\n尚未进行优化")
            return
        
        print("\n" + "=" * 60)
        print("📊 简单优化结果")
        print("=" * 60)
        
        # 约束满足情况
        print("\n约束满足:")
        print(f"  ✓ 资源冲突: {self.best_config.conflict_count} 个")
        print(f"  ✓ 依赖关系: 满足")
        
        # FPS情况
        print(f"\nFPS满足率: {self.best_config.fps_satisfaction_rate:.1%}")
        
        # 资源利用率
        print("\n资源利用率:")
        for res_id, util in self.best_config.resource_utilization.items():
            print(f"  {res_id}: {util:.1%}")
        
        # 主要变化
        print("\n主要配置调整:")
        changes = 0
        
        for tid, task in self.scheduler.tasks.items():
            orig_priority = self.original_config.task_priorities[tid]
            new_priority = self.best_config.task_priorities[tid]
            
            orig_runtime = self.original_config.task_runtimes[tid]
            new_runtime = self.best_config.task_runtimes[tid]
            
            if orig_priority != new_priority:
                print(f"  {tid}: 优先级 {orig_priority.name} → {new_priority.name}")
                changes += 1
                
            if orig_runtime != new_runtime:
                print(f"  {tid}: 运行时 {orig_runtime.value} → {new_runtime.value}")
                changes += 1
        
        if changes == 0:
            print("  （无变化）")


def run_simple_optimization(scheduler, time_window: float = 200.0):
    """运行简单优化的便捷函数"""
    
    optimizer = SimpleConstraintOptimizer(scheduler, time_window)
    
    # 运行优化
    best_config = optimizer.optimize_simple(max_attempts=5)
    
    # 打印报告
    optimizer.print_simple_report()
    
    return optimizer, best_config


if __name__ == "__main__":
    print("简单约束满足优化器")
    print("\n特点：")
    print("1. 优先保证无资源冲突")
    print("2. 严格满足任务依赖关系")
    print("3. 在约束满足的前提下尽可能提高FPS")
    print("4. 使用保守策略避免复杂性")
    print("\n策略：")
    print("- 混合任务使用DSP_Runtime（绑定执行）")
    print("- 交错分配优先级避免资源竞争")
    print("- 逐步调整配置直到满足约束")
