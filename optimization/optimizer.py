#!/usr/bin/env python3
"""
Task Scheduler Optimizer
任务调度优化器 - 联合优化优先级、运行时类型、分段配置和核心分配
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from itertools import product
from collections import defaultdict
import random
import time
import numpy as np

from core.enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from core.models import TaskConfig, OptimizationResult, PerformanceMetrics


@dataclass
class SchedulingSearchSpace:
    """定义单个任务的搜索空间"""
    task_id: str
    allowed_priorities: List[TaskPriority]
    allowed_runtime_types: List[RuntimeType]
    segmentation_options: Dict[str, List[int]]  # {段ID: [配置索引列表]}
    available_cores: Dict[ResourceType, List[str]]  # {资源类型: [核心ID列表]}


@dataclass
class SchedulingObjective:
    """优化目标函数组件和权重"""
    latency_weight: float = 1.0           # 延迟权重
    throughput_weight: float = 1.0        # 吞吐量权重
    utilization_weight: float = 0.5       # 利用率权重
    priority_violation_weight: float = 2.0 # 优先级违规惩罚权重
    overhead_weight: float = 0.3          # 分段开销权重


@dataclass
class OptimizationSolution:
    """优化解决方案"""
    task_configs: Dict[str, TaskConfig] = field(default_factory=dict)
    objective_value: float = float('inf')
    metrics: Optional[PerformanceMetrics] = None
    generation: int = 0
    
    def copy(self) -> 'OptimizationSolution':
        """创建解决方案副本"""
        return OptimizationSolution(
            task_configs={k: v.copy() for k, v in self.task_configs.items()},
            objective_value=self.objective_value,
            metrics=self.metrics,
            generation=self.generation
        )


class TaskSchedulerOptimizer:
    """任务调度优化器"""
    
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.search_spaces: Dict[str, SchedulingSearchSpace] = {}
        self.objective = SchedulingObjective()
        self.best_solution: Optional[OptimizationSolution] = None
        self.optimization_history: List[OptimizationSolution] = []
        
    def define_search_space(self, task_id: str, search_space: SchedulingSearchSpace):
        """定义任务的搜索空间"""
        self.search_spaces[task_id] = search_space
        
    def set_objective_weights(self, **kwargs):
        """设置目标函数权重"""
        for key, value in kwargs.items():
            if hasattr(self.objective, key):
                setattr(self.objective, key, value)
    
    def optimize_greedy(self, time_window: float = 500.0, iterations: int = 10) -> OptimizationSolution:
        """贪心优化算法"""
        print(f"🔍 开始贪心优化 (迭代次数: {iterations})")
        
        start_time = time.time()
        best_solution = None
        best_score = float('inf')
        
        for iteration in range(iterations):
            # 生成随机解决方案
            solution = self._generate_random_solution()
            
            # 评估解决方案
            score = self._evaluate_solution(solution, time_window)
            solution.objective_value = score
            
            if score < best_score:
                best_score = score
                best_solution = solution.copy()
                best_solution.generation = iteration
                
                if iteration % max(1, iterations // 4) == 0:
                    print(f"   迭代 {iteration}: 新最佳评分 {score:.2f}")
            
            self.optimization_history.append(solution)
        
        self.best_solution = best_solution
        optimization_time = time.time() - start_time
        
        print(f"✅ 贪心优化完成，耗时 {optimization_time:.2f}秒")
        print(f"   最佳评分: {best_score:.2f}")
        
        return best_solution
    
    def optimize_genetic(self, population_size: int = 30, generations: int = 10, 
                        time_window: float = 500.0) -> OptimizationSolution:
        """遗传算法优化"""
        print(f"🧬 开始遗传算法优化 (种群: {population_size}, 代数: {generations})")
        
        start_time = time.time()
        
        # 初始化种群
        population = [self._generate_random_solution() for _ in range(population_size)]
        
        # 评估初始种群
        for individual in population:
            individual.objective_value = self._evaluate_solution(individual, time_window)
        
        best_solution = min(population, key=lambda x: x.objective_value).copy()
        
        for generation in range(generations):
            # 选择父代
            parents = self._tournament_selection(population, tournament_size=3)
            
            # 生成子代
            offspring = []
            for i in range(0, len(parents), 2):
                parent1 = parents[i]
                parent2 = parents[(i + 1) % len(parents)]
                
                child1, child2 = self._crossover(parent1, parent2)
                
                # 变异
                child1 = self._mutate(child1, mutation_rate=0.1)
                child2 = self._mutate(child2, mutation_rate=0.1)
                
                offspring.extend([child1, child2])
            
            # 评估子代
            for individual in offspring:
                individual.objective_value = self._evaluate_solution(individual, time_window)
                individual.generation = generation
            
            # 环境选择（精英主义）
            all_individuals = population + offspring
            all_individuals.sort(key=lambda x: x.objective_value)
            population = all_individuals[:population_size]
            
            # 更新最佳解
            current_best = population[0]
            if current_best.objective_value < best_solution.objective_value:
                best_solution = current_best.copy()
                print(f"   第{generation}代: 新最佳评分 {current_best.objective_value:.2f}")
            
            self.optimization_history.extend(offspring)
        
        self.best_solution = best_solution
        optimization_time = time.time() - start_time
        
        print(f"✅ 遗传算法完成，耗时 {optimization_time:.2f}秒")
        print(f"   最佳评分: {best_solution.objective_value:.2f}")
        
        return best_solution
    
    def _generate_random_solution(self) -> OptimizationSolution:
        """生成随机解决方案"""
        solution = OptimizationSolution()
        
        for task_id, search_space in self.search_spaces.items():
            # 随机选择优先级
            priority = random.choice(search_space.allowed_priorities)
            
            # 随机选择运行时类型
            runtime_type = random.choice(search_space.allowed_runtime_types)
            
            # 随机选择分段配置
            segmentation_configs = {}
            for segment_id, options in search_space.segmentation_options.items():
                if options:
                    segmentation_configs[segment_id] = random.choice(options)
            
            # 随机选择核心分配
            core_assignments = {}
            for resource_type, cores in search_space.available_cores.items():
                if cores and segmentation_configs:
                    # 为每个需要该资源类型的段分配核心
                    for segment_id in segmentation_configs.keys():
                        if f"{resource_type.value.lower()}" in segment_id.lower():
                            core_assignments[segment_id] = random.choice(cores)
            
            task_config = TaskConfig(
                task_id=task_id,
                priority=priority,
                runtime_type=runtime_type,
                segmentation_configs=segmentation_configs,
                core_assignments=core_assignments
            )
            
            solution.task_configs[task_id] = task_config
        
        return solution
    
    def _evaluate_solution(self, solution: OptimizationSolution, time_window: float) -> float:
        """评估解决方案的目标函数值"""
        # 保存原始任务配置
        original_configs = {}
        for task_id, task in self.scheduler.tasks.items():
            original_configs[task_id] = {
                'priority': task.priority,
                'runtime_type': task.runtime_type
            }
        
        try:
            # 应用解决方案配置
            self._apply_solution_to_scheduler(solution)
            
            # 运行调度
            results = self.scheduler.priority_aware_schedule_with_segmentation(time_window)
            
            if not results or not self.scheduler.schedule_history:
                return float('inf')  # 调度失败
            
            # 计算性能指标
            metrics = self._calculate_metrics(results, time_window)
            solution.metrics = metrics
            
            # 计算目标函数值
            objective_value = self._calculate_objective_value(metrics)
            
            return objective_value
            
        except Exception as e:
            print(f"⚠️  评估解决方案时出错: {e}")
            return float('inf')
        
        finally:
            # 恢复原始配置
            self._restore_original_configs(original_configs)
    
    def _apply_solution_to_scheduler(self, solution: OptimizationSolution):
        """将解决方案应用到调度器"""
        for task_id, config in solution.task_configs.items():
            if task_id in self.scheduler.tasks:
                task = self.scheduler.tasks[task_id]
                task.priority = config.priority
                task.runtime_type = config.runtime_type
                
                # 应用分段配置
                for segment_id, config_idx in config.segmentation_configs.items():
                    if hasattr(task, 'select_cut_configuration'):
                        try:
                            task.select_cut_configuration(segment_id, config_idx)
                        except:
                            pass  # 忽略配置错误
    
    def _restore_original_configs(self, original_configs: Dict):
        """恢复原始任务配置"""
        for task_id, config in original_configs.items():
            if task_id in self.scheduler.tasks:
                task = self.scheduler.tasks[task_id]
                task.priority = config['priority']
                task.runtime_type = config['runtime_type']
                
                # 重置分段配置
                if hasattr(task, 'current_segmentation'):
                    task.current_segmentation = {}
    
    def _calculate_metrics(self, results, time_window: float) -> PerformanceMetrics:
        """计算性能指标"""
        metrics = PerformanceMetrics()
        
        if not self.scheduler.schedule_history:
            return metrics
        
        # 基础指标
        metrics.total_tasks = len(self.scheduler.schedule_history)
        metrics.makespan = max(s.end_time for s in self.scheduler.schedule_history)
        
        latencies = [s.end_time - s.start_time for s in self.scheduler.schedule_history]
        metrics.average_latency = sum(latencies) / len(latencies)
        
        # 资源利用率
        resource_usage = {}
        for schedule in self.scheduler.schedule_history:
            for res_type, res_id in schedule.assigned_resources.items():
                if res_id not in resource_usage:
                    resource_usage[res_id] = 0
                resource_usage[res_id] += schedule.end_time - schedule.start_time
        
        for res_id, usage in resource_usage.items():
            metrics.resource_utilization[res_id] = (usage / metrics.makespan) * 100
        
        if metrics.resource_utilization:
            metrics.average_utilization = sum(metrics.resource_utilization.values()) / len(metrics.resource_utilization)
        
        # 违规检查（简化版）
        task_counts = {}
        for schedule in self.scheduler.schedule_history:
            task_id = schedule.task_id
            task_counts[task_id] = task_counts.get(task_id, 0) + 1
        
        for task_id, task in self.scheduler.tasks.items():
            if hasattr(task, 'fps_requirement') and task.fps_requirement:
                count = task_counts.get(task_id, 0)
                achieved_fps = count / (time_window / 1000.0)
                if achieved_fps < task.fps_requirement * 0.95:
                    metrics.fps_violations += 1
            
            if hasattr(task, 'latency_requirement') and task.latency_requirement:
                for schedule in self.scheduler.schedule_history:
                    if schedule.task_id == task_id:
                        if schedule.end_time - schedule.start_time > task.latency_requirement:
                            metrics.latency_violations += 1
                            break
        
        # 优先级分布
        for task in self.scheduler.tasks.values():
            metrics.priority_distribution[task.priority] = metrics.priority_distribution.get(task.priority, 0) + 1
        
        return metrics
    
    def _calculate_objective_value(self, metrics: PerformanceMetrics) -> float:
        """计算目标函数值（越小越好）"""
        # 延迟分量
        latency_component = metrics.average_latency * self.objective.latency_weight
        
        # 资源利用率分量（转换为惩罚，低利用率=高惩罚）
        utilization_penalty = (100 - metrics.average_utilization) * self.objective.utilization_weight
        
        # 违规惩罚
        violation_penalty = (metrics.fps_violations + metrics.latency_violations) * self.objective.priority_violation_weight
        
        # 总目标值
        objective_value = latency_component + utilization_penalty + violation_penalty
        
        return objective_value
    
    def _tournament_selection(self, population: List[OptimizationSolution], 
                            tournament_size: int = 3) -> List[OptimizationSolution]:
        """锦标赛选择"""
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(population, min(tournament_size, len(population)))
            winner = min(tournament, key=lambda x: x.objective_value)
            selected.append(winner)
        return selected
    
    def _crossover(self, parent1: OptimizationSolution, 
                  parent2: OptimizationSolution) -> Tuple[OptimizationSolution, OptimizationSolution]:
        """交叉操作"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # 随机选择任务进行交叉
        task_ids = list(parent1.task_configs.keys())
        crossover_point = random.randint(1, len(task_ids) - 1)
        
        for i, task_id in enumerate(task_ids):
            if i >= crossover_point:
                child1.task_configs[task_id] = parent2.task_configs[task_id].copy()
                child2.task_configs[task_id] = parent1.task_configs[task_id].copy()
        
        return child1, child2
    
    def _mutate(self, solution: OptimizationSolution, mutation_rate: float = 0.1) -> OptimizationSolution:
        """变异操作"""
        mutated = solution.copy()
        
        for task_id, config in mutated.task_configs.items():
            if random.random() < mutation_rate and task_id in self.search_spaces:
                search_space = self.search_spaces[task_id]
                
                # 随机选择变异类型
                mutation_type = random.choice(['priority', 'runtime', 'segmentation'])
                
                if mutation_type == 'priority' and search_space.allowed_priorities:
                    config.priority = random.choice(search_space.allowed_priorities)
                elif mutation_type == 'runtime' and search_space.allowed_runtime_types:
                    config.runtime_type = random.choice(search_space.allowed_runtime_types)
                elif mutation_type == 'segmentation' and search_space.segmentation_options:
                    segment_id = random.choice(list(search_space.segmentation_options.keys()))
                    options = search_space.segmentation_options[segment_id]
                    if options:
                        config.segmentation_configs[segment_id] = random.choice(options)
        
        return mutated
    
    def print_solution(self, solution: OptimizationSolution):
        """打印解决方案信息"""
        if not solution:
            print("❌ 没有解决方案可显示")
            return
        
        print(f"\n📊 优化解决方案 (评分: {solution.objective_value:.2f})")
        print(f"{'任务ID':<8} {'优先级':<12} {'运行时':<15} {'分段配置':<20}")
        print("-" * 60)
        
        for task_id, config in solution.task_configs.items():
            seg_config = ", ".join([f"{k}:{v}" for k, v in config.segmentation_configs.items()])
            if not seg_config:
                seg_config = "无"
            
            print(f"{task_id:<8} {config.priority.name:<12} {config.runtime_type.value:<15} {seg_config:<20}")
        
        if solution.metrics:
            print(f"\n性能指标:")
            print(f"  • 总完成时间: {solution.metrics.makespan:.1f}ms")
            print(f"  • 平均延迟: {solution.metrics.average_latency:.1f}ms")
            print(f"  • 平均利用率: {solution.metrics.average_utilization:.1f}%")
            print(f"  • FPS违规: {solution.metrics.fps_violations}")
            print(f"  • 延迟违规: {solution.metrics.latency_violations}")
    
    def get_optimization_history(self) -> List[OptimizationSolution]:
        """获取优化历史"""
        return self.optimization_history.copy()


if __name__ == "__main__":
    print("=== 任务调度优化器测试 ===")
    print("请通过主程序或演示脚本运行优化器测试")
