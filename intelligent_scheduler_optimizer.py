#!/usr/bin/env python3
"""
智能调度优化框架
使用机器学习方法寻找最佳的调度配置组合
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import random
import copy
from collections import defaultdict

from enums import TaskPriority, RuntimeType, SegmentationStrategy


@dataclass
class SchedulingConfiguration:
    """调度配置"""
    
    # 任务配置
    task_priorities: Dict[str, TaskPriority]  # 任务ID -> 优先级
    task_runtimes: Dict[str, RuntimeType]     # 任务ID -> 运行时类型
    task_segmentations: Dict[str, Dict]       # 任务ID -> 分段配置
    
    # 调度器配置
    scheduler_params: Dict[str, Any]          # 调度器参数
    
    # 性能指标
    fitness_score: float = 0.0
    fps_satisfaction_rate: float = 0.0
    resource_utilization: Dict[str, float] = None
    conflict_count: int = 0
    
    def __post_init__(self):
        if self.resource_utilization is None:
            self.resource_utilization = {}


class OptimizationMethod(Enum):
    """优化方法"""
    GREEDY = "greedy"
    GENETIC = "genetic"
    DYNAMIC_PROGRAMMING = "dp"
    SIMULATED_ANNEALING = "sa"
    REINFORCEMENT_LEARNING = "rl"


class IntelligentSchedulerOptimizer:
    """智能调度优化器基类"""
    
    def __init__(self, scheduler, time_window: float = 200.0):
        self.scheduler = scheduler
        self.time_window = time_window
        self.original_config = self._save_original_config()
        self.best_config = None
        self.optimization_history = []
        
    def _save_original_config(self) -> SchedulingConfiguration:
        """保存原始配置"""
        config = SchedulingConfiguration(
            task_priorities={tid: task.priority for tid, task in self.scheduler.tasks.items()},
            task_runtimes={tid: task.runtime_type for tid, task in self.scheduler.tasks.items()},
            task_segmentations=self._extract_segmentation_config(),
            scheduler_params={}
        )
        return config
    
    def _extract_segmentation_config(self) -> Dict:
        """提取分段配置"""
        seg_config = {}
        for tid, task in self.scheduler.tasks.items():
            seg_config[tid] = {
                'strategy': task.segmentation_strategy,
                'cut_points': getattr(task, 'current_segmentation', {})
            }
        return seg_config
    
    def evaluate_configuration(self, config: SchedulingConfiguration) -> float:
        """评估配置的适应度"""
        
        # 应用配置
        self._apply_configuration(config)
        
        # 运行调度
        self.scheduler.schedule_history.clear()
        results = self.scheduler.priority_aware_schedule_with_segmentation(self.time_window)
        
        # 计算指标
        metrics = self._calculate_metrics(results)
        
        # 更新配置的性能指标
        config.fps_satisfaction_rate = metrics['fps_satisfaction_rate']
        config.resource_utilization = metrics['resource_utilization']
        config.conflict_count = metrics['conflict_count']
        
        # 计算综合适应度分数
        fitness = self._calculate_fitness(metrics)
        config.fitness_score = fitness
        
        return fitness
    
    def _apply_configuration(self, config: SchedulingConfiguration):
        """应用配置到调度器"""
        
        # 应用任务优先级
        for tid, priority in config.task_priorities.items():
            if tid in self.scheduler.tasks:
                self.scheduler.tasks[tid].priority = priority
        
        # 应用运行时类型
        for tid, runtime in config.task_runtimes.items():
            if tid in self.scheduler.tasks:
                self.scheduler.tasks[tid].runtime_type = runtime
        
        # 应用分段配置
        for tid, seg_config in config.task_segmentations.items():
            if tid in self.scheduler.tasks:
                task = self.scheduler.tasks[tid]
                task.segmentation_strategy = seg_config['strategy']
                if 'cut_points' in seg_config:
                    task.current_segmentation = seg_config['cut_points']
    
    def _calculate_metrics(self, results: List) -> Dict:
        """计算性能指标"""
        
        metrics = {
            'fps_satisfaction_rate': 0.0,
            'resource_utilization': {},
            'conflict_count': 0,
            'total_events': len(results),
            'task_stats': {}
        }
        
        # 统计任务执行
        task_counts = defaultdict(int)
        for event in results:
            task_counts[event.task_id] += 1
        
        # 计算FPS满足率
        satisfied_tasks = 0
        total_tasks = len(self.scheduler.tasks)
        
        for tid, task in self.scheduler.tasks.items():
            expected = int((self.time_window / 1000.0) * task.fps_requirement)
            actual = task_counts[tid]
            rate = (actual / expected) if expected > 0 else 0
            
            metrics['task_stats'][tid] = {
                'expected': expected,
                'actual': actual,
                'rate': rate
            }
            
            if rate >= 0.95:  # 95%即满足
                satisfied_tasks += 1
        
        metrics['fps_satisfaction_rate'] = satisfied_tasks / total_tasks if total_tasks > 0 else 0
        
        # 计算资源利用率
        resource_busy_time = defaultdict(float)
        for event in results:
            duration = event.end_time - event.start_time
            for res_type, res_id in event.assigned_resources.items():
                resource_busy_time[res_id] += duration
        
        for res_id, busy_time in resource_busy_time.items():
            metrics['resource_utilization'][res_id] = busy_time / self.time_window
        
        # 检测冲突（简化版）
        metrics['conflict_count'] = self._detect_conflicts(results)
        
        return metrics
    
    def _detect_conflicts(self, results: List) -> int:
        """检测资源冲突"""
        
        conflicts = 0
        resource_timeline = defaultdict(list)
        
        for event in results:
            for res_type, res_id in event.assigned_resources.items():
                resource_timeline[res_id].append((event.start_time, event.end_time))
        
        for res_id, timeline in resource_timeline.items():
            timeline.sort()
            for i in range(len(timeline) - 1):
                if timeline[i][1] > timeline[i+1][0]:
                    conflicts += 1
        
        return conflicts
    
    def _calculate_fitness(self, metrics: Dict) -> float:
        """计算适应度分数"""
        
        # 权重配置
        weights = {
            'fps': 0.5,      # FPS满足率权重
            'utilization': 0.2,  # 资源利用率权重
            'conflicts': 0.3     # 冲突惩罚权重
        }
        
        # FPS得分
        fps_score = metrics['fps_satisfaction_rate']
        
        # 资源利用率得分（平均利用率）
        if metrics['resource_utilization']:
            avg_utilization = sum(metrics['resource_utilization'].values()) / len(metrics['resource_utilization'])
        else:
            avg_utilization = 0
        
        # 冲突惩罚
        conflict_penalty = 1.0 / (1 + metrics['conflict_count'])
        
        # 综合得分
        fitness = (weights['fps'] * fps_score + 
                  weights['utilization'] * avg_utilization * conflict_penalty)
        
        return fitness
    
    def optimize(self, method: OptimizationMethod, **kwargs) -> SchedulingConfiguration:
        """执行优化"""
        
        print(f"\n🚀 开始{method.value}优化")
        print("=" * 60)
        
        if method == OptimizationMethod.GREEDY:
            return self._greedy_optimization(**kwargs)
        elif method == OptimizationMethod.GENETIC:
            return self._genetic_optimization(**kwargs)
        elif method == OptimizationMethod.DYNAMIC_PROGRAMMING:
            return self._dynamic_programming_optimization(**kwargs)
        elif method == OptimizationMethod.SIMULATED_ANNEALING:
            return self._simulated_annealing_optimization(**kwargs)
        else:
            raise NotImplementedError(f"优化方法 {method} 尚未实现")
    
    def _greedy_optimization(self, max_iterations: int = 100) -> SchedulingConfiguration:
        """贪心优化"""
        
        print("使用贪心算法优化...")
        
        current_config = copy.deepcopy(self.original_config)
        current_fitness = self.evaluate_configuration(current_config)
        
        print(f"初始适应度: {current_fitness:.3f}")
        
        for iteration in range(max_iterations):
            improved = False
            
            # 尝试所有可能的单步改进
            for improvement in self._generate_single_improvements(current_config):
                new_fitness = self.evaluate_configuration(improvement)
                
                if new_fitness > current_fitness:
                    current_config = improvement
                    current_fitness = new_fitness
                    improved = True
                    print(f"  迭代 {iteration + 1}: 适应度提升到 {current_fitness:.3f}")
                    break
            
            if not improved:
                print(f"  迭代 {iteration + 1}: 无法继续改进")
                break
        
        self.best_config = current_config
        return current_config
    
    def _generate_single_improvements(self, config: SchedulingConfiguration) -> List[SchedulingConfiguration]:
        """生成单步改进"""
        
        improvements = []
        
        # 尝试改变每个任务的优先级
        for tid in config.task_priorities:
            for priority in TaskPriority:
                if priority != config.task_priorities[tid]:
                    new_config = copy.deepcopy(config)
                    new_config.task_priorities[tid] = priority
                    improvements.append(new_config)
        
        # 尝试改变运行时类型
        for tid in config.task_runtimes:
            for runtime in RuntimeType:
                if runtime != config.task_runtimes[tid]:
                    new_config = copy.deepcopy(config)
                    new_config.task_runtimes[tid] = runtime
                    improvements.append(new_config)
        
        # 尝试改变分段策略
        for tid in config.task_segmentations:
            for strategy in SegmentationStrategy:
                if strategy != config.task_segmentations[tid]['strategy']:
                    new_config = copy.deepcopy(config)
                    new_config.task_segmentations[tid]['strategy'] = strategy
                    improvements.append(new_config)
        
        return improvements
    
    def _genetic_optimization(self, population_size: int = 50, 
                            generations: int = 100,
                            mutation_rate: float = 0.1) -> SchedulingConfiguration:
        """遗传算法优化"""
        
        print(f"使用遗传算法优化 (种群大小: {population_size}, 代数: {generations})")
        
        # 初始化种群
        population = self._initialize_population(population_size)
        
        # 评估初始种群
        for individual in population:
            self.evaluate_configuration(individual)
        
        # 排序
        population.sort(key=lambda x: x.fitness_score, reverse=True)
        best_fitness = population[0].fitness_score
        print(f"初始最佳适应度: {best_fitness:.3f}")
        
        # 进化
        for generation in range(generations):
            # 选择
            parents = self._selection(population)
            
            # 交叉
            offspring = self._crossover(parents)
            
            # 变异
            self._mutation(offspring, mutation_rate)
            
            # 评估新个体
            for individual in offspring:
                self.evaluate_configuration(individual)
            
            # 合并种群
            population.extend(offspring)
            population.sort(key=lambda x: x.fitness_score, reverse=True)
            
            # 保留最好的个体
            population = population[:population_size]
            
            # 记录进展
            if population[0].fitness_score > best_fitness:
                best_fitness = population[0].fitness_score
                print(f"  第 {generation + 1} 代: 适应度提升到 {best_fitness:.3f}")
        
        self.best_config = population[0]
        return population[0]
    
    def _initialize_population(self, size: int) -> List[SchedulingConfiguration]:
        """初始化种群"""
        
        population = []
        
        # 添加原始配置
        population.append(copy.deepcopy(self.original_config))
        
        # 生成随机个体
        while len(population) < size:
            individual = self._generate_random_configuration()
            population.append(individual)
        
        return population
    
    def _generate_random_configuration(self) -> SchedulingConfiguration:
        """生成随机配置"""
        
        config = SchedulingConfiguration(
            task_priorities={},
            task_runtimes={},
            task_segmentations={},
            scheduler_params={}
        )
        
        for tid, task in self.scheduler.tasks.items():
            # 随机优先级
            config.task_priorities[tid] = random.choice(list(TaskPriority))
            
            # 随机运行时
            config.task_runtimes[tid] = random.choice(list(RuntimeType))
            
            # 随机分段策略
            config.task_segmentations[tid] = {
                'strategy': random.choice(list(SegmentationStrategy)),
                'cut_points': {}
            }
        
        return config
    
    def _selection(self, population: List[SchedulingConfiguration]) -> List[SchedulingConfiguration]:
        """选择操作（锦标赛选择）"""
        
        parents = []
        tournament_size = 3
        
        for _ in range(len(population) // 2):
            # 随机选择参赛者
            tournament = random.sample(population, tournament_size)
            # 选择最好的
            winner = max(tournament, key=lambda x: x.fitness_score)
            parents.append(copy.deepcopy(winner))
        
        return parents
    
    def _crossover(self, parents: List[SchedulingConfiguration]) -> List[SchedulingConfiguration]:
        """交叉操作"""
        
        offspring = []
        
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            # 创建两个子代
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)
            
            # 随机交换一些任务的配置
            for tid in parent1.task_priorities:
                if random.random() < 0.5:
                    # 交换优先级
                    child1.task_priorities[tid], child2.task_priorities[tid] = \
                        child2.task_priorities[tid], child1.task_priorities[tid]
                    
                    # 交换运行时
                    child1.task_runtimes[tid], child2.task_runtimes[tid] = \
                        child2.task_runtimes[tid], child1.task_runtimes[tid]
            
            offspring.extend([child1, child2])
        
        return offspring
    
    def _mutation(self, population: List[SchedulingConfiguration], rate: float):
        """变异操作"""
        
        for individual in population:
            for tid in individual.task_priorities:
                # 变异优先级
                if random.random() < rate:
                    individual.task_priorities[tid] = random.choice(list(TaskPriority))
                
                # 变异运行时
                if random.random() < rate:
                    individual.task_runtimes[tid] = random.choice(list(RuntimeType))
                
                # 变异分段策略
                if random.random() < rate:
                    individual.task_segmentations[tid]['strategy'] = \
                        random.choice(list(SegmentationStrategy))
    
    def _dynamic_programming_optimization(self, **kwargs) -> SchedulingConfiguration:
        """动态规划优化（适用于有明确阶段的问题）"""
        
        print("使用动态规划优化...")
        
        # 这里实现一个简化版本
        # 将问题分解为子问题：每个任务的最优配置
        
        optimal_config = copy.deepcopy(self.original_config)
        
        # 对每个任务独立优化
        for tid, task in self.scheduler.tasks.items():
            print(f"\n优化任务 {tid} ({task.name})...")
            
            best_priority = task.priority
            best_runtime = task.runtime_type
            best_fitness = 0
            
            # 尝试所有组合
            for priority in TaskPriority:
                for runtime in RuntimeType:
                    # 创建临时配置
                    temp_config = copy.deepcopy(optimal_config)
                    temp_config.task_priorities[tid] = priority
                    temp_config.task_runtimes[tid] = runtime
                    
                    # 评估
                    fitness = self.evaluate_configuration(temp_config)
                    
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_priority = priority
                        best_runtime = runtime
            
            # 应用最优选择
            optimal_config.task_priorities[tid] = best_priority
            optimal_config.task_runtimes[tid] = best_runtime
            
            print(f"  最优配置: 优先级={best_priority.name}, 运行时={best_runtime.value}")
        
        self.best_config = optimal_config
        return optimal_config
    
    def _simulated_annealing_optimization(self, initial_temp: float = 100.0,
                                        cooling_rate: float = 0.95,
                                        min_temp: float = 1.0) -> SchedulingConfiguration:
        """模拟退火优化"""
        
        print(f"使用模拟退火优化 (初始温度: {initial_temp}, 冷却率: {cooling_rate})")
        
        current_config = copy.deepcopy(self.original_config)
        current_fitness = self.evaluate_configuration(current_config)
        
        best_config = copy.deepcopy(current_config)
        best_fitness = current_fitness
        
        temperature = initial_temp
        
        while temperature > min_temp:
            # 生成邻居解
            neighbor = self._generate_neighbor(current_config)
            neighbor_fitness = self.evaluate_configuration(neighbor)
            
            # 计算接受概率
            delta = neighbor_fitness - current_fitness
            
            if delta > 0 or random.random() < math.exp(delta / temperature):
                # 接受新解
                current_config = neighbor
                current_fitness = neighbor_fitness
                
                # 更新最佳解
                if current_fitness > best_fitness:
                    best_config = copy.deepcopy(current_config)
                    best_fitness = current_fitness
                    print(f"  温度 {temperature:.1f}: 找到更好的解 (适应度: {best_fitness:.3f})")
            
            # 降温
            temperature *= cooling_rate
        
        self.best_config = best_config
        return best_config
    
    def _generate_neighbor(self, config: SchedulingConfiguration) -> SchedulingConfiguration:
        """生成邻居解"""
        
        neighbor = copy.deepcopy(config)
        
        # 随机选择要改变的内容
        change_type = random.choice(['priority', 'runtime', 'segmentation'])
        task_id = random.choice(list(config.task_priorities.keys()))
        
        if change_type == 'priority':
            # 改变优先级
            current = neighbor.task_priorities[task_id]
            options = [p for p in TaskPriority if p != current]
            if options:
                neighbor.task_priorities[task_id] = random.choice(options)
                
        elif change_type == 'runtime':
            # 改变运行时
            current = neighbor.task_runtimes[task_id]
            options = [r for r in RuntimeType if r != current]
            if options:
                neighbor.task_runtimes[task_id] = random.choice(options)
                
        else:
            # 改变分段策略
            current = neighbor.task_segmentations[task_id]['strategy']
            options = [s for s in SegmentationStrategy if s != current]
            if options:
                neighbor.task_segmentations[task_id]['strategy'] = random.choice(options)
        
        return neighbor
    
    def print_optimization_report(self):
        """打印优化报告"""
        
        if not self.best_config:
            print("尚未进行优化")
            return
        
        print("\n" + "=" * 60)
        print("📊 优化结果报告")
        print("=" * 60)
        
        print(f"\n最佳适应度: {self.best_config.fitness_score:.3f}")
        print(f"FPS满足率: {self.best_config.fps_satisfaction_rate:.1%}")
        print(f"资源冲突: {self.best_config.conflict_count} 个")
        
        print("\n资源利用率:")
        for res_id, util in self.best_config.resource_utilization.items():
            print(f"  {res_id}: {util:.1%}")
        
        print("\n任务配置变化:")
        for tid, task in self.scheduler.tasks.items():
            orig_priority = self.original_config.task_priorities[tid]
            new_priority = self.best_config.task_priorities[tid]
            
            orig_runtime = self.original_config.task_runtimes[tid]
            new_runtime = self.best_config.task_runtimes[tid]
            
            if orig_priority != new_priority or orig_runtime != new_runtime:
                print(f"\n  {tid} ({task.name}):")
                if orig_priority != new_priority:
                    print(f"    优先级: {orig_priority.name} → {new_priority.name}")
                if orig_runtime != new_runtime:
                    print(f"    运行时: {orig_runtime.value} → {new_runtime.value}")


# 导入必要的数学库
try:
    import math
except ImportError:
    pass


if __name__ == "__main__":
    print("智能调度优化框架")
    print("\n支持的优化方法：")
    for method in OptimizationMethod:
        print(f"  - {method.value}")
    print("\n特性：")
    print("1. 多种优化算法（贪心、遗传、动态规划、模拟退火）")
    print("2. 自动搜索最佳配置组合")
    print("3. 综合考虑FPS满足率、资源利用率和冲突")
    print("4. 支持优先级、运行时类型和分段策略的联合优化")
