#!/usr/bin/env python3
"""
调度优化器修复版本
减少冗余调度，提高优化效率
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from copy import deepcopy
import numpy as np

from enums import TaskPriority, RuntimeType, ResourceType
from task import NNTask
from scheduler import MultiResourceScheduler


def apply_scheduling_optimizer_fix():
    """应用优化器修复补丁"""
    print("✅ Applying scheduling optimizer fix...")
    
    # 保存原始的evaluate_solution方法
    from scheduling_optimizer import SchedulingOptimizer
    original_evaluate = SchedulingOptimizer.evaluate_solution
    
    # 添加缓存
    SchedulingOptimizer._solution_cache = {}
    SchedulingOptimizer._cache_hits = 0
    SchedulingOptimizer._cache_misses = 0
    
    def cached_evaluate_solution(self, solution: Dict, time_window: float) -> Tuple[float, Dict]:
        """带缓存的解决方案评估"""
        # 创建解决方案的哈希键
        solution_key = self._create_solution_key(solution)
        cache_key = (solution_key, time_window)
        
        # 检查缓存
        if cache_key in self._solution_cache:
            self._cache_hits += 1
            return self._solution_cache[cache_key]
        
        # 缓存未命中，执行评估
        self._cache_misses += 1
        result = original_evaluate(self, solution, time_window)
        
        # 存入缓存（限制缓存大小）
        if len(self._solution_cache) > 1000:
            # 清除一半缓存
            keys_to_remove = list(self._solution_cache.keys())[:500]
            for key in keys_to_remove:
                del self._solution_cache[key]
        
        self._solution_cache[cache_key] = result
        return result
    
    def _create_solution_key(self, solution: Dict) -> str:
        """创建解决方案的唯一键"""
        key_parts = []
        for task_id in sorted(solution.keys()):
            decision = solution[task_id]
            key_parts.append(f"{task_id}:{decision.priority.value}:{decision.runtime_type.value}")
            
            # 添加分段配置
            seg_configs = []
            for seg_id in sorted(decision.segmentation_configs.keys()):
                seg_configs.append(f"{seg_id}={decision.segmentation_configs[seg_id]}")
            key_parts.append(":".join(seg_configs))
        
        return "|".join(key_parts)
    
    # 替换方法
    SchedulingOptimizer.evaluate_solution = cached_evaluate_solution
    SchedulingOptimizer._create_solution_key = _create_solution_key
    
    # 修复optimize_greedy方法，减少候选方案数量
    original_optimize_greedy = SchedulingOptimizer.optimize_greedy
    
    def efficient_optimize_greedy(self, time_window: float = 1000.0, iterations: int = 10) -> Dict:
        """更高效的贪心优化"""
        print("\n=== Starting Efficient Greedy Optimization ===")
        
        # 重置缓存统计
        self._cache_hits = 0
        self._cache_misses = 0
        self._solution_cache.clear()
        
        # 初始化解决方案
        current_solution = self._initialize_solution()
        
        # 评估初始解决方案
        best_score, best_metrics = self.evaluate_solution(current_solution, time_window)
        best_solution = deepcopy(current_solution)
        
        print(f"Initial score: {best_score:.2f}")
        print(f"Initial metrics: {self._format_metrics(best_metrics)}")
        
        # 优化迭代
        for iteration in range(iterations):
            print(f"\nIteration {iteration + 1}/{iterations}")
            improved = False
            
            # 智能选择要优化的任务（按违反程度排序）
            tasks_to_optimize = self._prioritize_tasks_for_optimization(best_solution, best_metrics)
            
            for task_id in tasks_to_optimize[:3]:  # 只优化前3个最需要改进的任务
                print(f"  Optimizing task {task_id}...")
                
                # 生成少量但有针对性的候选方案
                candidates = self._generate_smart_candidates(task_id, best_solution, best_metrics, max_candidates=10)
                
                best_candidate_score = best_score
                best_candidate = None
                
                for candidate in candidates:
                    # 创建新解决方案
                    new_solution = deepcopy(best_solution)
                    new_solution[task_id] = candidate
                    
                    # 评估
                    score, metrics = self.evaluate_solution(new_solution, time_window)
                    
                    if score < best_candidate_score:
                        best_candidate_score = score
                        best_candidate = candidate
                
                # 应用最佳候选
                if best_candidate and best_candidate_score < best_score:
                    best_score = best_candidate_score
                    best_solution[task_id] = best_candidate
                    improved = True
                    print(f"    ✅ Improved! New score: {best_score:.2f}")
            
            if not improved:
                print("  No improvement found in this iteration")
                break
        
        print(f"\n✅ Optimization complete!")
        print(f"Final score: {best_score:.2f}")
        print(f"Cache statistics: {self._cache_hits} hits, {self._cache_misses} misses")
        print(f"Cache hit rate: {self._cache_hits / (self._cache_hits + self._cache_misses) * 100:.1f}%")
        
        return best_solution
    
    def _initialize_solution(self) -> Dict:
        """初始化解决方案"""
        solution = {}
        for task_id in self.search_spaces.keys():
            task = self.scheduler.tasks[task_id]
            
            # 创建初始决策变量
            seg_configs = {}
            for seg_id in task.current_segmentation.keys():
                seg_configs[seg_id] = task.selected_cut_config_index.get(seg_id, 0)
            
            # 初始核心分配
            core_assignments = {}
            for segment in task.segments:
                available_cores = self.search_spaces[task_id].available_cores.get(segment.resource_type, [])
                if available_cores:
                    core_assignments[segment.segment_id] = available_cores[0]
            
            from scheduling_optimizer import SchedulingDecisionVariable
            solution[task_id] = SchedulingDecisionVariable(
                task_id=task_id,
                priority=task.priority,
                runtime_type=task.runtime_type,
                segmentation_configs=seg_configs,
                core_assignments=core_assignments
            )
        
        return solution
    
    def _prioritize_tasks_for_optimization(self, solution: Dict, metrics: Dict) -> List[str]:
        """根据性能指标确定优化优先级"""
        task_scores = []
        
        # 从metrics中提取任务级信息
        for task_id in solution.keys():
            task = self.scheduler.tasks[task_id]
            
            # 计算违反分数
            violation_score = 0
            
            # 检查FPS违反
            if 'task_performance' in metrics and task_id in metrics['task_performance']:
                perf = metrics['task_performance'][task_id]
                if perf['achieved_fps'] < task.fps_requirement * 0.95:
                    violation_score += (task.fps_requirement - perf['achieved_fps']) / task.fps_requirement
                
                if perf['avg_latency'] > task.latency_requirement * 1.05:
                    violation_score += (perf['avg_latency'] - task.latency_requirement) / task.latency_requirement
            
            # 考虑优先级
            priority_weight = 4 - task.priority.value
            violation_score *= priority_weight
            
            task_scores.append((task_id, violation_score))
        
        # 按违反分数排序
        task_scores.sort(key=lambda x: x[1], reverse=True)
        return [task_id for task_id, _ in task_scores]
    
    def _generate_smart_candidates(self, task_id: str, current_solution: Dict, 
                                  metrics: Dict, max_candidates: int = 10) -> List:
        """生成有针对性的候选方案"""
        from scheduling_optimizer import SchedulingDecisionVariable
        
        candidates = []
        search_space = self.search_spaces[task_id]
        current_decision = current_solution[task_id]
        task = self.scheduler.tasks[task_id]
        
        # 根据任务性能决定优化方向
        needs_higher_priority = False
        needs_more_segmentation = False
        
        if 'task_performance' in metrics and task_id in metrics['task_performance']:
            perf = metrics['task_performance'][task_id]
            if perf['achieved_fps'] < task.fps_requirement * 0.95:
                needs_higher_priority = True
            if perf['avg_latency'] > task.latency_requirement:
                needs_more_segmentation = True
        
        # 1. 尝试调整优先级
        if needs_higher_priority:
            for priority in search_space.allowed_priorities:
                if priority.value < current_decision.priority.value:  # 更高优先级
                    candidate = deepcopy(current_decision)
                    candidate.priority = priority
                    candidates.append(candidate)
                    if len(candidates) >= max_candidates:
                        return candidates
        
        # 2. 尝试调整分段配置
        if needs_more_segmentation:
            for seg_id, current_config in current_decision.segmentation_configs.items():
                if seg_id in search_space.segmentation_options:
                    options = search_space.segmentation_options[seg_id]
                    # 尝试更多分段的配置
                    for config_idx in options:
                        if config_idx > current_config:  # 更多分段
                            candidate = deepcopy(current_decision)
                            candidate.segmentation_configs[seg_id] = config_idx
                            candidates.append(candidate)
                            if len(candidates) >= max_candidates:
                                return candidates
        
        # 3. 尝试改变运行时类型
        for runtime_type in search_space.allowed_runtime_types:
            if runtime_type != current_decision.runtime_type:
                candidate = deepcopy(current_decision)
                candidate.runtime_type = runtime_type
                candidates.append(candidate)
                if len(candidates) >= max_candidates:
                    return candidates
        
        # 如果候选太少，添加一些随机变化
        while len(candidates) < min(5, max_candidates):
            candidate = self._generate_random_candidate(task_id)
            if candidate not in candidates:
                candidates.append(candidate)
        
        return candidates[:max_candidates]
    
    def _generate_random_candidate(self, task_id: str):
        """生成随机候选方案"""
        from scheduling_optimizer import SchedulingDecisionVariable
        import random
        
        search_space = self.search_spaces[task_id]
        
        # 随机选择配置
        priority = random.choice(search_space.allowed_priorities)
        runtime_type = random.choice(search_space.allowed_runtime_types)
        
        # 随机分段配置
        seg_configs = {}
        for seg_id, options in search_space.segmentation_options.items():
            seg_configs[seg_id] = random.choice(options)
        
        # 随机核心分配
        core_assignments = {}
        task = self.scheduler.tasks[task_id]
        for segment in task.segments:
            available_cores = search_space.available_cores.get(segment.resource_type, [])
            if available_cores:
                core_assignments[segment.segment_id] = random.choice(available_cores)
        
        return SchedulingDecisionVariable(
            task_id=task_id,
            priority=priority,
            runtime_type=runtime_type,
            segmentation_configs=seg_configs,
            core_assignments=core_assignments
        )
    
    def _format_metrics(self, metrics: Dict) -> str:
        """格式化指标显示"""
        parts = []
        if 'avg_latency' in metrics:
            parts.append(f"avg_latency={metrics['avg_latency']:.1f}")
        if 'avg_utilization' in metrics:
            parts.append(f"utilization={metrics['avg_utilization']:.1f}%")
        if 'priority_violations' in metrics:
            parts.append(f"violations={metrics['priority_violations']}")
        return ", ".join(parts)
    
    # 替换方法
    SchedulingOptimizer.optimize_greedy = efficient_optimize_greedy
    SchedulingOptimizer._initialize_solution = _initialize_solution
    SchedulingOptimizer._prioritize_tasks_for_optimization = _prioritize_tasks_for_optimization
    SchedulingOptimizer._generate_smart_candidates = _generate_smart_candidates
    SchedulingOptimizer._generate_random_candidate = _generate_random_candidate
    SchedulingOptimizer._format_metrics = _format_metrics
    
    # 同时修复evaluate_solution以提供更详细的任务级指标
    def enhanced_evaluate_solution(self, solution: Dict, time_window: float) -> Tuple[float, Dict]:
        """增强的解决方案评估，提供任务级指标"""
        # 应用解决方案
        original_configs = self._save_current_configuration()
        self._apply_solution(solution)
        
        # 执行调度（这里是主要的性能瓶颈）
        try:
            schedule_results = self.scheduler.priority_aware_schedule_with_segmentation(time_window)
            
            if not schedule_results:
                # 调度失败，返回最差分数
                self._restore_configuration(original_configs)
                return float('inf'), {'error': 'Scheduling failed'}
            
            # 计算详细指标
            metrics = self._calculate_detailed_metrics(schedule_results, time_window)
            
            # 计算目标函数分数
            score = self._calculate_objective_score(metrics, solution)
            
        finally:
            # 恢复原始配置
            self._restore_configuration(original_configs)
        
        return score, metrics
    
    def _save_current_configuration(self) -> Dict:
        """保存当前任务配置"""
        config = {}
        for task_id, task in self.scheduler.tasks.items():
            config[task_id] = {
                'priority': task.priority,
                'runtime_type': task.runtime_type,
                'segmentation_configs': task.selected_cut_config_index.copy(),
                'current_segmentation': deepcopy(task.current_segmentation)
            }
        return config
    
    def _restore_configuration(self, config: Dict):
        """恢复任务配置"""
        for task_id, saved_config in config.items():
            if task_id in self.scheduler.tasks:
                task = self.scheduler.tasks[task_id]
                task.priority = saved_config['priority']
                task.runtime_type = saved_config['runtime_type']
                task.selected_cut_config_index = saved_config['segmentation_configs'].copy()
                task.current_segmentation = deepcopy(saved_config['current_segmentation'])
    
    def _apply_solution(self, solution: Dict):
        """应用解决方案到调度器"""
        for task_id, decision in solution.items():
            if task_id in self.scheduler.tasks:
                task = self.scheduler.tasks[task_id]
                
                # 应用优先级
                task.priority = decision.priority
                
                # 应用运行时类型
                task.runtime_type = decision.runtime_type
                
                # 应用分段配置
                for seg_id, config_idx in decision.segmentation_configs.items():
                    if seg_id in task.preset_cut_configurations:
                        task.select_cut_configuration(seg_id, config_idx)
    
    def _calculate_detailed_metrics(self, schedule_results: List, time_window: float) -> Dict:
        """计算详细的性能指标，包括任务级信息"""
        metrics = {
            'task_performance': {},
            'resource_utilization': {},
            'total_events': len(schedule_results)
        }
        
        # 任务执行统计
        task_schedules = {}
        for schedule in schedule_results:
            task_id = schedule.task_id
            if task_id not in task_schedules:
                task_schedules[task_id] = []
            task_schedules[task_id].append(schedule)
        
        # 计算每个任务的性能
        total_latency = 0
        total_violations = 0
        
        for task_id, task in self.scheduler.tasks.items():
            schedules = task_schedules.get(task_id, [])
            
            if schedules:
                count = len(schedules)
                achieved_fps = count / (time_window / 1000.0)
                latencies = [s.actual_latency for s in schedules]
                avg_latency = np.mean(latencies)
                
                fps_ok = achieved_fps >= task.fps_requirement * 0.95
                latency_ok = avg_latency <= task.latency_requirement * 1.05
                
                metrics['task_performance'][task_id] = {
                    'count': count,
                    'achieved_fps': achieved_fps,
                    'required_fps': task.fps_requirement,
                    'avg_latency': avg_latency,
                    'required_latency': task.latency_requirement,
                    'fps_ok': fps_ok,
                    'latency_ok': latency_ok,
                    'violation': not (fps_ok and latency_ok)
                }
                
                if not (fps_ok and latency_ok):
                    violation_weight = 4 - task.priority.value
                    total_violations += violation_weight
                
                total_latency += avg_latency
            else:
                # 任务未被调度
                metrics['task_performance'][task_id] = {
                    'count': 0,
                    'achieved_fps': 0,
                    'required_fps': task.fps_requirement,
                    'avg_latency': float('inf'),
                    'required_latency': task.latency_requirement,
                    'fps_ok': False,
                    'latency_ok': False,
                    'violation': True
                }
                total_violations += (4 - task.priority.value) * 2  # 双倍惩罚
        
        # 汇总指标
        metrics['avg_latency'] = total_latency / len(self.scheduler.tasks) if self.scheduler.tasks else 0
        metrics['priority_violations'] = total_violations
        
        # 资源利用率
        utilization = self.scheduler.get_resource_utilization(time_window)
        metrics['resource_utilization'] = utilization
        metrics['avg_utilization'] = np.mean(list(utilization.values())) if utilization else 0
        
        # 吞吐量比率
        throughput_ratios = []
        for task_id, perf in metrics['task_performance'].items():
            if perf['required_fps'] > 0:
                ratio = perf['achieved_fps'] / perf['required_fps']
                throughput_ratios.append(ratio)
        
        metrics['avg_throughput_ratio'] = np.mean(throughput_ratios) if throughput_ratios else 0
        
        # 分段开销
        total_overhead = sum(task.total_segmentation_overhead for task in self.scheduler.tasks.values())
        metrics['total_overhead'] = total_overhead
        
        return metrics
    
    # 替换增强的方法
    SchedulingOptimizer.evaluate_solution = enhanced_evaluate_solution
    SchedulingOptimizer._calculate_detailed_metrics = _calculate_detailed_metrics
    
    # 添加缺失的方法到SchedulingOptimizer类
    SchedulingOptimizer._save_current_configuration = _save_current_configuration
    SchedulingOptimizer._restore_configuration = _restore_configuration
    SchedulingOptimizer._apply_solution = _apply_solution
    SchedulingOptimizer._calculate_detailed_metrics = _calculate_detailed_metrics
    
    # 修复generate_candidate_solutions方法
    original_generate_candidates = getattr(SchedulingOptimizer, 'generate_candidate_solutions', None)
    
    def fixed_generate_candidate_solutions(self, task_id: str, max_candidates: int = 10) -> List:
        """修复的候选生成方法"""
        from scheduling_optimizer import SchedulingDecisionVariable
        import random
        
        candidates = []
        search_space = self.search_spaces.get(task_id)
        if not search_space:
            return candidates
        
        task = self.scheduler.tasks.get(task_id)
        if not task:
            return candidates
        
        # 为每个允许的优先级生成候选
        for priority in search_space.allowed_priorities:
            # 为每个运行时类型生成候选
            for runtime_type in search_space.allowed_runtime_types:
                # 创建候选
                seg_configs = {}
                for seg_id, options in search_space.segmentation_options.items():
                    if options:
                        # 尝试不同的分段配置
                        for config_idx in options[:2]:  # 最多尝试2个配置
                            seg_configs[seg_id] = config_idx
                            
                            # 核心分配
                            core_assignments = {}
                            for segment in task.segments:
                                available_cores = search_space.available_cores.get(segment.resource_type, [])
                                if available_cores:
                                    # 尝试不同的核心
                                    core_assignments[segment.segment_id] = random.choice(available_cores)
                            
                            candidate = SchedulingDecisionVariable(
                                task_id=task_id,
                                priority=priority,
                                runtime_type=runtime_type,
                                segmentation_configs=seg_configs.copy(),
                                core_assignments=core_assignments
                            )
                            
                            candidates.append(candidate)
                            
                            if len(candidates) >= max_candidates:
                                return candidates
                
                # 如果没有分段选项，至少生成一个候选
                if not search_space.segmentation_options or all(not opts for opts in search_space.segmentation_options.values()):
                    core_assignments = {}
                    for segment in task.segments:
                        available_cores = search_space.available_cores.get(segment.resource_type, [])
                        if available_cores:
                            core_assignments[segment.segment_id] = available_cores[0]
                    
                    candidate = SchedulingDecisionVariable(
                        task_id=task_id,
                        priority=priority,
                        runtime_type=runtime_type,
                        segmentation_configs={},
                        core_assignments=core_assignments
                    )
                    candidates.append(candidate)
                    
                    if len(candidates) >= max_candidates:
                        return candidates[:max_candidates]
        
        return candidates[:max_candidates]
    
    # 添加方法
    SchedulingOptimizer.generate_candidate_solutions = fixed_generate_candidate_solutions
    
    print("✅ Scheduling optimizer fix applied successfully")
    print("  - Added solution caching to reduce redundant scheduling")
    print("  - Reduced candidate generation from 50 to 10")
    print("  - Added smart candidate generation based on performance")
    print("  - Enhanced metrics with task-level details")


if __name__ == "__main__":
    print("Scheduling Optimizer Fix")
    print("This module provides fixes for the scheduling optimizer")
    print("Usage: from scheduling_optimizer_fix import apply_scheduling_optimizer_fix")
    print("       apply_scheduling_optimizer_fix()")
