#!/usr/bin/env python3
"""
Scheduling Optimizer - treats priority, runtime type, segmentation config, and core assignment as variables
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from itertools import product
from collections import defaultdict
import numpy as np

from enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from models import ResourceUnit, TaskScheduleInfo
from task import NNTask
from scheduler import MultiResourceScheduler


@dataclass
class SchedulingDecisionVariable:
    """Represents all decision variables for a single task"""
    task_id: str
    priority: TaskPriority
    runtime_type: RuntimeType
    segmentation_configs: Dict[str, int]  # {segment_id: config_index}
    core_assignments: Dict[str, str]  # {segment_id: core_id}
    
    def __hash__(self):
        # Make it hashable for use in sets/dicts
        seg_config_str = str(sorted(self.segmentation_configs.items()))
        core_assign_str = str(sorted(self.core_assignments.items()))
        return hash((self.task_id, self.priority, self.runtime_type, seg_config_str, core_assign_str))


@dataclass
class SchedulingSearchSpace:
    """Defines the search space for each task"""
    task_id: str
    allowed_priorities: List[TaskPriority]
    allowed_runtime_types: List[RuntimeType]
    segmentation_options: Dict[str, List[int]]  # {segment_id: [config_indices]}
    available_cores: Dict[ResourceType, List[str]]  # {resource_type: [core_ids]}


@dataclass
class SchedulingObjective:
    """Objective function components and weights"""
    latency_weight: float = 1.0
    throughput_weight: float = 1.0
    utilization_weight: float = 0.5
    priority_violation_weight: float = 2.0  # Penalty for not meeting priority requirements
    overhead_weight: float = 0.3


class SchedulingOptimizer:
    """Optimizes all scheduling variables jointly"""
    
    def __init__(self, scheduler: MultiResourceScheduler):
        self.scheduler = scheduler
        self.search_spaces: Dict[str, SchedulingSearchSpace] = {}
        self.objective = SchedulingObjective()
        self.best_solution: Optional[Dict[str, SchedulingDecisionVariable]] = None
        self.best_score: float = float('inf')
        
    def define_search_space(self, task_id: str, search_space: SchedulingSearchSpace):
        """Define the search space for a task"""
        self.search_spaces[task_id] = search_space
        
    def define_search_space_from_task(self, task: NNTask, 
                                    allowed_priorities: Optional[List[TaskPriority]] = None,
                                    allowed_runtime_types: Optional[List[RuntimeType]] = None):
        """Define search space from task properties"""
        # Default allowed values if not specified
        if allowed_priorities is None:
            allowed_priorities = list(TaskPriority)
        if allowed_runtime_types is None:
            allowed_runtime_types = list(RuntimeType)
            
        # Get segmentation options from task
        segmentation_options = {}
        for segment in task.segments:
            if segment.segment_id in task.preset_cut_configurations:
                num_configs = len(task.preset_cut_configurations[segment.segment_id])
                segmentation_options[segment.segment_id] = list(range(num_configs))
            else:
                # If no preset configs, assume binary choice (cut or no cut)
                segmentation_options[segment.segment_id] = [0, 1]  # 0: no cuts, 1: all cuts
        
        # Get available cores by resource type
        available_cores = {}
        for segment in task.segments:
            res_type = segment.resource_type
            if res_type not in available_cores:
                available_cores[res_type] = [r.unit_id for r in self.scheduler.resources[res_type]]
        
        search_space = SchedulingSearchSpace(
            task_id=task.task_id,
            allowed_priorities=allowed_priorities,
            allowed_runtime_types=allowed_runtime_types,
            segmentation_options=segmentation_options,
            available_cores=available_cores
        )
        
        self.search_spaces[task.task_id] = search_space
        
    def generate_candidate_solutions(self, task_id: str, max_candidates: int = 100) -> List[SchedulingDecisionVariable]:
        """Generate candidate solutions for a task"""
        search_space = self.search_spaces[task_id]
        task = self.scheduler.tasks[task_id]
        candidates = []
        
        # Get all possible combinations
        all_combinations = []
        
        # Priority options
        for priority in search_space.allowed_priorities:
            # Runtime type options
            for runtime_type in search_space.allowed_runtime_types:
                # Segmentation config combinations
                seg_config_options = []
                for seg_id, config_indices in search_space.segmentation_options.items():
                    seg_config_options.append([(seg_id, idx) for idx in config_indices])
                
                if seg_config_options:
                    for seg_configs in product(*seg_config_options):
                        seg_config_dict = dict(seg_configs)
                        
                        # Core assignment combinations
                        # Group segments by resource type
                        segments_by_type = defaultdict(list)
                        for segment in task.segments:
                            segments_by_type[segment.resource_type].append(segment.segment_id)
                        
                        # Generate core assignments
                        core_assignment_options = []
                        for res_type, segment_ids in segments_by_type.items():
                            available_cores = search_space.available_cores.get(res_type, [])
                            if available_cores:
                                # For each segment, choose a core
                                for seg_id in segment_ids:
                                    core_assignment_options.append([(seg_id, core) for core in available_cores])
                        
                        if core_assignment_options:
                            for core_assignments in product(*core_assignment_options):
                                core_assign_dict = dict(core_assignments)
                                
                                # Create candidate
                                candidate = SchedulingDecisionVariable(
                                    task_id=task_id,
                                    priority=priority,
                                    runtime_type=runtime_type,
                                    segmentation_configs=seg_config_dict,
                                    core_assignments=core_assign_dict
                                )
                                all_combinations.append(candidate)
        
        # Sample if too many combinations
        if len(all_combinations) > max_candidates:
            # Use intelligent sampling based on heuristics
            candidates = self._intelligent_sampling(all_combinations, max_candidates)
        else:
            candidates = all_combinations
            
        return candidates
    
    def _intelligent_sampling(self, all_candidates: List[SchedulingDecisionVariable], 
                            max_candidates: int) -> List[SchedulingDecisionVariable]:
        """Intelligently sample from candidates based on heuristics"""
        # Always include some baseline configurations
        baseline_candidates = []
        
        for candidate in all_candidates:
            # Include high priority configurations
            if candidate.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
                baseline_candidates.append(candidate)
                if len(baseline_candidates) >= max_candidates // 3:
                    break
        
        # Random sample from the rest
        remaining = max_candidates - len(baseline_candidates)
        other_candidates = [c for c in all_candidates if c not in baseline_candidates]
        
        if len(other_candidates) > remaining:
            sampled = np.random.choice(other_candidates, size=remaining, replace=False)
            return baseline_candidates + list(sampled)
        else:
            return baseline_candidates + other_candidates
    
    def evaluate_solution(self, solution: Dict[str, SchedulingDecisionVariable], 
                         time_window: float = 1000.0) -> Tuple[float, Dict[str, float]]:
        """Evaluate a complete solution (all tasks)"""
        # Apply the solution to tasks
        for task_id, decision in solution.items():
            task = self.scheduler.tasks[task_id]
            
            # Apply priority
            task.priority = decision.priority
            
            # Apply runtime type
            task.runtime_type = decision.runtime_type
            
            # Apply segmentation configurations
            for seg_id, config_idx in decision.segmentation_configs.items():
                if seg_id in task.preset_cut_configurations:
                    task.select_cut_configuration(seg_id, config_idx)
        
        # Run scheduling with fixed core assignments
        schedule_results = self._run_scheduling_with_core_assignments(solution, time_window)
        
        # Calculate metrics
        metrics = self._calculate_metrics(schedule_results, time_window)
        
        # Calculate objective score
        score = self._calculate_objective_score(metrics, solution)
        
        return score, metrics
    
    def _run_scheduling_with_core_assignments(self, solution: Dict[str, SchedulingDecisionVariable], 
                                            time_window: float) -> List[TaskScheduleInfo]:
        """Run scheduling with specific core assignments"""
        # This is a simplified version - in practice, you'd modify the scheduler
        # to respect the core assignments in the solution
        
        # For now, we'll use the regular scheduler and estimate impact
        return self.scheduler.priority_aware_schedule_with_segmentation(time_window)
    
    def _calculate_metrics(self, schedule_results: List[TaskScheduleInfo], 
                          time_window: float) -> Dict[str, float]:
        """Calculate performance metrics"""
        metrics = {}
        
        if not schedule_results:
            return metrics
        
        # Latency metrics
        latencies = [s.actual_latency for s in schedule_results]
        metrics['avg_latency'] = np.mean(latencies) if latencies else 0
        metrics['max_latency'] = np.max(latencies) if latencies else 0
        
        # Throughput metrics
        task_counts = defaultdict(int)
        for schedule in schedule_results:
            task_counts[schedule.task_id] += 1
        
        throughputs = []
        for task_id, count in task_counts.items():
            task = self.scheduler.tasks[task_id]
            achieved_fps = count / (time_window / 1000.0)
            required_fps = task.fps_requirement
            throughputs.append(achieved_fps / required_fps if required_fps > 0 else 1.0)
        
        metrics['avg_throughput_ratio'] = np.mean(throughputs) if throughputs else 0
        
        # Utilization metrics
        utilization = self.scheduler.get_resource_utilization(time_window)
        metrics['avg_utilization'] = np.mean(list(utilization.values())) if utilization else 0
        
        # Overhead metrics
        total_overhead = sum(s.segmentation_overhead for s in schedule_results)
        metrics['total_overhead'] = total_overhead
        
        # Priority violation metrics
        priority_violations = 0
        for task_id, task in self.scheduler.tasks.items():
            task_schedules = [s for s in schedule_results if s.task_id == task_id]
            if task_schedules:
                avg_latency = np.mean([s.actual_latency for s in task_schedules])
                if avg_latency > task.latency_requirement:
                    # Weight violation by priority
                    violation_weight = 4 - task.priority.value  # Higher weight for higher priority
                    priority_violations += violation_weight
        
        metrics['priority_violations'] = priority_violations
        
        return metrics
    
    def _calculate_objective_score(self, metrics: Dict[str, float], 
                                 solution: Dict[str, SchedulingDecisionVariable]) -> float:
        """Calculate objective function score (lower is better)"""
        score = 0.0
        
        # Latency component
        score += self.objective.latency_weight * metrics.get('avg_latency', 0)
        
        # Throughput component (inverse, as higher is better)
        throughput_ratio = metrics.get('avg_throughput_ratio', 0)
        score += self.objective.throughput_weight * (1.0 / (throughput_ratio + 0.001))
        
        # Utilization component (inverse, as higher is better)
        utilization = metrics.get('avg_utilization', 0)
        score += self.objective.utilization_weight * (100 - utilization)
        
        # Priority violation penalty
        score += self.objective.priority_violation_weight * metrics.get('priority_violations', 0)
        
        # Overhead penalty
        score += self.objective.overhead_weight * metrics.get('total_overhead', 0)
        
        return score
    
    def optimize_greedy(self, time_window: float = 1000.0, iterations: int = 10) -> Dict[str, SchedulingDecisionVariable]:
        """Greedy optimization - optimize one task at a time"""
        print("\n=== Starting Greedy Optimization ===")
        
        # Initialize with current task configurations
        current_solution = {}
        for task_id in self.search_spaces.keys():
            task = self.scheduler.tasks[task_id]
            
            # Create initial decision variable
            seg_configs = {}
            for seg_id in task.current_segmentation.keys():
                seg_configs[seg_id] = task.selected_cut_config_index.get(seg_id, 0)
            
            # Initial core assignments (just use first available)
            core_assignments = {}
            for segment in task.segments:
                available_cores = self.search_spaces[task_id].available_cores.get(segment.resource_type, [])
                if available_cores:
                    core_assignments[segment.segment_id] = available_cores[0]
            
            current_solution[task_id] = SchedulingDecisionVariable(
                task_id=task_id,
                priority=task.priority,
                runtime_type=task.runtime_type,
                segmentation_configs=seg_configs,
                core_assignments=core_assignments
            )
        
        # Evaluate initial solution
        best_score, best_metrics = self.evaluate_solution(current_solution, time_window)
        best_solution = current_solution.copy()
        
        print(f"Initial score: {best_score:.2f}")
        print(f"Initial metrics: {best_metrics}")
        
        # Greedy improvement
        for iteration in range(iterations):
            print(f"\nIteration {iteration + 1}/{iterations}")
            improved = False
            
            # Try to improve each task
            for task_id in self.search_spaces.keys():
                print(f"  Optimizing task {task_id}...")
                
                # Generate candidates for this task
                candidates = self.generate_candidate_solutions(task_id, max_candidates=50)
                
                # Try each candidate
                for candidate in candidates:
                    # Create new solution with this candidate
                    new_solution = best_solution.copy()
                    new_solution[task_id] = candidate
                    
                    # Evaluate
                    score, metrics = self.evaluate_solution(new_solution, time_window)
                    
                    # Check if improved
                    if score < best_score:
                        best_score = score
                        best_solution = new_solution
                        best_metrics = metrics
                        improved = True
                        print(f"    Found improvement! New score: {best_score:.2f}")
                        break
            
            if not improved:
                print("  No improvement found in this iteration")
                break
        
        print(f"\nOptimization complete!")
        print(f"Final score: {best_score:.2f}")
        print(f"Final metrics: {best_metrics}")
        
        self.best_solution = best_solution
        self.best_score = best_score
        
        return best_solution
    
    def optimize_genetic(self, population_size: int = 50, generations: int = 20, 
                        time_window: float = 1000.0) -> Dict[str, SchedulingDecisionVariable]:
        """Genetic algorithm optimization"""
        print("\n=== Starting Genetic Algorithm Optimization ===")
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = {}
            for task_id in self.search_spaces.keys():
                candidates = self.generate_candidate_solutions(task_id, max_candidates=10)
                if candidates:
                    individual[task_id] = np.random.choice(candidates)
            population.append(individual)
        
        # Evolution
        for generation in range(generations):
            print(f"\nGeneration {generation + 1}/{generations}")
            
            # Evaluate population
            scores = []
            for individual in population:
                score, _ = self.evaluate_solution(individual, time_window)
                scores.append(score)
            
            # Sort by fitness (lower is better)
            sorted_indices = np.argsort(scores)
            population = [population[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]
            
            print(f"  Best score: {scores[0]:.2f}")
            print(f"  Worst score: {scores[-1]:.2f}")
            print(f"  Average score: {np.mean(scores):.2f}")
            
            # Keep best solutions
            elite_size = population_size // 4
            new_population = population[:elite_size]
            
            # Crossover and mutation
            while len(new_population) < population_size:
                # Select parents
                parent1 = self._tournament_selection(population, scores)
                parent2 = self._tournament_selection(population, scores)
                
                # Crossover
                child = self._crossover(parent1, parent2)
                
                # Mutation
                if np.random.random() < 0.1:  # 10% mutation rate
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        # Return best solution
        best_individual = population[0]
        best_score, best_metrics = self.evaluate_solution(best_individual, time_window)
        
        print(f"\nOptimization complete!")
        print(f"Final score: {best_score:.2f}")
        print(f"Final metrics: {best_metrics}")
        
        self.best_solution = best_individual
        self.best_score = best_score
        
        return best_individual
    
    def _tournament_selection(self, population: List[Dict], scores: List[float], 
                            tournament_size: int = 3) -> Dict:
        """Tournament selection"""
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_scores = [scores[i] for i in indices]
        winner_idx = indices[np.argmin(tournament_scores)]
        return population[winner_idx]
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover two solutions"""
        child = {}
        for task_id in parent1.keys():
            # Randomly choose from parents
            if np.random.random() < 0.5:
                child[task_id] = parent1[task_id]
            else:
                child[task_id] = parent2[task_id]
        return child
    
    def _mutate(self, individual: Dict) -> Dict:
        """Mutate a solution"""
        mutated = individual.copy()
        
        # Randomly select a task to mutate
        task_id = np.random.choice(list(individual.keys()))
        
        # Generate new candidate for this task
        candidates = self.generate_candidate_solutions(task_id, max_candidates=5)
        if candidates:
            mutated[task_id] = np.random.choice(candidates)
        
        return mutated
    
    def print_solution(self, solution: Dict[str, SchedulingDecisionVariable]):
        """Print a solution in readable format"""
        print("\n=== Optimized Solution ===")
        print(f"{'Task':<8} {'Priority':<12} {'Runtime':<15} {'Segmentation':<30} {'Core Assignment'}")
        print("-" * 80)
        
        for task_id, decision in sorted(solution.items()):
            seg_str = ", ".join([f"{seg}:cfg{idx}" for seg, idx in decision.segmentation_configs.items()])
            core_str = ", ".join([f"{seg}→{core}" for seg, core in decision.core_assignments.items()])
            
            print(f"{task_id:<8} {decision.priority.name:<12} {decision.runtime_type.value:<15} {seg_str:<30} {core_str}")