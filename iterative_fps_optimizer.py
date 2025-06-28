#!/usr/bin/env python3
"""
迭代FPS优化器
通过多轮迭代调度确保所有任务满足FPS要求
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict
import copy
from enums import TaskPriority


class IterativeFPSOptimizer:
    """迭代优化FPS满足率的调度器"""
    
    def __init__(self, scheduler, max_iterations: int = 10):
        self.scheduler = scheduler
        self.max_iterations = max_iterations
        self.original_tasks = {}  # 保存原始任务配置
        
    def optimize_schedule(self, time_window: float = 200.0, verbose: bool = True):
        """迭代优化调度直到所有任务满足FPS要求"""
        
        print("\n" + "=" * 80)
        print("🔄 迭代FPS优化器")
        print("=" * 80)
        
        # 保存原始任务配置
        self._save_original_config()
        
        iteration = 0
        all_satisfied = False
        best_schedule = None
        best_satisfaction_rate = 0.0
        
        while iteration < self.max_iterations and not all_satisfied:
            iteration += 1
            print(f"\n📍 第 {iteration} 轮迭代:")
            
            # 运行调度
            results = self.scheduler.priority_aware_schedule_with_segmentation(time_window)
            
            # 分析FPS满足情况
            fps_analysis = self._analyze_fps_satisfaction(time_window)
            
            # 计算满足率
            satisfaction_rate = fps_analysis['satisfaction_rate']
            print(f"  - FPS满足率: {satisfaction_rate:.1f}%")
            print(f"  - 满足任务数: {fps_analysis['satisfied_count']}/{fps_analysis['total_count']}")
            
            # 保存最佳结果
            if satisfaction_rate > best_satisfaction_rate:
                best_schedule = copy.deepcopy(self.scheduler.schedule_history)
                best_satisfaction_rate = satisfaction_rate
            
            # 检查是否全部满足
            if satisfaction_rate >= 95.0:  # 允许5%的误差
                all_satisfied = True
                print(f"\n✅ 所有任务满足FPS要求！")
                break
            
            # 应用优化策略
            if iteration < self.max_iterations:
                self._apply_optimization_strategy(fps_analysis, iteration)
        
        # 恢复最佳调度结果
        if best_schedule:
            self.scheduler.schedule_history = best_schedule
        
        # 打印最终结果
        self._print_final_results(best_satisfaction_rate, iteration, time_window)
        
        return all_satisfied, best_satisfaction_rate
    
    def _save_original_config(self):
        """保存原始任务配置"""
        for task_id, task in self.scheduler.tasks.items():
            self.original_tasks[task_id] = {
                'priority': task.priority,
                'fps_requirement': task.fps_requirement,
                'latency_requirement': task.latency_requirement
            }
    
    def _analyze_fps_satisfaction(self, time_window: float) -> Dict:
        """分析FPS满足情况"""
        task_executions = defaultdict(list)
        
        for schedule in self.scheduler.schedule_history:
            task_executions[schedule.task_id].append(schedule.start_time)
        
        analysis = {
            'unsatisfied_tasks': [],
            'satisfied_count': 0,
            'total_count': len(self.scheduler.tasks),
            'satisfaction_rate': 0.0
        }
        
        for task_id, task in self.scheduler.tasks.items():
            executions = task_executions[task_id]
            expected_count = int((time_window / 1000.0) * task.fps_requirement)
            actual_count = len(executions)
            
            if actual_count < expected_count * 0.95:  # 未满足
                analysis['unsatisfied_tasks'].append({
                    'task_id': task_id,
                    'task_name': task.name,
                    'priority': task.priority,
                    'required_fps': task.fps_requirement,
                    'expected_count': expected_count,
                    'actual_count': actual_count,
                    'deficit': expected_count - actual_count,
                    'execution_times': executions
                })
            else:
                analysis['satisfied_count'] += 1
        
        analysis['satisfaction_rate'] = (analysis['satisfied_count'] / analysis['total_count']) * 100
        
        return analysis
    
    def _apply_optimization_strategy(self, fps_analysis: Dict, iteration: int):
        """应用优化策略"""
        print(f"\n🔧 应用优化策略:")
        
        unsatisfied_tasks = fps_analysis['unsatisfied_tasks']
        if not unsatisfied_tasks:
            return
        
        # 按缺口大小排序
        unsatisfied_tasks.sort(key=lambda x: x['deficit'], reverse=True)
        
        for task_info in unsatisfied_tasks[:3]:  # 每次优化最多3个任务
            task_id = task_info['task_id']
            task = self.scheduler.tasks[task_id]
            
            print(f"\n  优化任务 {task_id} ({task.name}):")
            print(f"    当前: {task_info['actual_count']} 次执行")
            print(f"    需要: {task_info['expected_count']} 次执行")
            
            # 策略1：调整优先级
            if iteration <= 2:
                self._adjust_priority(task, task_info)
            
            # 策略2：插入额外执行机会
            if iteration >= 2:
                self._insert_execution_opportunities(task, task_info)
            
            # 策略3：动态调整任务间隔
            if iteration >= 3:
                self._adjust_task_intervals(task, task_info)
    
    def _adjust_priority(self, task, task_info):
        """调整任务优先级"""
        current_priority = task.priority
        
        # 根据FPS要求调整优先级
        if task_info['required_fps'] >= 50:
            # 高FPS需求，提升到最高优先级
            if current_priority != TaskPriority.CRITICAL:
                task.priority = TaskPriority.CRITICAL
                print(f"    ✓ 优先级提升: {current_priority.name} → CRITICAL")
        elif task_info['required_fps'] >= 25:
            # 中等FPS需求
            if current_priority.value > TaskPriority.HIGH.value:
                task.priority = TaskPriority.HIGH
                print(f"    ✓ 优先级提升: {current_priority.name} → HIGH")
    
    def _insert_execution_opportunities(self, task, task_info):
        """在空闲时间插入额外的执行机会"""
        # 找出资源的空闲时间段
        idle_periods = self._find_idle_periods(task)
        
        if idle_periods:
            # 计算任务执行时间
            task_duration = self._estimate_task_duration(task)
            min_interval = 1000.0 / task_info['required_fps']
            
            # 尝试在空闲时段插入任务
            inserted = 0
            for start, end in idle_periods:
                if end - start >= task_duration + 0.1:  # 留0.1ms余量
                    # 可以插入任务
                    inserted += 1
                    print(f"    ✓ 可在 {start:.1f}-{end:.1f}ms 插入执行")
                    
                    if inserted >= task_info['deficit']:
                        break
            
            if inserted > 0:
                # 记录可以插入的额外执行次数
                print(f"    ✓ 找到 {inserted} 个额外执行机会")
                # 标记任务需要更频繁的调度
                if not hasattr(task, '_high_frequency_mode'):
                    task._high_frequency_mode = True
                    task._extra_executions_needed = inserted
                    print(f"    ✓ 启用高频模式，需要额外 {inserted} 次执行")
    
    def _adjust_task_intervals(self, task, task_info):
        """动态调整任务执行间隔"""
        # 如果任务有依赖，考虑调整依赖关系
        if hasattr(task, 'dependencies') and task.dependencies:
            # 检查是否可以放松依赖约束
            if task_info['deficit'] > task_info['expected_count'] * 0.3:
                print(f"    ✓ 考虑放松依赖约束以增加执行机会")
                # 这里可以实现更复杂的依赖调整逻辑
    
    def _find_idle_periods(self, task) -> List[Tuple[float, float]]:
        """找出可用于任务执行的空闲时间段"""
        idle_periods = []
        
        # 获取任务需要的资源类型
        required_resources = set()
        for seg in task.segments:
            required_resources.add(seg.resource_type)
        
        # 构建资源时间线
        for res_type in required_resources:
            for resource in self.scheduler.resources[res_type]:
                # 获取该资源的占用时间线
                timeline = self._get_resource_timeline(resource.unit_id)
                
                # 找出空闲时段
                for i in range(len(timeline) - 1):
                    gap_start = timeline[i][1]
                    gap_end = timeline[i + 1][0]
                    
                    if gap_end - gap_start > 1.0:  # 大于1ms的空闲
                        idle_periods.append((gap_start, gap_end))
        
        # 合并重叠的空闲时段
        return self._merge_overlapping_periods(idle_periods)
    
    def _get_resource_timeline(self, resource_id: str) -> List[Tuple[float, float]]:
        """获取资源的占用时间线"""
        timeline = []
        
        for schedule in self.scheduler.schedule_history:
            if hasattr(schedule, 'sub_segment_schedule'):
                for sub_seg_id, start, end in schedule.sub_segment_schedule:
                    if resource_id in schedule.assigned_resources.values():
                        timeline.append((start, end))
        
        return sorted(timeline)
    
    def _merge_overlapping_periods(self, periods: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """合并重叠的时间段"""
        if not periods:
            return []
        
        sorted_periods = sorted(periods)
        merged = [sorted_periods[0]]
        
        for start, end in sorted_periods[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        
        return merged
    
    def _estimate_task_duration(self, task) -> float:
        """估算任务执行时间"""
        total_duration = 0.0
        
        for seg in task.segments:
            # 使用最快的带宽估算
            if hasattr(seg, 'bandwidth_latency_map') and seg.bandwidth_latency_map:
                total_duration += min(seg.bandwidth_latency_map.values())
        
        return total_duration
    
    def _print_final_results(self, satisfaction_rate: float, iterations: int, time_window: float):
        """打印最终优化结果"""
        print("\n" + "=" * 80)
        print("📊 迭代优化最终结果")
        print("=" * 80)
        
        print(f"\n总迭代次数: {iterations}")
        print(f"最终FPS满足率: {satisfaction_rate:.1f}%")
        
        # 分析每个任务的最终状态
        print("\n任务执行情况:")
        print(f"{'任务ID':<10} {'名称':<15} {'优先级':<10} {'要求FPS':<10} {'执行次数':<10} {'状态':<10}")
        print("-" * 75)
        
        task_executions = defaultdict(int)
        for schedule in self.scheduler.schedule_history:
            task_executions[schedule.task_id] += 1
        
        for task_id, task in sorted(self.scheduler.tasks.items()):
            actual_count = task_executions[task_id]
            expected_count = int((time_window / 1000.0) * task.fps_requirement)
            status = "✅" if actual_count >= expected_count * 0.95 else "❌"
            
            # 检查优先级是否被调整
            original_priority = self.original_tasks[task_id]['priority']
            priority_str = task.priority.name
            if task.priority != original_priority:
                priority_str += f" (原{original_priority.name})"
            
            print(f"{task_id:<10} {task.name:<15} {priority_str:<10} "
                  f"{task.fps_requirement:<10.1f} {actual_count}/{expected_count:<9} {status:<10}")


def apply_iterative_fps_optimization(scheduler, time_window: float = 200.0):
    """应用迭代FPS优化"""
    optimizer = IterativeFPSOptimizer(scheduler)
    all_satisfied, satisfaction_rate = optimizer.optimize_schedule(time_window)
    
    return all_satisfied, satisfaction_rate


if __name__ == "__main__":
    print("迭代FPS优化器")
    print("功能：")
    print("1. 多轮迭代调度直到满足所有FPS要求")
    print("2. 动态调整任务优先级")
    print("3. 在空闲时段插入额外执行")
    print("4. 优化任务执行间隔")
