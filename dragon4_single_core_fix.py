#!/usr/bin/env python3
"""
Dragon4单核系统修复方案
处理复杂任务(如T1)在资源受限环境下的调度
"""

from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from models import TaskScheduleInfo
from enums import TaskPriority, ResourceType
import copy


def apply_single_core_dragon4_fix(scheduler):
    """应用单核系统专用修复"""
    
    print("🔧 应用单核Dragon4修复...")
    
    # 1. 分析系统资源配置
    analyze_system_resources(scheduler)
    
    # 2. 优化任务调度策略
    fix_single_core_scheduling(scheduler)
    
    # 3. 添加增强的验证
    add_enhanced_validation(scheduler)
    
    print("✅ 单核Dragon4修复已应用")


def analyze_system_resources(scheduler):
    """分析系统资源配置"""
    
    print("\n📊 系统资源分析:")
    
    # 统计资源
    for res_type, resources in scheduler.resources.items():
        print(f"  {res_type.name}: {len(resources)} 个资源")
        for res in resources:
            print(f"    - {res.unit_id}: bandwidth={res.bandwidth}")
    
    # 分析任务复杂度
    print("\n任务复杂度分析:")
    for task_id, task in scheduler.tasks.items():
        segment_count = len(task.segments)
        npu_segments = sum(1 for s in task.segments if s.resource_type == ResourceType.NPU)
        dsp_segments = sum(1 for s in task.segments if s.resource_type == ResourceType.DSP)
        
        if segment_count > 1:
            print(f"  {task_id}: {segment_count} segments (NPU:{npu_segments}, DSP:{dsp_segments})")
            
            # 计算总执行时间
            total_time = 0
            for seg in task.segments:
                # 使用40的bandwidth估算
                duration = seg.get_duration(40) if 40 in seg.duration_table else 0
                total_time += duration
            
            print(f"    预估总执行时间: {total_time:.1f}ms")


def fix_single_core_scheduling(scheduler):
    """修复单核系统的调度逻辑"""
    
    class SingleCoreScheduler:
        def __init__(self):
            # 精确的资源时间线管理
            self.resource_timelines = defaultdict(list)  # {resource_id: [(start, end, task_id, info)]}
            self.task_execution_times = {}  # {task_id: [exec_times]}
            
        def find_feasible_time(self, resource_id, duration, earliest_start):
            """找到可行的调度时间"""
            
            timeline = sorted(self.resource_timelines[resource_id], key=lambda x: x[0])
            
            # 尝试在earliest_start开始
            current = earliest_start
            
            while current < earliest_start + 200:  # 最多向后搜索200ms
                # 检查[current, current+duration]是否可用
                conflict = False
                
                for start, end, _, _ in timeline:
                    if not (current + duration <= start or current >= end):
                        # 有冲突，跳到这个占用之后
                        current = end
                        conflict = True
                        break
                
                if not conflict:
                    return current
            
            return None
            
        def reserve_time_slot(self, resource_id, start, end, task_id, info=""):
            """预留时间槽"""
            self.resource_timelines[resource_id].append((start, end, task_id, info))
            self.resource_timelines[resource_id].sort(key=lambda x: x[0])
            
        def can_schedule_task(self, task, current_time):
            """检查任务是否可以调度"""
            
            # 检查FPS约束
            if task.task_id in self.task_execution_times:
                last_times = self.task_execution_times[task.task_id]
                if last_times:
                    min_interval = 1000.0 / task.fps_requirement
                    if current_time - last_times[-1] < min_interval - 0.1:
                        return False, "FPS约束"
            
            # 检查依赖
            for dep_id in task.dependencies:
                if dep_id not in self.task_execution_times or not self.task_execution_times[dep_id]:
                    return False, f"依赖{dep_id}未执行"
            
            return True, ""
            
        def schedule_complex_task(self, task, scheduler, current_time):
            """调度复杂任务（如T1）"""
            
            # 收集所有segments的调度计划
            segment_plans = []
            task_start = current_time
            
            # 逐个处理segment
            for i, segment in enumerate(task.segments):
                # 找到合适的资源
                best_resource = None
                best_duration = float('inf')
                
                for resource in scheduler.resources.get(segment.resource_type, []):
                    if resource.bandwidth in segment.duration_table:
                        duration = segment.duration_table[resource.bandwidth]
                        if duration < best_duration:
                            best_duration = duration
                            best_resource = resource
                
                if not best_resource:
                    return None
                
                # 计算segment应该开始的时间
                seg_ideal_start = task_start + segment.start_time
                
                # 找到实际可行的时间
                actual_start = self.find_feasible_time(
                    best_resource.unit_id,
                    best_duration,
                    seg_ideal_start
                )
                
                if actual_start is None:
                    return None
                
                # 调整任务开始时间以保持segment之间的相对时序
                if actual_start > seg_ideal_start:
                    delay = actual_start - seg_ideal_start
                    task_start += delay
                    # 重新计算之前的segments
                    for j, prev_plan in enumerate(segment_plans):
                        prev_plan['start'] += delay
                        prev_plan['end'] += delay
                
                segment_plans.append({
                    'segment_idx': i,
                    'resource': best_resource,
                    'start': task_start + segment.start_time,
                    'end': task_start + segment.start_time + best_duration,
                    'duration': best_duration
                })
            
            # 构建完整的调度计划
            if segment_plans:
                return {
                    'task_start': task_start,
                    'task_end': max(p['end'] for p in segment_plans),
                    'segments': segment_plans
                }
            
            return None
    
    scheduler._single_core_scheduler = SingleCoreScheduler()
    
    def single_core_schedule(time_window):
        """单核系统的调度实现"""
        
        print("\n🚀 开始单核系统调度...")
        
        # 初始化
        scheduler.schedule_history = []
        single_scheduler = scheduler._single_core_scheduler
        single_scheduler.resource_timelines.clear()
        single_scheduler.task_execution_times.clear()
        
        current_time = 0.0
        scheduled_count = 0
        
        # 主调度循环
        while current_time < time_window and scheduled_count < 100:
            # 收集可调度的任务
            schedulable_tasks = []
            
            for task in scheduler.tasks.values():
                can_schedule, reason = single_scheduler.can_schedule_task(task, current_time)
                if can_schedule:
                    schedulable_tasks.append(task)
            
            # 按优先级排序
            schedulable_tasks.sort(key=lambda t: (t.priority.value, t.task_id))
            
            # 尝试调度
            task_scheduled = False
            
            for task in schedulable_tasks:
                # 为任务制定调度计划
                if len(task.segments) > 4:  # 复杂任务
                    plan = single_scheduler.schedule_complex_task(task, scheduler, current_time)
                else:  # 简单任务
                    plan = schedule_simple_task(task, scheduler, single_scheduler, current_time)
                
                if plan:
                    # 执行调度计划
                    
                    # 预留所有资源
                    for seg_plan in plan['segments']:
                        single_scheduler.reserve_time_slot(
                            seg_plan['resource'].unit_id,
                            seg_plan['start'],
                            seg_plan['end'],
                            task.task_id,
                            f"seg{seg_plan['segment_idx']}"
                        )
                    
                    # 创建调度记录
                    assigned_resources = {}
                    for seg_plan in plan['segments']:
                        seg = task.segments[seg_plan['segment_idx']]
                        assigned_resources[seg.resource_type] = seg_plan['resource'].unit_id
                    
                    schedule_info = TaskScheduleInfo(
                        task_id=task.task_id,
                        start_time=plan['task_start'],
                        end_time=plan['task_end'],
                        assigned_resources=assigned_resources,
                        actual_latency=plan['task_end'] - plan['task_start'],
                        runtime_type=task.runtime_type,
                        used_cuts=[],
                        segmentation_overhead=0.0,
                        sub_segment_schedule=[]
                    )
                    
                    # 更新状态
                    if task.task_id not in single_scheduler.task_execution_times:
                        single_scheduler.task_execution_times[task.task_id] = []
                    single_scheduler.task_execution_times[task.task_id].append(plan['task_start'])
                    
                    scheduler.schedule_history.append(schedule_info)
                    scheduled_count += 1
                    task_scheduled = True
                    
                    print(f"  {plan['task_start']:6.1f}ms: [{task.priority.name:8}] {task.task_id} "
                          f"开始 (结束于 {plan['task_end']:.1f}ms)")
                    
                    # 如果是T1，显示详细信息
                    if task.task_id == "T1" and len(single_scheduler.task_execution_times[task.task_id]) == 1:
                        print(f"    T1 segments详情:")
                        for seg_plan in plan['segments']:
                            res_type = task.segments[seg_plan['segment_idx']].resource_type.name
                            print(f"      seg{seg_plan['segment_idx']}({res_type}): "
                                  f"{seg_plan['start']:.1f}-{seg_plan['end']:.1f}ms "
                                  f"on {seg_plan['resource'].unit_id}")
                    
                    break
            
            # 时间推进
            if task_scheduled:
                current_time += 0.1
            else:
                # 找下一个有意义的时间点
                next_time = current_time + 2.0
                
                # 检查任务就绪时间
                for task_id, exec_times in single_scheduler.task_execution_times.items():
                    if exec_times:
                        task = scheduler.tasks[task_id]
                        next_ready = exec_times[-1] + 1000.0 / task.fps_requirement
                        if next_ready > current_time:
                            next_time = min(next_time, next_ready)
                
                # 检查资源空闲时间
                for timeline in single_scheduler.resource_timelines.values():
                    for _, end, _, _ in timeline:
                        if end > current_time:
                            next_time = min(next_time, end)
                
                current_time = min(next_time, time_window)
        
        print(f"\n✅ 调度完成: {len(scheduler.schedule_history)} 个事件")
        
        # 显示统计
        show_single_core_stats(scheduler, single_scheduler, time_window)
        
        return scheduler.schedule_history
    
    # 替换调度方法
    scheduler.priority_aware_schedule_with_segmentation = single_core_schedule
    
    print("  ✓ 单核调度逻辑已应用")


def schedule_simple_task(task, scheduler, single_scheduler, current_time):
    """调度简单任务"""
    
    segment_plans = []
    max_end = current_time
    
    for i, segment in enumerate(task.segments):
        # 找资源
        best_resource = None
        best_duration = float('inf')
        
        for resource in scheduler.resources.get(segment.resource_type, []):
            duration = segment.get_duration(resource.bandwidth)
            if duration < best_duration:
                best_duration = duration
                best_resource = resource
        
        if not best_resource:
            return None
        
        # 计算时间
        seg_start = current_time + segment.start_time
        actual_start = single_scheduler.find_feasible_time(
            best_resource.unit_id,
            best_duration,
            seg_start
        )
        
        if actual_start is None:
            return None
        
        seg_end = actual_start + best_duration
        max_end = max(max_end, seg_end)
        
        segment_plans.append({
            'segment_idx': i,
            'resource': best_resource,
            'start': actual_start,
            'end': seg_end,
            'duration': best_duration
        })
    
    return {
        'task_start': current_time,
        'task_end': max_end,
        'segments': segment_plans
    }


def show_single_core_stats(scheduler, single_scheduler, time_window):
    """显示单核系统的统计信息"""
    
    print("\n任务执行统计:")
    
    # 计算每个任务的目标和实际执行次数
    for task_id, task in sorted(scheduler.tasks.items()):
        exec_times = single_scheduler.task_execution_times.get(task_id, [])
        actual = len(exec_times)
        expected = max(1, int((time_window * task.fps_requirement) / 1000.0))
        percentage = (actual / expected * 100) if expected > 0 else 0
        
        print(f"  {task_id}: {actual}/{expected} 次 ({percentage:.0f}%) - "
              f"优先级: {task.priority.name}")
    
    # 资源利用率
    print("\n资源利用率:")
    resource_busy = defaultdict(float)
    
    for res_id, timeline in single_scheduler.resource_timelines.items():
        for start, end, _, _ in timeline:
            resource_busy[res_id] += (end - start)
    
    if scheduler.schedule_history:
        actual_window = max(s.end_time for s in scheduler.schedule_history)
    else:
        actual_window = time_window
        
    for res_id, busy_time in sorted(resource_busy.items()):
        utilization = (busy_time / actual_window * 100) if actual_window > 0 else 0
        print(f"  {res_id}: {utilization:.1f}%")
    
    # 调度密度分析
    print("\n调度密度分析:")
    if single_scheduler.resource_timelines:
        for res_id, timeline in single_scheduler.resource_timelines.items():
            if timeline:
                sorted_timeline = sorted(timeline, key=lambda x: x[0])
                gaps = []
                
                for i in range(len(sorted_timeline) - 1):
                    gap = sorted_timeline[i+1][0] - sorted_timeline[i][1]
                    if gap > 5.0:
                        gaps.append(gap)
                
                if gaps:
                    avg_gap = sum(gaps) / len(gaps)
                    print(f"  {res_id}: 平均空隙 {avg_gap:.1f}ms, 最大空隙 {max(gaps):.1f}ms")


def add_enhanced_validation(scheduler):
    """添加增强的验证功能"""
    
    def validate_single_core_schedule():
        """验证单核调度结果"""
        
        print("\n=== 单核调度验证 ===")
        
        if not hasattr(scheduler, '_single_core_scheduler'):
            print("❌ 单核调度器未初始化")
            return False
        
        conflicts = []
        single_scheduler = scheduler._single_core_scheduler
        
        # 检查每个资源的冲突
        for res_id, timeline in single_scheduler.resource_timelines.items():
            sorted_timeline = sorted(timeline, key=lambda x: x[0])
            
            for i in range(len(sorted_timeline) - 1):
                s1, e1, t1, i1 = sorted_timeline[i]
                s2, e2, t2, i2 = sorted_timeline[i + 1]
                
                if e1 > s2 + 0.001:
                    conflicts.append({
                        'resource': res_id,
                        'conflict': f"{t1}/{i1} ({s1:.1f}-{e1:.1f}) vs {t2}/{i2} ({s2:.1f}-{e2:.1f})",
                        'overlap': e1 - s2
                    })
        
        if conflicts:
            print(f"❌ 发现 {len(conflicts)} 个资源冲突:")
            for c in conflicts[:5]:
                print(f"  {c['resource']}: {c['conflict']} - 重叠 {c['overlap']:.1f}ms")
            return False
        else:
            print("✅ 没有资源冲突")
            
            # 检查关键任务是否被调度
            if 'T1' in single_scheduler.task_execution_times:
                t1_count = len(single_scheduler.task_execution_times['T1'])
                print(f"  T1 执行了 {t1_count} 次")
            else:
                print("  ⚠️ T1 未被调度")
            
            return True
    
    scheduler.validate_schedule = validate_single_core_schedule
    print("  ✓ 增强验证已添加")


if __name__ == "__main__":
    print("Dragon4单核系统修复")
    print("=" * 60)
    print("专门处理资源受限环境下的复杂任务调度")
