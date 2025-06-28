#!/usr/bin/env python3
"""
综合调度分析器
分析并显示任务的帧率要求和实际达成情况
"""

from collections import defaultdict
from typing import Dict, List, Tuple


def analyze_fps_requirements(scheduler, time_window: float = 200.0):
    """分析所有任务的FPS要求和实际达成情况"""
    
    print("\n" + "=" * 80)
    print("📊 任务FPS分析报告")
    print("=" * 80)
    
    # 收集每个任务的执行信息
    task_executions = defaultdict(list)
    for schedule in scheduler.schedule_history:
        task_executions[schedule.task_id].append((schedule.start_time, schedule.end_time))
    
    # 分析每个任务
    all_satisfied = True
    fps_data = []
    
    for task_id, task in sorted(scheduler.tasks.items()):
        executions = task_executions[task_id]
        
        # 计算实际FPS
        if executions:
            # 计算平均间隔
            intervals = []
            for i in range(1, len(executions)):
                interval = executions[i][0] - executions[i-1][0]
                intervals.append(interval)
            
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                actual_fps = 1000.0 / avg_interval if avg_interval > 0 else 0
            else:
                # 只有一次执行
                actual_fps = len(executions) / (time_window / 1000.0)
        else:
            actual_fps = 0
        
        # 期望的执行次数
        expected_executions = int((time_window / 1000.0) * task.fps_requirement)
        actual_executions = len(executions)
        
        # 检查是否满足要求
        fps_ratio = actual_fps / task.fps_requirement if task.fps_requirement > 0 else 1.0
        is_satisfied = fps_ratio >= 0.95  # 允许5%的误差
        
        if not is_satisfied:
            all_satisfied = False
        
        # 收集数据
        fps_data.append({
            'task_id': task_id,
            'task_name': task.name,
            'priority': task.priority.name,
            'required_fps': task.fps_requirement,
            'actual_fps': actual_fps,
            'fps_ratio': fps_ratio,
            'expected_execs': expected_executions,
            'actual_execs': actual_executions,
            'is_satisfied': is_satisfied,
            'latency_req': task.latency_requirement,
            'runtime_type': task.runtime_type.name,
            'is_segmented': getattr(task, 'is_segmented', False)
        })
    
    # 打印报告
    print(f"\n{'任务ID':<8} {'名称':<15} {'优先级':<10} {'要求FPS':<10} {'实际FPS':<10} "
          f"{'达成率':<8} {'执行次数':<12} {'状态':<8}")
    print("-" * 100)
    
    for data in fps_data:
        status = "✅ 满足" if data['is_satisfied'] else "❌ 未满足"
        exec_info = f"{data['actual_execs']}/{data['expected_execs']}"
        
        print(f"{data['task_id']:<8} {data['task_name']:<15} {data['priority']:<10} "
              f"{data['required_fps']:<10.1f} {data['actual_fps']:<10.1f} "
              f"{data['fps_ratio']*100:<8.1f}% {exec_info:<12} {status:<8}")
    
    # 按优先级分组统计
    print("\n📈 按优先级分组统计:")
    priority_groups = defaultdict(list)
    for data in fps_data:
        priority_groups[data['priority']].append(data)
    
    for priority in ['CRITICAL', 'HIGH', 'NORMAL', 'LOW']:
        if priority in priority_groups:
            group = priority_groups[priority]
            satisfied = sum(1 for d in group if d['is_satisfied'])
            total = len(group)
            print(f"\n{priority} 优先级:")
            print(f"  - 任务数: {total}")
            print(f"  - 满足FPS要求: {satisfied}/{total} ({satisfied/total*100:.1f}%)")
            
            for data in group:
                if not data['is_satisfied']:
                    print(f"  - ⚠️ {data['task_id']} ({data['task_name']}): "
                          f"需要 {data['required_fps']} FPS, 实际 {data['actual_fps']:.1f} FPS")
    
    # 分段任务统计
    print("\n🔗 分段任务分析:")
    segmented_tasks = [d for d in fps_data if d['is_segmented']]
    if segmented_tasks:
        print(f"  - 分段任务数: {len(segmented_tasks)}")
        for data in segmented_tasks:
            print(f"  - {data['task_id']} ({data['task_name']}): "
                  f"FPS达成率 {data['fps_ratio']*100:.1f}%")
    else:
        # 检查是否有应该被分段但没有被标记的任务
        potential_segmented = []
        for task_id, task in scheduler.tasks.items():
            if hasattr(task, 'segmentation_strategy'):
                strategy = task.segmentation_strategy.name if hasattr(task.segmentation_strategy, 'name') else str(task.segmentation_strategy)
                if strategy not in ["NO_SEGMENTATION", "NONE"]:
                    potential_segmented.append((task_id, task.name, strategy))
        
        if potential_segmented:
            print(f"  - 检测到 {len(potential_segmented)} 个可能的分段任务:")
            for task_id, name, strategy in potential_segmented:
                print(f"    - {task_id} ({name}): {strategy}")
    
    return all_satisfied, fps_data


def suggest_schedule_improvements(fps_data: List[Dict], scheduler):
    """建议调度改进方案"""
    
    print("\n" + "=" * 80)
    print("💡 调度改进建议")
    print("=" * 80)
    
    unsatisfied_tasks = [d for d in fps_data if not d['is_satisfied']]
    
    if not unsatisfied_tasks:
        print("\n✅ 所有任务都满足FPS要求，调度性能良好！")
        return
    
    print(f"\n发现 {len(unsatisfied_tasks)} 个任务未满足FPS要求:")
    
    # 按优先级排序
    unsatisfied_tasks.sort(key=lambda x: (
        0 if x['priority'] == 'CRITICAL' else
        1 if x['priority'] == 'HIGH' else
        2 if x['priority'] == 'NORMAL' else 3
    ))
    
    for data in unsatisfied_tasks:
        print(f"\n任务 {data['task_id']} ({data['task_name']}):")
        print(f"  - 优先级: {data['priority']}")
        print(f"  - 需要: {data['required_fps']} FPS")
        print(f"  - 实际: {data['actual_fps']:.1f} FPS")
        print(f"  - 缺口: {data['required_fps'] - data['actual_fps']:.1f} FPS")
        
        # 分析原因
        task = scheduler.tasks[data['task_id']]
        print(f"  - 可能原因:")
        
        # 检查资源竞争
        if data['priority'] == 'LOW':
            print(f"    • 低优先级任务，可能被高优先级任务抢占")
        
        # 检查执行时间
        if hasattr(task, 'latency_requirement'):
            min_interval = 1000.0 / data['required_fps']
            if task.latency_requirement > min_interval * 0.8:
                print(f"    • 任务执行时间 ({task.latency_requirement}ms) 接近或超过最小间隔 ({min_interval:.1f}ms)")
        
        # 建议
        print(f"  - 建议:")
        if data['priority'] in ['NORMAL', 'LOW'] and data['fps_ratio'] < 0.5:
            print(f"    • 考虑提升任务优先级")
        if not data['is_segmented'] and hasattr(task, 'segmentation_strategy'):
            print(f"    • 考虑启用任务分段以提高调度灵活性")
        print(f"    • 检查是否有足够的资源容量")


def print_resource_timeline_gaps(scheduler, time_window: float = 200.0):
    """打印资源时间线中的空闲间隙"""
    
    print("\n" + "=" * 80)
    print("🕐 资源空闲时间分析")
    print("=" * 80)
    
    # 构建每个资源的时间线
    resource_timelines = defaultdict(list)
    
    for schedule in scheduler.schedule_history:
        if hasattr(schedule, 'sub_segment_schedule') and schedule.sub_segment_schedule:
            for sub_seg_id, start, end in schedule.sub_segment_schedule:
                # 找到对应的资源
                task = scheduler.tasks[schedule.task_id]
                for ss in task.get_sub_segments_for_scheduling():
                    if ss.sub_id == sub_seg_id:
                        if ss.resource_type in schedule.assigned_resources:
                            resource_id = schedule.assigned_resources[ss.resource_type]
                            resource_timelines[resource_id].append((start, end, schedule.task_id))
                        break
    
    # 分析每个资源的空闲时间
    for resource_id in sorted(resource_timelines.keys()):
        timeline = sorted(resource_timelines[resource_id])
        
        if not timeline:
            continue
        
        # 计算空闲间隙
        gaps = []
        for i in range(1, len(timeline)):
            prev_end = timeline[i-1][1]
            curr_start = timeline[i][0]
            gap = curr_start - prev_end
            if gap > 0.1:  # 只显示大于0.1ms的间隙
                gaps.append((prev_end, curr_start, gap))
        
        # 计算总空闲时间
        total_gap_time = sum(g[2] for g in gaps)
        
        print(f"\n{resource_id}:")
        print(f"  - 总空闲时间: {total_gap_time:.1f}ms ({total_gap_time/time_window*100:.1f}%)")
        
        if gaps and len(gaps) <= 5:
            print(f"  - 主要空闲间隙:")
            for start, end, gap in sorted(gaps, key=lambda x: x[2], reverse=True)[:5]:
                print(f"    • {start:.1f} - {end:.1f}ms (间隙: {gap:.1f}ms)")


def comprehensive_schedule_analysis(scheduler, time_window: float = 200.0):
    """综合调度分析"""
    
    # 1. FPS分析
    all_fps_satisfied, fps_data = analyze_fps_requirements(scheduler, time_window)
    
    # 2. 改进建议
    suggest_schedule_improvements(fps_data, scheduler)
    
    # 3. 资源空闲分析
    print_resource_timeline_gaps(scheduler, time_window)
    
    return all_fps_satisfied


if __name__ == "__main__":
    print("综合调度分析器")
    print("功能：")
    print("1. 分析任务FPS要求和实际达成情况")
    print("2. 提供调度改进建议")
    print("3. 分析资源空闲时间")
