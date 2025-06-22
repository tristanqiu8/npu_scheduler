#!/usr/bin/env python3
"""
Schedule Validator - 验证调度结果的正确性
"""

from collections import defaultdict
from typing import List, Dict, Tuple

def validate_schedule(scheduler) -> Tuple[bool, List[str]]:
    """
    验证调度结果，确保没有资源冲突
    返回: (是否有效, 错误信息列表)
    """
    errors = []
    
    # 为每个资源构建时间线
    resource_timeline = defaultdict(list)  # {resource_id: [(start, end, task_id, segment_info)]}
    
    for schedule in scheduler.schedule_history:
        task = scheduler.tasks[schedule.task_id]
        
        if task.is_segmented and schedule.sub_segment_schedule:
            # 处理分段任务
            for i, (sub_seg_id, start_time, end_time) in enumerate(schedule.sub_segment_schedule):
                # 找到对应的子段
                for sub_seg in task.get_sub_segments_for_scheduling():
                    if sub_seg.sub_id == sub_seg_id:
                        if sub_seg.resource_type in schedule.assigned_resources:
                            resource_id = schedule.assigned_resources[sub_seg.resource_type]
                            segment_info = f"{task.task_id}-{i+1}"
                            resource_timeline[resource_id].append(
                                (start_time, end_time, task.task_id, segment_info)
                            )
                        break
        else:
            # 处理非分段任务
            for seg in task.segments:
                if seg.resource_type in schedule.assigned_resources:
                    resource_id = schedule.assigned_resources[seg.resource_type]
                    resource_unit = next((r for r in scheduler.resources[seg.resource_type] 
                                        if r.unit_id == resource_id), None)
                    if resource_unit:
                        duration = seg.get_duration(resource_unit.bandwidth)
                        start_time = schedule.start_time + seg.start_time
                        end_time = start_time + duration
                        resource_timeline[resource_id].append(
                            (start_time, end_time, task.task_id, task.task_id)
                        )
    
    # 检查每个资源上的冲突
    for resource_id, timeline in resource_timeline.items():
        # 按开始时间排序
        timeline.sort(key=lambda x: x[0])
        
        # 检查重叠
        for i in range(len(timeline) - 1):
            curr_start, curr_end, curr_task, curr_seg = timeline[i]
            next_start, next_end, next_task, next_seg = timeline[i + 1]
            
            # 如果当前任务的结束时间大于下一个任务的开始时间，说明有重叠
            if curr_end > next_start + 0.001:  # 允许微小的浮点误差
                overlap = curr_end - next_start
                errors.append(
                    f"资源冲突: {resource_id} 上 {curr_seg} ({curr_start:.1f}-{curr_end:.1f}ms) "
                    f"与 {next_seg} ({next_start:.1f}-{next_end:.1f}ms) 重叠 {overlap:.1f}ms"
                )
    
    # 打印资源使用情况
    print("\n=== 资源使用时间线 ===")
    for resource_id in sorted(resource_timeline.keys()):
        timeline = resource_timeline[resource_id]
        print(f"\n{resource_id}:")
        for start, end, task_id, seg_info in timeline:
            print(f"  {start:6.1f} - {end:6.1f} ms: {seg_info}")
    
    # 检查任务执行频率
    print("\n=== 任务执行统计 ===")
    task_executions = defaultdict(list)
    for schedule in scheduler.schedule_history:
        task_executions[schedule.task_id].append((schedule.start_time, schedule.end_time))
    
    for task_id, task in scheduler.tasks.items():
        executions = task_executions[task_id]
        if executions:
            print(f"{task_id}: {len(executions)} 次执行")
            for i, (start, end) in enumerate(executions):
                print(f"  执行{i+1}: {start:.1f} - {end:.1f} ms")
                if i > 0:
                    interval = start - executions[i-1][0]
                    expected_interval = 1000.0 / task.fps_requirement
                    if interval < expected_interval * 0.9:  # 10%容差
                        errors.append(
                            f"任务{task_id}执行间隔过短: {interval:.1f}ms < {expected_interval:.1f}ms"
                        )
    
    return len(errors) == 0, errors

def analyze_chrome_trace(trace_file: str):
    """分析Chrome Trace文件中的潜在问题"""
    import json
    
    with open(trace_file, 'r') as f:
        data = json.load(f)
    
    events = data.get("traceEvents", [])
    
    # 按进程和线程组织事件
    timeline = defaultdict(lambda: defaultdict(list))
    
    for event in events:
        if event.get("ph") == "X":  # 完整事件
            pid = event.get("pid")
            tid = event.get("tid")
            if pid and tid:
                timeline[pid][tid].append({
                    "name": event.get("name"),
                    "start": event.get("ts", 0) / 1000.0,  # 转回毫秒
                    "duration": event.get("dur", 0) / 1000.0,
                    "end": event.get("ts", 0) / 1000.0 + event.get("dur", 0) / 1000.0
                })
    
    print(f"\n=== Chrome Trace 分析 ===")
    
    # 检查每个线程的事件
    for pid, threads in timeline.items():
        for tid, events in threads.items():
            # 按开始时间排序
            events.sort(key=lambda x: x["start"])
            
            print(f"\nPID {pid}, TID {tid}:")
            overlaps = []
            
            for i in range(len(events) - 1):
                curr = events[i]
                next_event = events[i + 1]
                
                print(f"  {curr['name']}: {curr['start']:.1f} - {curr['end']:.1f} ms")
                
                # 检查重叠
                if curr["end"] > next_event["start"] + 0.001:
                    overlap = curr["end"] - next_event["start"]
                    overlaps.append(
                        f"    ⚠️ 重叠: {curr['name']} 与 {next_event['name']} 重叠 {overlap:.1f}ms"
                    )
            
            if events:
                print(f"  {events[-1]['name']}: {events[-1]['start']:.1f} - {events[-1]['end']:.1f} ms")
            
            for overlap in overlaps:
                print(overlap)

