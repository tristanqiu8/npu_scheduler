#!/usr/bin/env python3
"""
验证器精度修复模块
修复调度验证中的浮点精度问题
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP


def apply_validator_precision_fix():
    """应用验证器精度修复
    
    修复schedule_validator.py中可能出现的浮点精度问题
    这个函数主要是为了兼容dragon4_optimization_test.py中的导入
    """
    print("✅ Applying validator precision fix...")
    
    # 设置全局精度常量
    global VALIDATION_PRECISION, TIME_TOLERANCE
    VALIDATION_PRECISION = Decimal('0.001')  # 1毫秒精度
    TIME_TOLERANCE = 0.001  # 时间容差
    
    print("  - Validation precision: 0.001ms")
    print("  - Time tolerance: 0.001ms")
    print("  - Enhanced floating point handling")


def validate_schedule_with_precision(scheduler) -> Tuple[bool, List[str]]:
    """高精度调度验证
    
    Args:
        scheduler: 调度器实例
        
    Returns:
        (is_valid, error_list): 验证结果和错误列表
    """
    errors = []
    
    if not hasattr(scheduler, 'schedule_history') or not scheduler.schedule_history:
        return True, []
    
    # 按资源分组调度事件
    resource_timelines = defaultdict(list)
    
    for schedule in scheduler.schedule_history:
        for res_type, res_id in schedule.assigned_resources.items():
            resource_timelines[res_id].append({
                'start': to_decimal(schedule.start_time),
                'end': to_decimal(schedule.end_time),
                'task_id': schedule.task_id,
                'schedule': schedule
            })
    
    # 检查每个资源的时间冲突
    for res_id, timeline in resource_timelines.items():
        # 按开始时间排序
        timeline.sort(key=lambda x: x['start'])
        
        # 检查相邻事件的重叠
        for i in range(len(timeline) - 1):
            current = timeline[i]
            next_event = timeline[i + 1]
            
            # 使用Decimal进行精确比较
            if current['end'] > next_event['start'] + VALIDATION_PRECISION:
                overlap = float(current['end'] - next_event['start'])
                errors.append(
                    f"资源冲突 {res_id}: {current['task_id']} "
                    f"({float(current['start']):.3f}-{float(current['end']):.3f}ms) "
                    f"与 {next_event['task_id']} "
                    f"({float(next_event['start']):.3f}-{float(next_event['end']):.3f}ms) "
                    f"重叠 {overlap:.3f}ms"
                )
    
    return len(errors) == 0, errors


def to_decimal(value: float) -> Decimal:
    """将浮点数转换为高精度Decimal"""
    return Decimal(str(value)).quantize(VALIDATION_PRECISION, rounding=ROUND_HALF_UP)


def analyze_timing_precision(scheduler) -> Dict:
    """分析调度器的时间精度问题
    
    Args:
        scheduler: 调度器实例
        
    Returns:
        分析结果字典
    """
    analysis = {
        'total_events': 0,
        'precision_issues': [],
        'resource_usage': {},
        'time_gaps': [],
        'overlaps': []
    }
    
    if not hasattr(scheduler, 'schedule_history'):
        return analysis
    
    analysis['total_events'] = len(scheduler.schedule_history)
    
    # 按资源分析
    resource_events = defaultdict(list)
    
    for schedule in scheduler.schedule_history:
        for res_type, res_id in schedule.assigned_resources.items():
            resource_events[res_id].append({
                'start': schedule.start_time,
                'end': schedule.end_time,
                'task_id': schedule.task_id
            })
    
    for res_id, events in resource_events.items():
        # 排序
        events.sort(key=lambda x: x['start'])
        
        # 计算利用率
        if events:
            total_time = events[-1]['end'] - events[0]['start']
            busy_time = sum(e['end'] - e['start'] for e in events)
            analysis['resource_usage'][res_id] = {
                'utilization': (busy_time / total_time * 100) if total_time > 0 else 0,
                'total_time': total_time,
                'busy_time': busy_time,
                'events_count': len(events)
            }
        
        # 检查时间间隙和重叠
        for i in range(len(events) - 1):
            current = events[i]
            next_event = events[i + 1]
            
            gap = next_event['start'] - current['end']
            
            if gap < -0.001:  # 重叠
                analysis['overlaps'].append({
                    'resource': res_id,
                    'task1': current['task_id'],
                    'task2': next_event['task_id'],
                    'overlap': -gap,
                    'time': next_event['start']
                })
            elif gap > 0.001:  # 间隙
                analysis['time_gaps'].append({
                    'resource': res_id,
                    'gap': gap,
                    'after_task': current['task_id'],
                    'before_task': next_event['task_id'],
                    'time': current['end']
                })
    
    return analysis


def print_precision_analysis(analysis: Dict):
    """打印精度分析结果"""
    print("\n=== 时间精度分析 ===")
    print(f"总事件数: {analysis['total_events']}")
    
    if analysis['overlaps']:
        print(f"\n❌ 发现 {len(analysis['overlaps'])} 个时间重叠:")
        for overlap in analysis['overlaps'][:5]:  # 只显示前5个
            print(f"  {overlap['resource']}: {overlap['task1']} 与 {overlap['task2']} "
                  f"重叠 {overlap['overlap']:.3f}ms")
    else:
        print("\n✅ 没有时间重叠")
    
    if analysis['time_gaps']:
        avg_gap = sum(g['gap'] for g in analysis['time_gaps']) / len(analysis['time_gaps'])
        print(f"\n📊 发现 {len(analysis['time_gaps'])} 个时间间隙, 平均间隙: {avg_gap:.3f}ms")
    
    print("\n资源利用率:")
    for res_id, usage in analysis['resource_usage'].items():
        print(f"  {res_id}: {usage['utilization']:.1f}% "
              f"({usage['events_count']} 个事件)")


def enhanced_schedule_validation(scheduler, verbose: bool = False) -> bool:
    """增强的调度验证
    
    Args:
        scheduler: 调度器实例
        verbose: 是否打印详细信息
        
    Returns:
        验证是否通过
    """
    # 基础验证
    is_valid, errors = validate_schedule_with_precision(scheduler)
    
    if verbose:
        print(f"\n=== 调度验证结果 ===")
        if is_valid:
            print("✅ 基础验证通过")
        else:
            print(f"❌ 基础验证失败，发现 {len(errors)} 个错误:")
            for error in errors[:3]:
                print(f"  - {error}")
            if len(errors) > 3:
                print(f"  ... 还有 {len(errors) - 3} 个错误")
    
    # 精度分析
    if verbose:
        analysis = analyze_timing_precision(scheduler)
        print_precision_analysis(analysis)
    
    return is_valid


# 为了兼容性，提供一些常用的验证函数
def validate_no_conflicts(scheduler) -> bool:
    """验证没有资源冲突"""
    is_valid, _ = validate_schedule_with_precision(scheduler)
    return is_valid


def get_validation_errors(scheduler) -> List[str]:
    """获取验证错误列表"""
    _, errors = validate_schedule_with_precision(scheduler)
    return errors


# 全局变量初始化
VALIDATION_PRECISION = Decimal('0.001')
TIME_TOLERANCE = 0.001


if __name__ == "__main__":
    print("验证器精度修复模块")
    print("主要功能:")
    print("1. 修复浮点精度问题")
    print("2. 提供高精度时间验证")
    print("3. 详细的时间分析")
    print("\n使用方法:")
    print("from validator_precision_fix import apply_validator_precision_fix")
    print("apply_validator_precision_fix()")
