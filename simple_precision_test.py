#!/usr/bin/env python3
"""
简单精度测试 - 直接测试时间冲突解决方案
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict, Optional
from decimal import Decimal, ROUND_UP
from dataclasses import dataclass
import copy

@dataclass
class SimpleEvent:
    """简单的调度事件"""
    task_name: str
    segment_id: str
    resource: str
    start_time: float
    end_time: float
    
    def __repr__(self):
        return f"{self.task_name}-{self.segment_id} on {self.resource}: {self.start_time:.1f}-{self.end_time:.1f}ms"


class PrecisionScheduler:
    """高精度调度器 - 解决时间冲突"""
    
    def __init__(self):
        self.events: List[SimpleEvent] = []
        self.PRECISION = Decimal('0.1')  # 0.1ms精度
        self.BUFFER = Decimal('0.1')     # 0.1ms缓冲
        
    def _to_decimal(self, value: float) -> Decimal:
        """转换为Decimal并量化到指定精度"""
        return Decimal(str(value)).quantize(self.PRECISION, rounding=ROUND_UP)
    
    def _check_conflict(self, resource: str, start: Decimal, end: Decimal) -> Optional[SimpleEvent]:
        """检查是否有资源冲突"""
        for event in self.events:
            if event.resource == resource:
                event_start = self._to_decimal(event.start_time)
                event_end = self._to_decimal(event.end_time)
                
                # 添加缓冲区检查
                if not (end + self.BUFFER <= event_start or start >= event_end + self.BUFFER):
                    return event
        return None
    
    def _find_next_available(self, resource: str, duration: Decimal, earliest: Decimal) -> Decimal:
        """找到资源的下一个可用时间"""
        # 获取该资源上的所有事件，按开始时间排序
        resource_events = sorted(
            [e for e in self.events if e.resource == resource],
            key=lambda e: e.start_time
        )
        
        if not resource_events:
            return earliest
        
        current = earliest
        
        for event in resource_events:
            event_start = self._to_decimal(event.start_time)
            event_end = self._to_decimal(event.end_time)
            
            # 如果当前时间段可以放下任务
            if current + duration + self.BUFFER <= event_start:
                return current
            
            # 否则移到这个事件结束后
            current = event_end + self.BUFFER
        
        return current
    
    def schedule_event(self, task_name: str, segment_id: str, resource: str, 
                      duration: float, earliest_start: float) -> Optional[SimpleEvent]:
        """调度一个事件"""
        duration_dec = self._to_decimal(duration)
        earliest_dec = self._to_decimal(earliest_start)
        
        # 找到可用时间
        start_time = self._find_next_available(resource, duration_dec, earliest_dec)
        end_time = start_time + duration_dec
        
        # 再次检查冲突（双重保险）
        conflict = self._check_conflict(resource, start_time, end_time)
        if conflict:
            print(f"⚠️ 仍有冲突: {task_name}-{segment_id} 与 {conflict}")
            return None
        
        # 创建事件
        event = SimpleEvent(
            task_name=task_name,
            segment_id=segment_id,
            resource=resource,
            start_time=float(start_time),
            end_time=float(end_time)
        )
        
        self.events.append(event)
        return event
    
    def validate_schedule(self) -> List[str]:
        """验证调度是否有冲突"""
        issues = []
        
        # 按资源分组
        by_resource = {}
        for event in self.events:
            if event.resource not in by_resource:
                by_resource[event.resource] = []
            by_resource[event.resource].append(event)
        
        # 检查每个资源
        for resource, events in by_resource.items():
            sorted_events = sorted(events, key=lambda e: e.start_time)
            
            for i in range(len(sorted_events) - 1):
                current = sorted_events[i]
                next_event = sorted_events[i + 1]
                
                current_end = self._to_decimal(current.end_time)
                next_start = self._to_decimal(next_event.start_time)
                
                if current_end > next_start:
                    overlap = float(current_end - next_start)
                    issues.append(
                        f"{resource}: {current.task_name}-{current.segment_id} "
                        f"({current.start_time:.1f}-{current.end_time:.1f}) 与 "
                        f"{next_event.task_name}-{next_event.segment_id} "
                        f"({next_event.start_time:.1f}-{next_event.end_time:.1f}) "
                        f"重叠 {overlap:.1f}ms"
                    )
                elif current_end + self.BUFFER > next_start:
                    gap = float(next_start - current_end)
                    issues.append(
                        f"{resource}: 间隔过小 {gap:.1f}ms between "
                        f"{current.task_name}-{current.segment_id} 和 "
                        f"{next_event.task_name}-{next_event.segment_id}"
                    )
        
        return issues
    
    def print_timeline(self):
        """打印时间线"""
        # 按资源分组
        by_resource = {}
        for event in self.events:
            if event.resource not in by_resource:
                by_resource[event.resource] = []
            by_resource[event.resource].append(event)
        
        print("\n=== 资源时间线 ===")
        for resource in sorted(by_resource.keys()):
            print(f"\n{resource}:")
            events = sorted(by_resource[resource], key=lambda e: e.start_time)
            for event in events:
                print(f"  {event.start_time:6.1f} - {event.end_time:6.1f} ms: "
                      f"{event.task_name}-{event.segment_id}")


def test_precision_scheduling():
    """测试精度调度"""
    print("=== 精度调度测试 ===\n")
    
    scheduler = PrecisionScheduler()
    
    # 模拟之前有问题的场景
    print("调度任务段...")
    
    # T1的多次执行（每100ms）
    for i in range(4):
        base_time = i * 100.0
        # T1有3个段，每个约6-7ms
        scheduler.schedule_event("T1", "1", "NPU_0", 6.7, base_time)
        scheduler.schedule_event("T1", "2", "NPU_0", 7.0, base_time + 6.7)
        scheduler.schedule_event("T1", "3", "NPU_0", 6.6, base_time + 13.7)
        # 有时使用NPU_1
        if i % 2 == 1:
            scheduler.schedule_event("T1", "1", "NPU_1", 6.7, base_time)
            scheduler.schedule_event("T1", "2", "NPU_1", 7.0, base_time + 6.7)
            scheduler.schedule_event("T1", "3", "NPU_1", 6.6, base_time + 13.7)
    
    # T2的多次执行（每40ms）
    for i in range(5):
        base_time = i * 40.0
        # DSP段
        scheduler.schedule_event("T2", "1", "DSP_0", 10.0, base_time)
        # NPU段（2个7.5ms的段）
        npu_start = base_time + 10.0
        
        # 尝试在NPU_0和NPU_1之间分配
        if i % 2 == 0:
            scheduler.schedule_event("T2", "2", "NPU_0", 7.5, npu_start)
            scheduler.schedule_event("T2", "3", "NPU_0", 7.5, npu_start + 7.5)
        else:
            scheduler.schedule_event("T2", "2", "NPU_1", 7.5, npu_start)
            scheduler.schedule_event("T2", "3", "NPU_1", 7.5, npu_start + 7.5)
    
    # 打印时间线
    scheduler.print_timeline()
    
    # 验证
    print("\n=== 验证结果 ===")
    issues = scheduler.validate_schedule()
    
    if issues:
        print(f"❌ 发现 {len(issues)} 个问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("✅ 所有调度都正确，没有时间冲突！")
    
    # 统计
    print(f"\n=== 统计 ===")
    print(f"总共调度了 {len(scheduler.events)} 个事件")
    
    # 按任务统计
    task_stats = {}
    for event in scheduler.events:
        if event.task_name not in task_stats:
            task_stats[event.task_name] = 0
        task_stats[event.task_name] += 1
    
    for task, count in sorted(task_stats.items()):
        print(f"{task}: {count} 个段")
    
    # 资源利用率
    print("\n=== 资源利用率 (前200ms) ===")
    for resource in ["NPU_0", "NPU_1", "DSP_0"]:
        resource_events = [e for e in scheduler.events if e.resource == resource and e.end_time <= 200]
        if resource_events:
            busy_time = sum(e.end_time - e.start_time for e in resource_events)
            utilization = (busy_time / 200.0) * 100
            print(f"{resource}: {utilization:.1f}%")


if __name__ == "__main__":
    test_precision_scheduling()
