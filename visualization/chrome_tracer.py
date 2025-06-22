#!/usr/bin/env python3
"""
Chrome Tracing Exporter
Chrome追踪格式导出器，用于性能分析
"""

import json
import time
from typing import List, Dict, Any, Optional
from core.enums import ResourceType, TaskPriority


class ChromeTracer:
    """Chrome追踪格式导出器"""
    
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.events = []
        self.process_id = 1
        self.thread_ids = {}
        
    def export(self, filename: str = "trace.json"):
        """导出Chrome追踪文件"""
        self._generate_events()
        self._write_trace_file(filename)
    
    def _generate_events(self):
        """生成追踪事件"""
        self.events = []
        self._assign_thread_ids()
        
        # 添加进程和线程元数据
        self._add_metadata_events()
        
        # 添加任务执行事件
        self._add_task_events()
        
        # 添加资源绑定事件
        self._add_binding_events()
        
        # 添加性能标记
        self._add_performance_markers()
    
    def _assign_thread_ids(self):
        """为每个资源分配线程ID"""
        thread_id = 1
        self.thread_ids = {}
        
        # NPU资源
        for resource in self.scheduler.resources[ResourceType.NPU]:
            self.thread_ids[resource.unit_id] = thread_id
            thread_id += 1
        
        # DSP资源
        for resource in self.scheduler.resources[ResourceType.DSP]:
            self.thread_ids[resource.unit_id] = thread_id
            thread_id += 1
    
    def _add_metadata_events(self):
        """添加元数据事件"""
        # 进程名称
        self.events.append({
            "name": "process_name",
            "ph": "M",
            "pid": self.process_id,
            "args": {"name": "NPU Scheduler"}
        })
        
        # 线程名称
        for resource_id, thread_id in self.thread_ids.items():
            self.events.append({
                "name": "thread_name",
                "ph": "M",
                "pid": self.process_id,
                "tid": thread_id,
                "args": {"name": resource_id}
            })
    
    def _add_task_events(self):
        """添加任务执行事件"""
        for schedule in self.scheduler.schedule_history:
            task = self.scheduler.tasks[schedule.task_id]
            
            # 处理分段任务
            if task.is_segmented and hasattr(schedule, 'sub_segment_schedule') and schedule.sub_segment_schedule:
                self._add_segmented_task_events(task, schedule)
            else:
                self._add_regular_task_events(task, schedule)
    
    def _add_segmented_task_events(self, task, schedule):
        """添加分段任务事件"""
        for i, (sub_seg_id, start_time, end_time) in enumerate(schedule.sub_segment_schedule):
            # 查找对应资源
            sub_seg = None
            for ss in task.get_sub_segments_for_scheduling():
                if ss.sub_id == sub_seg_id:
                    sub_seg = ss
                    break
            
            if sub_seg and sub_seg.resource_type in schedule.assigned_resources:
                resource_id = schedule.assigned_resources[sub_seg.resource_type]
                thread_id = self.thread_ids.get(resource_id)
                
                if thread_id:
                    # 转换为微秒（Chrome Tracing使用微秒）
                    ts = int(start_time * 1000)
                    dur = int((end_time - start_time) * 1000)
                    
                    self.events.append({
                        "name": f"{task.task_id}_seg{i}",
                        "cat": "task_segment",
                        "ph": "X",  # Complete event
                        "ts": ts,
                        "dur": dur,
                        "pid": self.process_id,
                        "tid": thread_id,
                        "args": {
                            "task_id": task.task_id,
                            "task_name": task.name,
                            "priority": task.priority.name,
                            "runtime_type": task.runtime_type.value,
                            "segment_id": sub_seg_id,
                            "resource_type": sub_seg.resource_type.value,
                            "segment_index": i
                        }
                    })
    
    def _add_regular_task_events(self, task, schedule):
        """添加常规任务事件"""
        for seg in task.segments:
            if seg.resource_type in schedule.assigned_resources:
                resource_id = schedule.assigned_resources[seg.resource_type]
                thread_id = self.thread_ids.get(resource_id)
                
                if thread_id:
                    # 计算时间
                    resource_unit = next((r for r in self.scheduler.resources[seg.resource_type] 
                                        if r.unit_id == resource_id), None)
                    if resource_unit:
                        duration = seg.get_duration(resource_unit.bandwidth)
                        start_time = schedule.start_time + seg.start_time
                        
                        # 转换为微秒
                        ts = int(start_time * 1000)
                        dur = int(duration * 1000)
                        
                        self.events.append({
                            "name": task.task_id,
                            "cat": "task",
                            "ph": "X",
                            "ts": ts,
                            "dur": dur,
                            "pid": self.process_id,
                            "tid": thread_id,
                            "args": {
                                "task_id": task.task_id,
                                "task_name": task.name,
                                "priority": task.priority.name,
                                "runtime_type": task.runtime_type.value,
                                "resource_type": seg.resource_type.value,
                                "bandwidth": resource_unit.bandwidth,
                                "is_segmented": task.is_segmented
                            }
                        })
    
    def _add_binding_events(self):
        """添加资源绑定事件"""
        if hasattr(self.scheduler, 'active_bindings'):
            for i, binding in enumerate(self.scheduler.active_bindings):
                # 为每个绑定的资源添加事件
                for resource_id in binding.bound_resources:
                    thread_id = self.thread_ids.get(resource_id)
                    if thread_id:
                        ts = int(binding.binding_start * 1000)
                        dur = int((binding.binding_end - binding.binding_start) * 1000)
                        
                        self.events.append({
                            "name": f"Binding_{i}",
                            "cat": "resource_binding",
                            "ph": "X",
                            "ts": ts,
                            "dur": dur,
                            "pid": self.process_id,
                            "tid": thread_id,
                            "args": {
                                "binding_id": i,
                                "bound_resources": list(binding.bound_resources),
                                "resource_id": resource_id
                            }
                        })
    
    def _add_performance_markers(self):
        """添加性能标记"""
        if not self.scheduler.schedule_history:
            return
        
        # 调度开始标记
        first_start = min(s.start_time for s in self.scheduler.schedule_history)
        self.events.append({
            "name": "Scheduling Start",
            "cat": "marker",
            "ph": "I",  # Instant event
            "ts": int(first_start * 1000),
            "pid": self.process_id,
            "tid": 1,
            "args": {"marker_type": "scheduling_start"}
        })
        
        # 调度结束标记
        last_end = max(s.end_time for s in self.scheduler.schedule_history)
        self.events.append({
            "name": "Scheduling End",
            "cat": "marker",
            "ph": "I",
            "ts": int(last_end * 1000),
            "pid": self.process_id,
            "tid": 1,
            "args": {"marker_type": "scheduling_end"}
        })
        
        # 添加任务优先级标记
        priority_times = {}
        for schedule in self.scheduler.schedule_history:
            task = self.scheduler.tasks[schedule.task_id]
            priority = task.priority.name
            if priority not in priority_times:
                priority_times[priority] = []
            priority_times[priority].append(schedule.start_time)
        
        for priority, times in priority_times.items():
            for time_point in times:
                self.events.append({
                    "name": f"{priority} Task",
                    "cat": "priority",
                    "ph": "I",
                    "ts": int(time_point * 1000),
                    "pid": self.process_id,
                    "tid": 1,
                    "args": {"priority": priority}
                })
    
    def _write_trace_file(self, filename: str):
        """写入追踪文件"""
        trace_data = {
            "traceEvents": self.events,
            "displayTimeUnit": "ms",
            "otherData": {
                "version": "NPU Scheduler Trace",
                "generator": "ChromeTracer",
                "timestamp": time.time(),
                "scheduler_info": {
                    "total_tasks": len(self.scheduler.schedule_history),
                    "total_resources": sum(len(resources) for resources in self.scheduler.resources.values()),
                    "npu_count": len(self.scheduler.resources[ResourceType.NPU]),
                    "dsp_count": len(self.scheduler.resources[ResourceType.DSP]),
                    "segmentation_enabled": getattr(self.scheduler, 'enable_segmentation', False)
                }
            }
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(trace_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Chrome追踪文件已保存: {filename}")
            print(f"📊 包含 {len(self.events)} 个追踪事件")
            self._print_usage_instructions()
            
        except Exception as e:
            print(f"❌ 保存追踪文件失败: {e}")
    
    def _print_usage_instructions(self):
        """打印使用说明"""
        print("\n🔍 查看Chrome追踪文件:")
        print("   1. 打开Chrome浏览器")
        print("   2. 在地址栏输入: chrome://tracing")
        print("   3. 点击 'Load' 按钮")
        print("   4. 选择生成的JSON文件")
        print("   5. 使用以下快捷键导航:")
        print("      • W/S: 放大/缩小")
        print("      • A/D: 左移/右移")
        print("      • 鼠标滚轮: 缩放")
        print("      • 鼠标拖拽: 平移")
    
    def export_performance_summary(self, filename: str = "performance_summary.json"):
        """导出性能摘要"""
        if not self.scheduler.schedule_history:
            print("⚠️  没有调度历史数据")
            return
        
        # 计算性能指标
        latencies = [s.end_time - s.start_time for s in self.scheduler.schedule_history]
        
        # 资源利用率
        resource_usage = {}
        total_time = max(s.end_time for s in self.scheduler.schedule_history)
        
        for schedule in self.scheduler.schedule_history:
            for res_type, res_id in schedule.assigned_resources.items():
                if res_id not in resource_usage:
                    resource_usage[res_id] = 0
                resource_usage[res_id] += schedule.end_time - schedule.start_time
        
        # 计算利用率百分比
        resource_utilization = {
            res_id: (usage / total_time) * 100 
            for res_id, usage in resource_usage.items()
        }
        
        # 优先级统计
        priority_stats = {}
        for schedule in self.scheduler.schedule_history:
            task = self.scheduler.tasks[schedule.task_id]
            priority = task.priority.name
            if priority not in priority_stats:
                priority_stats[priority] = {"count": 0, "total_latency": 0}
            priority_stats[priority]["count"] += 1
            priority_stats[priority]["total_latency"] += schedule.end_time - schedule.start_time
        
        # 计算平均延迟
        for priority, stats in priority_stats.items():
            stats["avg_latency"] = stats["total_latency"] / stats["count"]
        
        summary = {
            "timestamp": time.time(),
            "total_tasks": len(self.scheduler.schedule_history),
            "scheduling_window": total_time,
            "latency_stats": {
                "min": min(latencies),
                "max": max(latencies),
                "avg": sum(latencies) / len(latencies),
                "total": sum(latencies)
            },
            "resource_utilization": resource_utilization,
            "priority_statistics": priority_stats,
            "system_info": {
                "npu_count": len(self.scheduler.resources[ResourceType.NPU]),
                "dsp_count": len(self.scheduler.resources[ResourceType.DSP]),
                "segmentation_enabled": getattr(self.scheduler, 'enable_segmentation', False)
            }
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"📈 性能摘要已保存: {filename}")
        except Exception as e:
            print(f"❌ 保存性能摘要失败: {e}")
    
    def create_minimal_trace(self) -> List[Dict[str, Any]]:
        """创建最小化的追踪数据（用于内存分析）"""
        minimal_events = []
        
        for schedule in self.scheduler.schedule_history:
            task = self.scheduler.tasks[schedule.task_id]
            
            # 只记录基本的任务执行信息
            for res_type, res_id in schedule.assigned_resources.items():
                minimal_events.append({
                    "name": task.task_id,
                    "start": schedule.start_time,
                    "end": schedule.end_time,
                    "resource": res_id,
                    "priority": task.priority.name,
                    "runtime": task.runtime_type.value
                })
        
        return minimal_events


def export_multiple_formats(scheduler, base_filename: str = "scheduler_trace"):
    """导出多种格式的追踪文件"""
    tracer = ChromeTracer(scheduler)
    
    # Chrome追踪格式
    tracer.export(f"{base_filename}.json")
    
    # 性能摘要
    tracer.export_performance_summary(f"{base_filename}_summary.json")
    
    # 最小化追踪（CSV格式）
    minimal_trace = tracer.create_minimal_trace()
    try:
        import csv
        with open(f"{base_filename}_minimal.csv", 'w', newline='', encoding='utf-8') as f:
            if minimal_trace:
                writer = csv.DictWriter(f, fieldnames=minimal_trace[0].keys())
                writer.writeheader()
                writer.writerows(minimal_trace)
        print(f"📄 最小化追踪已保存: {base_filename}_minimal.csv")
    except ImportError:
        print("⚠️  CSV模块不可用，跳过最小化追踪导出")


if __name__ == "__main__":
    # 测试追踪器功能
    print("=== Chrome追踪器测试 ===")
    print("请通过主程序运行来测试追踪功能")