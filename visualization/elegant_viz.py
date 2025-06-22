#!/usr/bin/env python3
"""
Elegant Scheduler Visualizer
优雅的调度器可视化模块，提供现代化的图表展示
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import time

from core.enums import ResourceType, TaskPriority, RuntimeType


class ElegantSchedulerVisualizer:
    """优雅的调度器可视化器"""
    
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._setup_style()
    
    def _setup_style(self):
        """设置现代化的可视化风格"""
        # 设置现代色彩方案
        self.colors = {
            TaskPriority.CRITICAL: '#EF4444',    # 红色
            TaskPriority.HIGH: '#F97316',        # 橙色  
            TaskPriority.NORMAL: '#EAB308',      # 黄色
            TaskPriority.LOW: '#22C55E'          # 绿色
        }
        
        # 设置matplotlib参数
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 10,
            'axes.linewidth': 0.5,
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            'xtick.minor.size': 2,
            'ytick.minor.size': 2,
            'axes.spines.top': False,
            'axes.spines.right': False
        })
    
    def plot_elegant_gantt(self, time_window: float = None, bar_height: float = 0.35, 
                          spacing: float = 0.8, figsize: Tuple[int, int] = None):
        """绘制优雅的甘特图"""
        
        if not self.scheduler.schedule_history:
            print("⚠️  没有调度历史，请先运行调度算法")
            return
        
        # 计算时间窗口
        if time_window is None:
            time_window = max(s.end_time for s in self.scheduler.schedule_history) * 1.1
        
        # 收集所有资源
        npu_resources = [r.unit_id for r in self.scheduler.resources[ResourceType.NPU]]
        dsp_resources = [r.unit_id for r in self.scheduler.resources[ResourceType.DSP]] 
        all_resources = npu_resources + dsp_resources
        
        # 设置图形尺寸
        if figsize is None:
            fig_height = max(8, len(all_resources) * spacing + 2)
            figsize = (20, fig_height)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 设置现代化背景
        fig.patch.set_facecolor('#FAFAFA')
        ax.set_facecolor('#FFFFFF')
        
        # 创建资源位置映射
        y_positions = {}
        current_y = 0
        
        # NPU资源组
        for res in npu_resources:
            y_positions[res] = current_y
            current_y += spacing
        
        # 添加分组分隔符
        if npu_resources and dsp_resources:
            separator_y = current_y - spacing/2
            ax.axhline(y=separator_y, color='#E5E7EB', linewidth=1, alpha=0.5)
        
        # DSP资源组
        for res in dsp_resources:
            y_positions[res] = current_y
            current_y += spacing
        
        # 绘制任务
        for schedule in self.scheduler.schedule_history:
            task = self.scheduler.tasks[schedule.task_id]
            
            if task.is_segmented and hasattr(schedule, 'sub_segment_schedule') and schedule.sub_segment_schedule:
                self._plot_segmented_task(ax, task, schedule, y_positions, bar_height)
            else:
                self._plot_regular_task(ax, task, schedule, y_positions, bar_height)
        
        # 绘制资源绑定
        self._draw_resource_bindings(ax, y_positions, time_window, spacing)
        
        # 配置坐标轴
        self._configure_axes(ax, all_resources, y_positions, time_window, spacing)
        
        # 添加图例和指标
        self._add_legend_and_metrics(ax, time_window)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_segmented_task(self, ax, task, schedule, y_positions, bar_height):
        """绘制分段任务"""
        base_color = self.colors[task.priority]
        num_segments = len(schedule.sub_segment_schedule)
        
        for i, (sub_seg_id, start_time, end_time) in enumerate(schedule.sub_segment_schedule):
            # 查找对应的子段
            sub_seg = None
            for ss in task.get_sub_segments_for_scheduling():
                if ss.sub_id == sub_seg_id:
                    sub_seg = ss
                    break
            
            if sub_seg and sub_seg.resource_type in schedule.assigned_resources:
                resource_id = schedule.assigned_resources[sub_seg.resource_type]
                if resource_id in y_positions:
                    y_pos = y_positions[resource_id]
                    duration = end_time - start_time
                    
                    # 不同段的透明度变化
                    alpha = 0.9 - (i * 0.1 / num_segments)
                    
                    # 绘制圆角矩形
                    rect = FancyBboxPatch(
                        (start_time, y_pos - bar_height/2), 
                        duration, 
                        bar_height,
                        boxstyle="round,pad=0.01,rounding_size=0.02",
                        facecolor=base_color,
                        edgecolor='none',
                        alpha=alpha,
                        linewidth=0
                    )
                    ax.add_patch(rect)
                    
                    # 段间分隔线
                    if i < num_segments - 1:
                        ax.plot([end_time, end_time], 
                               [y_pos - bar_height/2 + 2, y_pos + bar_height/2 - 2],
                               color='white', linewidth=1.5, solid_capstyle='round')
                    
                    # 标签（仅在第一段）
                    if i == 0 and duration > 20:
                        ax.text(start_time + 3, y_pos, f"{task.task_id}",
                               ha='left', va='center', fontsize=8,
                               color='white', fontweight='500')
    
    def _plot_regular_task(self, ax, task, schedule, y_positions, bar_height):
        """绘制常规任务"""
        base_color = self.colors[task.priority]
        
        for seg in task.segments:
            if seg.resource_type in schedule.assigned_resources:
                resource_id = schedule.assigned_resources[seg.resource_type]
                if resource_id in y_positions:
                    y_pos = y_positions[resource_id]
                    
                    # 计算持续时间
                    resource_unit = next((r for r in self.scheduler.resources[seg.resource_type] 
                                        if r.unit_id == resource_id), None)
                    if resource_unit:
                        duration = seg.get_duration(resource_unit.bandwidth)
                        start_time = schedule.start_time + seg.start_time
                        
                        # 绘制圆角矩形
                        rect = FancyBboxPatch(
                            (start_time, y_pos - bar_height/2), 
                            duration, 
                            bar_height,
                            boxstyle="round,pad=0.01,rounding_size=0.02",
                            facecolor=base_color,
                            edgecolor='none',
                            alpha=0.85,
                            linewidth=0
                        )
                        ax.add_patch(rect)
                        
                        # 标签
                        if duration > 20:
                            ax.text(start_time + 3, y_pos, f"{task.task_id}",
                                   ha='left', va='center', fontsize=8,
                                   color='white', fontweight='500')
    
    def _draw_resource_bindings(self, ax, y_positions, time_window, spacing):
        """绘制资源绑定关系"""
        if hasattr(self.scheduler, 'active_bindings'):
            for binding in self.scheduler.active_bindings:
                if binding.binding_end <= time_window:
                    bound_positions = [y_positions[res_id] for res_id in binding.bound_resources 
                                     if res_id in y_positions]
                    
                    if len(bound_positions) > 1:
                        min_y = min(bound_positions) - spacing/2
                        max_y = max(bound_positions) + spacing/2
                        
                        # 绑定区域
                        binding_rect = Rectangle(
                            (binding.binding_start, min_y), 
                            binding.binding_end - binding.binding_start,
                            max_y - min_y,
                            facecolor='#6366F1',
                            alpha=0.03,
                            zorder=0
                        )
                        ax.add_patch(binding_rect)
                        
                        # 绑定边界线
                        for x in [binding.binding_start, binding.binding_end]:
                            ax.plot([x, x], [min_y, max_y], 
                                   color='#6366F1', linewidth=0.5, 
                                   linestyle=':', alpha=0.3)
    
    def _configure_axes(self, ax, all_resources, y_positions, time_window, spacing):
        """配置坐标轴样式"""
        # Y轴设置
        ax.set_ylim(-spacing/2, len(all_resources) * spacing - spacing/2)
        ax.set_yticks([y_positions[res] for res in all_resources])
        ax.set_yticklabels(all_resources, fontsize=10, color='#4B5563')
        
        # 资源类型指示器
        for res in all_resources:
            marker = 'o' if 'NPU' in res else 's'
            color = '#3B82F6' if 'NPU' in res else '#8B5CF6'
            ax.plot(-3, y_positions[res], marker=marker, markersize=6, 
                   color=color, alpha=0.6)
        
        # X轴设置
        ax.set_xlim(-5, time_window * 1.05)
        ax.set_xlabel('Time (ms)', fontsize=11, color='#374151')
        ax.set_title('Task Scheduling Timeline', fontsize=14, fontweight='600', 
                    color='#111827', pad=15)
        
        # 网格
        ax.grid(True, axis='x', alpha=0.15, linestyle='-', linewidth=0.5, color='#E5E7EB')
        ax.set_axisbelow(True)
        
        # 边框
        for spine in ax.spines.values():
            spine.set_color('#E5E7EB')
            spine.set_linewidth(0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 时间标记
        for t in range(0, int(time_window), 50):
            if t > 0:
                ax.text(t, -spacing/3, str(t), ha='center', va='top', 
                       fontsize=8, color='#9CA3AF')
    
    def _add_legend_and_metrics(self, ax, time_window):
        """添加图例和性能指标"""
        # 创建图例
        legend_elements = []
        for priority, color in self.colors.items():
            elem = FancyBboxPatch(
                (0, 0), 1, 1,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='none',
                alpha=0.85,
                label=priority.name
            )
            legend_elements.append(elem)
        
        legend = ax.legend(handles=legend_elements, loc='upper right', 
                          frameon=True, fancybox=True, shadow=False,
                          fontsize=9, title='Priority')
        legend.get_frame().set_facecolor('#FFFFFF')
        legend.get_frame().set_alpha(0.9)
        
        # 添加简单指标
        if self.scheduler.schedule_history:
            total_tasks = len(self.scheduler.schedule_history)
            avg_latency = np.mean([s.end_time - s.start_time for s in self.scheduler.schedule_history])
            
            metrics_text = f"Tasks: {total_tasks} | Avg Latency: {avg_latency:.1f}ms"
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   fontsize=9, color='#6B7280', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='#F9FAFB', alpha=0.8))
    
    def export_chrome_tracing(self, filename: str = "trace.json"):
        """导出Chrome追踪格式文件"""
        from .chrome_tracer import ChromeTracer
        
        tracer = ChromeTracer(self.scheduler)
        tracer.export(filename)
        print(f"📊 Chrome追踪文件已保存为: {filename}")
    
    def plot_performance_analysis(self):
        """绘制性能分析图表"""
        if not self.scheduler.schedule_history:
            print("⚠️  没有调度历史数据")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Analysis Dashboard', fontsize=16, fontweight='600')
        
        # 1. 任务延迟分布
        latencies = [s.end_time - s.start_time for s in self.scheduler.schedule_history]
        ax1.hist(latencies, bins=10, color='#3B82F6', alpha=0.7, edgecolor='white')
        ax1.set_title('Task Latency Distribution')
        ax1.set_xlabel('Latency (ms)')
        ax1.set_ylabel('Count')
        
        # 2. 资源利用率
        resource_usage = {}
        for schedule in self.scheduler.schedule_history:
            for res_type, res_id in schedule.assigned_resources.items():
                if res_id not in resource_usage:
                    resource_usage[res_id] = 0
                resource_usage[res_id] += schedule.end_time - schedule.start_time
        
        resources = list(resource_usage.keys())
        usage = list(resource_usage.values())
        colors = ['#3B82F6' if 'NPU' in r else '#8B5CF6' for r in resources]
        
        ax2.bar(resources, usage, color=colors, alpha=0.7)
        ax2.set_title('Resource Utilization')
        ax2.set_ylabel('Total Usage (ms)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 优先级分布
        priority_counts = {}
        for schedule in self.scheduler.schedule_history:
            task = self.scheduler.tasks[schedule.task_id]
            priority = task.priority.name
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        priorities = list(priority_counts.keys())
        counts = list(priority_counts.values())
        pie_colors = [self.colors[TaskPriority[p]] for p in priorities]
        
        ax3.pie(counts, labels=priorities, colors=pie_colors, autopct='%1.1f%%')
        ax3.set_title('Task Priority Distribution')
        
        # 4. 时间线分析
        timeline = []
        for schedule in self.scheduler.schedule_history:
            timeline.append((schedule.start_time, 'start', schedule.task_id))
            timeline.append((schedule.end_time, 'end', schedule.task_id))
        
        timeline.sort()
        active_tasks = 0
        times = []
        loads = []
        
        for time, event, task_id in timeline:
            times.append(time)
            if event == 'start':
                active_tasks += 1
            else:
                active_tasks -= 1
            loads.append(active_tasks)
        
        ax4.plot(times, loads, color='#10B981', linewidth=2)
        ax4.fill_between(times, loads, alpha=0.3, color='#10B981')
        ax4.set_title('System Load Over Time')
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('Active Tasks')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 测试可视化功能
    print("=== 可视化模块测试 ===")
    
    # 这里需要一个调度器实例来测试
    # 在实际使用中，调度器会从主程序传入
    print("请通过主程序运行来测试可视化功能")
