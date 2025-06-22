#!/usr/bin/env python3
"""
Elegant visualization module with cool color palette and refined layout
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
import json
from typing import Dict, List, Optional
from datetime import datetime

from enums import ResourceType, TaskPriority, RuntimeType
from scheduler import MultiResourceScheduler


class ElegantSchedulerVisualizer:
    """Elegant visualizer with cool tones and minimalist design"""
    
    def __init__(self, scheduler: MultiResourceScheduler):
        self.scheduler = scheduler
        
        # Cool, professional color palette
        self.priority_colors = {
            TaskPriority.CRITICAL: '#7C2D12',    # Dark rust
            TaskPriority.HIGH: '#1E3A8A',        # Navy blue
            TaskPriority.NORMAL: '#065F46',      # Dark teal
            TaskPriority.LOW: '#312E81'          # Indigo
        }
        
        # Alternative palette (more vibrant but still cool)
        self.alt_priority_colors = {
            TaskPriority.CRITICAL: '#B91C1C',    # Crimson
            TaskPriority.HIGH: '#1E40AF',        # Royal blue
            TaskPriority.NORMAL: '#047857',      # Emerald
            TaskPriority.LOW: '#4C1D95'          # Purple
        }
    
    def plot_elegant_gantt(self, 
                          time_window: float = None, 
                          bar_height: float = 0.35,  # Thinner bars
                          spacing: float = 0.8,       # Tighter spacing
                          use_alt_colors: bool = False):
        """Create an elegant Gantt chart with cool tones"""
        if not self.scheduler.schedule_history:
            print("No schedule history available")
            return
        
        # Select color palette
        colors = self.alt_priority_colors if use_alt_colors else self.priority_colors
        
        # Determine time window
        if time_window is None:
            time_window = max(s.end_time for s in self.scheduler.schedule_history) * 1.1
        
        # Prepare resources
        npu_resources = []
        dsp_resources = []
        
        for resource in self.scheduler.resources[ResourceType.NPU]:
            npu_resources.append(resource.unit_id)
        for resource in self.scheduler.resources[ResourceType.DSP]:
            dsp_resources.append(resource.unit_id)
        
        all_resources = npu_resources + dsp_resources
        
        # Create figure with elegant proportions
        fig_height = len(all_resources) * spacing + 2
        fig, ax = plt.subplots(figsize=(20, fig_height))
        
        # Subtle background
        fig.patch.set_facecolor('#FAFAFA')
        ax.set_facecolor('#FFFFFF')
        
        # Resource positions
        y_positions = {}
        current_y = 0
        
        # NPU group
        for i, res in enumerate(npu_resources):
            y_positions[res] = current_y
            current_y += spacing
        
        # Add subtle separator
        if npu_resources and dsp_resources:
            separator_y = current_y - spacing/2
            ax.axhline(y=separator_y, color='#E5E7EB', linewidth=1, alpha=0.5)
        
        # DSP group
        for i, res in enumerate(dsp_resources):
            y_positions[res] = current_y
            current_y += spacing
        
        # Plot tasks with elegant styling
        for schedule in self.scheduler.schedule_history:
            task = self.scheduler.tasks[schedule.task_id]
            
            if task.is_segmented and schedule.sub_segment_schedule:
                self._plot_elegant_segmented_task(ax, task, schedule, y_positions, bar_height, colors)
            else:
                self._plot_elegant_regular_task(ax, task, schedule, y_positions, bar_height, colors)
        
        # Draw minimal bindings
        self._draw_elegant_bindings(ax, y_positions, time_window, spacing)
        
        # Configure axes with minimal style
        ax.set_ylim(-spacing/2, len(all_resources) * spacing - spacing/2)
        ax.set_xlim(-5, time_window * 1.05)
        
        # Y-axis: Minimal resource labels
        ax.set_yticks([y_positions[res] for res in all_resources])
        ax.set_yticklabels(all_resources, fontsize=10, color='#4B5563')
        
        # Subtle resource type indicators
        for i, res in enumerate(all_resources):
            if 'NPU' in res:
                ax.plot(-3, y_positions[res], marker='o', markersize=6, 
                       color='#3B82F6', alpha=0.6)
            else:
                ax.plot(-3, y_positions[res], marker='s', markersize=6, 
                       color='#8B5CF6', alpha=0.6)
        
        # X-axis with minimal styling
        ax.set_xlabel('Time (ms)', fontsize=11, color='#374151')
        ax.set_title('Task Scheduling Timeline', fontsize=14, fontweight='600', 
                    color='#111827', pad=15)
        
        # Minimal grid
        ax.grid(True, axis='x', alpha=0.15, linestyle='-', linewidth=0.5, color='#E5E7EB')
        ax.set_axisbelow(True)
        
        # Clean spines
        for spine in ax.spines.values():
            spine.set_color('#E5E7EB')
            spine.set_linewidth(0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Time markers - subtle
        for t in range(0, int(time_window), 50):
            if t > 0:  # Skip 0
                ax.text(t, -spacing/3, str(t), ha='center', va='top', 
                       fontsize=8, color='#9CA3AF')
        
        # Minimal legend
        self._create_elegant_legend(ax, colors)
        
        # Compact metrics
        self._add_elegant_metrics(ax, time_window)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_elegant_segmented_task(self, ax, task, schedule, y_positions, bar_height, colors):
        """Plot segmented task with elegant minimal style"""
        base_color = colors[task.priority]
        
        # Calculate color variations for segments
        num_segments = len(schedule.sub_segment_schedule)
        
        for i, (sub_seg_id, start_time, end_time) in enumerate(schedule.sub_segment_schedule):
            # Find resource
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
                    
                    # Subtle opacity variation for segments
                    alpha = 0.9 - (i * 0.1 / num_segments)
                    
                    # Main rectangle with rounded corners
                    rect = patches.FancyBboxPatch(
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
                    
                    # Segment separator - thin white line
                    if i < num_segments - 1:
                        ax.plot([end_time, end_time], 
                               [y_pos - bar_height/2 + 2, y_pos + bar_height/2 - 2],
                               color='white', linewidth=1.5, solid_capstyle='round')
                    
                    # Minimal label - only show on first segment
                    if i == 0 and duration > 20:
                        label = f"{task.task_id}"
                        ax.text(start_time + 3, y_pos, label,
                               ha='left', va='center', fontsize=8,
                               color='white', fontweight='500')
    
    def _plot_elegant_regular_task(self, ax, task, schedule, y_positions, bar_height, colors):
        """Plot regular task with elegant style"""
        base_color = colors[task.priority]
        
        for seg in task.segments:
            if seg.resource_type in schedule.assigned_resources:
                resource_id = schedule.assigned_resources[seg.resource_type]
                if resource_id in y_positions:
                    y_pos = y_positions[resource_id]
                    
                    # Calculate duration
                    resource_unit = next((r for r in self.scheduler.resources[seg.resource_type] 
                                        if r.unit_id == resource_id), None)
                    if resource_unit:
                        duration = seg.get_duration(resource_unit.bandwidth)
                        start_time = schedule.start_time + seg.start_time
                        
                        # Rounded rectangle
                        rect = patches.FancyBboxPatch(
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
                        
                        # Minimal label
                        if duration > 20:
                            label = f"{task.task_id}"
                            ax.text(start_time + 3, y_pos, label,
                                   ha='left', va='center', fontsize=8,
                                   color='white', fontweight='500')
    
    def _draw_elegant_bindings(self, ax, y_positions, time_window, spacing):
        """Draw resource bindings with minimal style"""
        for binding in self.scheduler.active_bindings:
            if binding.binding_end <= time_window:
                bound_positions = [y_positions[res_id] for res_id in binding.bound_resources 
                                 if res_id in y_positions]
                
                if len(bound_positions) > 1:
                    min_y = min(bound_positions) - spacing/2
                    max_y = max(bound_positions) + spacing/2
                    
                    # Very subtle binding indicator
                    binding_rect = Rectangle(
                        (binding.binding_start, min_y), 
                        binding.binding_end - binding.binding_start,
                        max_y - min_y,
                        facecolor='#6366F1',
                        alpha=0.03,
                        zorder=0
                    )
                    ax.add_patch(binding_rect)
                    
                    # Minimal vertical lines
                    for x in [binding.binding_start, binding.binding_end]:
                        ax.plot([x, x], [min_y, max_y], 
                               color='#6366F1', linewidth=0.5, 
                               linestyle=':', alpha=0.3)
    
    def _create_elegant_legend(self, ax, colors):
        """Create minimal elegant legend"""
        legend_elements = []
        
        # Priority indicators with rounded style
        for priority, color in colors.items():
            elem = patches.FancyBboxPatch(
                (0, 0), 1, 1,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='none',
                alpha=0.85,
                label=priority.name
            )
            legend_elements.append(elem)
        
        # Create compact legend
        legend = ax.legend(handles=legend_elements, 
                          loc='upper right',
                          frameon=True,
                          fancybox=False,
                          shadow=False,
                          ncol=4,  # Horizontal layout
                          fontsize=9,
                          handlelength=1.5,
                          handletextpad=0.5,
                          columnspacing=1.0)
        
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('#E5E7EB')
        legend.get_frame().set_linewidth(0.5)
        legend.get_frame().set_alpha(0.95)
    
    def _add_elegant_metrics(self, ax, time_window):
        """Add compact metrics display"""
        utilization = self.scheduler.get_resource_utilization(time_window)
        avg_util = sum(utilization.values()) / len(utilization) if utilization else 0
        
        # Compact metrics line
        metrics = (f"Utilization: {avg_util:.0f}% | "
                  f"Segmented: {self.scheduler.segmentation_stats['segmented_tasks']} | "
                  f"Overhead: {self.scheduler.segmentation_stats['total_overhead']:.0f}ms | "
                  f"Complete: {self.scheduler.schedule_history[-1].end_time:.0f}ms")
        
        # Single line metrics
        ax.text(0.02, 0.02, metrics, 
               transform=ax.transAxes, 
               fontsize=9, 
               color='#6B7280',
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', 
                        edgecolor='#E5E7EB',
                        linewidth=0.5,
                        alpha=0.95))
    
    def export_chrome_tracing(self, filename: str = "elegant_trace.json"):
        """Export to Chrome Tracing format"""
        # Same implementation as before
        if not self.scheduler.schedule_history:
            print("No schedule history to export")
            return
        
        events = []
        pid_counter = 1
        resource_mapping = {}
        
        # Create process structure
        for res_type in [ResourceType.NPU, ResourceType.DSP]:
            pid = pid_counter
            pid_counter += 1
            
            events.append({
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "args": {"name": res_type.value}
            })
            
            tid = 1
            for resource in self.scheduler.resources[res_type]:
                resource_mapping[resource.unit_id] = (pid, tid)
                
                events.append({
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": resource.unit_id}
                })
                tid += 1
        
        # Convert schedules
        for schedule in self.scheduler.schedule_history:
            task = self.scheduler.tasks[schedule.task_id]
            
            if task.is_segmented and schedule.sub_segment_schedule:
                for i, (sub_seg_id, start_time, end_time) in enumerate(schedule.sub_segment_schedule):
                    for sub_seg in task.get_sub_segments_for_scheduling():
                        if sub_seg.sub_id == sub_seg_id:
                            if sub_seg.resource_type in schedule.assigned_resources:
                                resource_id = schedule.assigned_resources[sub_seg.resource_type]
                                if resource_id in resource_mapping:
                                    pid, tid = resource_mapping[resource_id]
                                    
                                    event = {
                                        "name": f"{task.task_id}-{i+1}",
                                        "cat": task.priority.name,
                                        "ph": "X",
                                        "ts": int(start_time * 1000),  # 转换为微秒
                                        "dur": int((end_time - start_time) * 1000),
                                        "pid": pid,
                                        "tid": tid,
                                        "args": {
                                            "task": task.name,
                                            "priority": task.priority.name,
                                            "segment": f"{i+1}/{len(schedule.sub_segment_schedule)}",
                                            "start_ms": start_time,
                                            "end_ms": end_time,
                                            "resource": resource_id
                                        }
                                    }
                                    
                                    # 添加颜色
                                    color_map = {
                                        TaskPriority.CRITICAL: "thread_state_runnable",
                                        TaskPriority.HIGH: "rail_animation", 
                                        TaskPriority.NORMAL: "generic_work",
                                        TaskPriority.LOW: "good"
                                    }
                                    event["cname"] = color_map.get(task.priority, "generic_work")
                                    
                                    events.append(event)
            else:
                for seg in task.segments:
                    if seg.resource_type in schedule.assigned_resources:
                        resource_id = schedule.assigned_resources[seg.resource_type]
                        if resource_id in resource_mapping:
                            resource_unit = next((r for r in self.scheduler.resources[seg.resource_type] 
                                                if r.unit_id == resource_id), None)
                            if resource_unit:
                                duration = seg.get_duration(resource_unit.bandwidth)
                                start = schedule.start_time + seg.start_time
                                
                                pid, tid = resource_mapping[resource_id]
                                
                                events.append({
                                    "name": task.task_id,
                                    "cat": task.priority.name,
                                    "ph": "X",
                                    "ts": int(start * 1000),
                                    "dur": int(duration * 1000),
                                    "pid": pid,
                                    "tid": tid,
                                    "args": {
                                        "task": task.name,
                                        "priority": task.priority.name
                                    }
                                })
        
        # Save
        with open(filename, 'w') as f:
            json.dump({"traceEvents": events}, f, indent=2)
        
        print(f"Chrome tracing data exported to {filename}")
