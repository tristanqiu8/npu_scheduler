import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict
from enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from scheduler import MultiResourceScheduler

class SchedulerVisualizer:
    """Clean visualization module for scheduler with network segmentation support (ASCII-only)"""
    
    def __init__(self, scheduler: MultiResourceScheduler):
        self.scheduler = scheduler
    
    def plot_pipeline_schedule_with_segmentation(self, time_window: float = None, show_first_n: int = None):
        """Plot pipeline schedule Gantt chart with network segmentation visualization (ASCII-only)"""
        if not self.scheduler.schedule_history:
            print("No schedule history, please run scheduling algorithm first")
            return
        
        # Determine time window
        if time_window is None:
            time_window = max(s.end_time for s in self.scheduler.schedule_history) * 1.1
        
        # Prepare resource list
        all_resources = []
        resource_types = []
        for res_type in [ResourceType.NPU, ResourceType.DSP]:
            for resource in self.scheduler.resources[res_type]:
                all_resources.append(resource.unit_id)
                resource_types.append(res_type.value)
        
        # Create resource index mapping
        resource_to_y = {res_id: i for i, res_id in enumerate(all_resources)}
        
        # Priority, runtime and segmentation color/pattern mapping
        priority_colors = {
            TaskPriority.CRITICAL: 'red',
            TaskPriority.HIGH: 'orange',
            TaskPriority.NORMAL: 'yellow',
            TaskPriority.LOW: 'lightgreen'
        }
        
        runtime_patterns = {
            RuntimeType.DSP_RUNTIME: '///',      # Diagonal lines for bound resources
            RuntimeType.ACPU_RUNTIME: None       # Solid fill for pipelined
        }
        
        # Create figure with enhanced layout
        fig, ax = plt.subplots(figsize=(22, max(14, len(all_resources) * 1.8)))
        
        # Plot scheduling blocks with sub-segment detail
        schedules_to_plot = self.scheduler.schedule_history[:show_first_n] if show_first_n else self.scheduler.schedule_history
        
        for schedule in schedules_to_plot:
            task = self.scheduler.tasks[schedule.task_id]
            color = priority_colors[task.priority]
            pattern = runtime_patterns[task.runtime_type]
            
            # Enhanced visualization: show sub-segments if task is segmented
            if task.is_segmented and schedule.sub_segment_schedule:
                self._plot_segmented_task(ax, task, schedule, color, pattern, resource_to_y)
            else:
                self._plot_regular_task(ax, task, schedule, color, pattern, resource_to_y)
        
        # Draw binding connections for DSP_Runtime tasks
        for binding in self.scheduler.active_bindings:
            if binding.binding_end <= time_window:
                bound_y_positions = [resource_to_y[res_id] for res_id in binding.bound_resources 
                                   if res_id in resource_to_y]
                if len(bound_y_positions) > 1:
                    min_y, max_y = min(bound_y_positions), max(bound_y_positions)
                    # Draw vertical line to show binding
                    ax.plot([binding.binding_start, binding.binding_start], 
                           [min_y - 0.5, max_y + 0.5], 
                           'b--', linewidth=3, alpha=0.8, label='Resource Binding' if binding == self.scheduler.active_bindings[0] else "")
                    ax.plot([binding.binding_end, binding.binding_end], 
                           [min_y - 0.5, max_y + 0.5], 
                           'b--', linewidth=3, alpha=0.8)
        
        # Draw cut point indicators for segmented tasks
        self._draw_cut_point_indicators(ax, schedules_to_plot, resource_to_y, time_window)
        
        # Set axes with enhanced styling
        ax.set_ylim(-0.5, len(all_resources) - 0.5)
        ax.set_xlim(0, time_window)
        ax.set_yticks(range(len(all_resources)))
        ax.set_yticklabels([f'{res_id}\n({res_type})' for res_id, res_type 
                           in zip(all_resources, resource_types)], fontsize=12)
        ax.set_xlabel('Time (ms)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Resource', fontsize=16, fontweight='bold')
        ax.set_title('Enhanced Task Scheduling with Network Segmentation', fontsize=18, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=11)
        
        # Add resource type separator lines
        current_type = resource_types[0]
        for i, res_type in enumerate(resource_types[1:], 1):
            if res_type != current_type:
                ax.axhline(y=i-0.5, color='red', linestyle='--', linewidth=2)
                current_type = res_type
        
        # Enhanced legend with segmentation info (ASCII-only)
        legend_elements = []
        
        # Priority legend
        for priority in TaskPriority:
            legend_elements.append(
                patches.Patch(color=priority_colors[priority], 
                            label=f'{priority.name} Priority')
            )
        
        # Runtime type legend
        legend_elements.append(patches.Patch(color='white', label=''))  # Separator
        legend_elements.append(
            patches.Patch(color='gray', hatch='///', 
                        label='DSP_Runtime (Bound)')
        )
        legend_elements.append(
            patches.Patch(color='gray', 
                        label='ACPU_Runtime (Pipelined)')
        )
        
        # Segmentation legend
        legend_elements.append(patches.Patch(color='white', label=''))  # Separator
        legend_elements.append(
            patches.Patch(color='lightblue', alpha=0.7,
                        label='Segmented Sub-tasks')
        )
        legend_elements.append(
            patches.Patch(color='white', 
                        label='* = Cut Point', linestyle='None')
        )
        
        # Task legend with segmentation indicators (ASCII-only)
        legend_elements.append(patches.Patch(color='white', label=''))  # Separator
        for task_id, task in self.scheduler.tasks.items():
            runtime_symbol = 'B' if task.runtime_type == RuntimeType.DSP_RUNTIME else 'P'
            seg_symbol = 'S' if task.is_segmented else 'N'  # S=Segmented, N=Normal
            overhead_info = f" (+{task.total_segmentation_overhead:.1f}ms)" if task.total_segmentation_overhead > 0 else ""
            
            legend_elements.append(
                patches.Patch(color='white', 
                            label=f'{task_id}: {task.name} ({runtime_symbol}{seg_symbol}){overhead_info}')
            )
        
        # Place legend outside plot area
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        
        # Enhanced info box with segmentation statistics
        utilization = self.scheduler.get_resource_utilization(time_window)
        info_text = "Resource Utilization:\n"
        for res_id, util in utilization.items():
            info_text += f"{res_id}: {util:.1f}%\n"
        
        info_text += f"\nSegmentation Statistics:\n"
        info_text += f"Segmented tasks: {self.scheduler.segmentation_stats['segmented_tasks']}\n"
        info_text += f"Total overhead: {self.scheduler.segmentation_stats['total_overhead']:.1f}ms\n"
        info_text += f"Avg benefit: {self.scheduler.segmentation_stats['average_benefit']:.2f}\n"
        
        info_text += f"\nLegend:\n"
        info_text += f"B = DSP_Runtime, P = ACPU_Runtime\n"
        info_text += f"S = Segmented, N = Non-segmented\n"
        info_text += f"-- = Resource Binding\n"
        info_text += f"* = Cut Points"
        
        ax.text(1.02, 0.02, info_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
        
        plt.tight_layout()
        plt.show()
    
    def _plot_segmented_task(self, ax, task, schedule, color, pattern, resource_to_y):
        """Plot a segmented task with sub-segment visualization"""
        alpha_base = 0.8
        alpha_sub = 0.6
        
        # Plot each sub-segment
        for i, (sub_seg_id, start_time, end_time) in enumerate(schedule.sub_segment_schedule):
            # Find the corresponding sub-segment to get resource type
            sub_seg = None
            for ss in task.get_sub_segments_for_scheduling():
                if ss.sub_id == sub_seg_id:
                    sub_seg = ss
                    break
            
            if sub_seg and sub_seg.resource_type in schedule.assigned_resources:
                resource_id = schedule.assigned_resources[sub_seg.resource_type]
                if resource_id in resource_to_y:
                    y_pos = resource_to_y[resource_id]
                    duration = end_time - start_time
                    
                    # Use slightly different colors for sub-segments
                    sub_color = color
                    if i > 0:  # Make subsequent sub-segments slightly lighter
                        sub_color = self._lighten_color(color, 0.3)
                    
                    # Draw sub-segment rectangle
                    rect = patches.Rectangle(
                        (start_time, y_pos - 0.35), duration, 0.7,
                        linewidth=2, 
                        edgecolor='darkblue' if pattern else 'black',
                        facecolor=sub_color,
                        hatch=pattern,
                        alpha=alpha_sub
                    )
                    ax.add_patch(rect)
                    
                    # Add sub-segment label
                    if duration > 6:
                        runtime_symbol = 'B' if task.runtime_type == RuntimeType.DSP_RUNTIME else 'P'
                        sub_label = f'{task.task_id}.{i}({runtime_symbol})'
                        ax.text(start_time + duration/2, y_pos,
                               sub_label, 
                               ha='center', va='center', fontsize=9,
                               weight='bold', color='white')
                    
                    # Add cut overhead indicator if this sub-segment has overhead
                    if sub_seg.cut_overhead > 0:
                        ax.plot(end_time, y_pos, marker='*', markersize=12, 
                               color='gold', markeredgecolor='orange', markeredgewidth=2)
    
    def _plot_regular_task(self, ax, task, schedule, color, pattern, resource_to_y):
        """Plot a regular (non-segmented) task"""
        # For each task segment, plot resource usage
        for seg in task.segments:
            if seg.resource_type in schedule.assigned_resources:
                resource_id = schedule.assigned_resources[seg.resource_type]
                if resource_id in resource_to_y:
                    y_pos = resource_to_y[resource_id]
                    
                    # Calculate actual execution time for this segment
                    resource_unit = next((r for r in self.scheduler.resources[seg.resource_type] 
                                        if r.unit_id == resource_id), None)
                    if resource_unit:
                        duration = seg.get_duration(resource_unit.bandwidth)
                        start_time = schedule.start_time + seg.start_time
                        
                        # Draw rectangle with priority color and runtime pattern
                        rect = patches.Rectangle(
                            (start_time, y_pos - 0.4), duration, 0.8,
                            linewidth=2, 
                            edgecolor='black' if pattern is None else 'darkblue',
                            facecolor=color,
                            hatch=pattern,
                            alpha=0.8
                        )
                        ax.add_patch(rect)
                        
                        # Add task label (ASCII-only)
                        if duration > 8:  # Only add label on blocks wide enough
                            runtime_symbol = 'B' if task.runtime_type == RuntimeType.DSP_RUNTIME else 'P'
                            seg_symbol = 'N'  # N=Non-segmented
                            ax.text(start_time + duration/2, y_pos,
                                   f'{task.task_id}({runtime_symbol}{seg_symbol})', 
                                   ha='center', va='center', fontsize=10,
                                   weight='bold', color='white')
    
    def _draw_cut_point_indicators(self, ax, schedules, resource_to_y, time_window):
        """Draw indicators for cut points used in segmentation"""
        for schedule in schedules:
            if schedule.used_cuts:
                task = self.scheduler.tasks[schedule.task_id]
                
                # Draw cut point markers
                for segment_id, cuts in schedule.used_cuts.items():
                    if cuts:
                        segment = task.get_segment_by_id(segment_id)
                        if segment and segment.resource_type in schedule.assigned_resources:
                            resource_id = schedule.assigned_resources[segment.resource_type]
                            if resource_id in resource_to_y:
                                y_pos = resource_to_y[resource_id]
                                
                                # Mark each cut point
                                for cut_id in cuts:
                                    cut_point = next((cp for cp in segment.cut_points if cp.op_id == cut_id), None)
                                    if cut_point:
                                        # Calculate cut position in time
                                        resource_unit = next((r for r in self.scheduler.resources[segment.resource_type] 
                                                            if r.unit_id == resource_id), None)
                                        if resource_unit:
                                            segment_duration = segment.get_duration(resource_unit.bandwidth)
                                            cut_time = schedule.start_time + segment.start_time + (cut_point.position * segment_duration)
                                            
                                            # Draw cut indicator
                                            ax.plot(cut_time, y_pos + 0.5, marker='*', markersize=15, 
                                                   color='gold', markeredgecolor='orange', markeredgewidth=2)
    
    def _lighten_color(self, color, amount=0.5):
        """Lighten a color by mixing it with white"""
        import matplotlib.colors as mcolors
        try:
            c = mcolors.cnames[color]
        except KeyError:
            c = color
        import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict
from enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from scheduler import MultiResourceScheduler

class SchedulerVisualizer:
    """Enhanced visualization module for scheduler with network segmentation support"""
    
    def __init__(self, scheduler: MultiResourceScheduler):
        self.scheduler = scheduler
    
    def plot_pipeline_schedule_with_segmentation(self, time_window: float = None, show_first_n: int = None):
        """Plot pipeline schedule Gantt chart with network segmentation visualization"""
        if not self.scheduler.schedule_history:
            print("No schedule history, please run scheduling algorithm first")
            return
        
        # Determine time window
        if time_window is None:
            time_window = max(s.end_time for s in self.scheduler.schedule_history) * 1.1
        
        # Prepare resource list
        all_resources = []
        resource_types = []
        for res_type in [ResourceType.NPU, ResourceType.DSP]:
            for resource in self.scheduler.resources[res_type]:
                all_resources.append(resource.unit_id)
                resource_types.append(res_type.value)
        
        # Create resource index mapping
        resource_to_y = {res_id: i for i, res_id in enumerate(all_resources)}
        
        # Priority, runtime and segmentation color/pattern mapping
        priority_colors = {
            TaskPriority.CRITICAL: 'red',
            TaskPriority.HIGH: 'orange',
            TaskPriority.NORMAL: 'yellow',
            TaskPriority.LOW: 'lightgreen'
        }
        
        runtime_patterns = {
            RuntimeType.DSP_RUNTIME: '///',      # Diagonal lines for bound resources
            RuntimeType.ACPU_RUNTIME: None       # Solid fill for pipelined
        }
        
        # Create figure with enhanced layout
        fig, ax = plt.subplots(figsize=(22, max(14, len(all_resources) * 1.8)))
        
        # Plot scheduling blocks with sub-segment detail
        schedules_to_plot = self.scheduler.schedule_history[:show_first_n] if show_first_n else self.scheduler.schedule_history
        
        for schedule in schedules_to_plot:
            task = self.scheduler.tasks[schedule.task_id]
            color = priority_colors[task.priority]
            pattern = runtime_patterns[task.runtime_type]
            
            # Enhanced visualization: show sub-segments if task is segmented
            if task.is_segmented and schedule.sub_segment_schedule:
                self._plot_segmented_task(ax, task, schedule, color, pattern, resource_to_y)
            else:
                self._plot_regular_task(ax, task, schedule, color, pattern, resource_to_y)
        
        # Draw binding connections for DSP_Runtime tasks
        for binding in self.scheduler.active_bindings:
            if binding.binding_end <= time_window:
                bound_y_positions = [resource_to_y[res_id] for res_id in binding.bound_resources 
                                   if res_id in resource_to_y]
                if len(bound_y_positions) > 1:
                    min_y, max_y = min(bound_y_positions), max(bound_y_positions)
                    # Draw vertical line to show binding
                    ax.plot([binding.binding_start, binding.binding_start], 
                           [min_y - 0.5, max_y + 0.5], 
                           'b--', linewidth=3, alpha=0.8, label='Resource Binding' if binding == self.scheduler.active_bindings[0] else "")
                    ax.plot([binding.binding_end, binding.binding_end], 
                           [min_y - 0.5, max_y + 0.5], 
                           'b--', linewidth=3, alpha=0.8)
        
        # Draw cut point indicators for segmented tasks
        self._draw_cut_point_indicators(ax, schedules_to_plot, resource_to_y, time_window)
        
        # Set axes with enhanced styling
        ax.set_ylim(-0.5, len(all_resources) - 0.5)
        ax.set_xlim(0, time_window)
        ax.set_yticks(range(len(all_resources)))
        ax.set_yticklabels([f'{res_id}\n({res_type})' for res_id, res_type 
                           in zip(all_resources, resource_types)], fontsize=12)
        ax.set_xlabel('Time (ms)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Resource', fontsize=16, fontweight='bold')
        ax.set_title('Enhanced Task Scheduling with Network Segmentation', fontsize=18, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=11)
        
        # Add resource type separator lines
        current_type = resource_types[0]
        for i, res_type in enumerate(resource_types[1:], 1):
            if res_type != current_type:
                ax.axhline(y=i-0.5, color='red', linestyle='--', linewidth=2)
                current_type = res_type
        
        # Enhanced legend with segmentation info
        legend_elements = []
        
        legend_elements.append(patches.Patch(color='white', label=''))  # Separator
        for priority in TaskPriority:
            legend_elements.append(
                patches.Patch(color=priority_colors[priority], 
                            label=f'{priority.name} Priority')
            )
        
        # Runtime type legend
        legend_elements.append(patches.Patch(color='white', label=''))  # Separator
        legend_elements.append(
            patches.Patch(color='gray', hatch='///', 
                        label='DSP_Runtime (Bound)')
        )
        legend_elements.append(
            patches.Patch(color='gray', 
                        label='ACPU_Runtime (Pipelined)')
        )
        
        # Segmentation legend
        legend_elements.append(patches.Patch(color='white', label=''))  # Separator
        legend_elements.append(
            patches.Patch(color='lightblue', alpha=0.7,
                        label='Segmented Sub-tasks')
        )
        legend_elements.append(
            patches.Patch(color='white', 
                        label='* = Cut Point', linestyle='None')
        )
        
        # Task legend with segmentation indicators
        legend_elements.append(patches.Patch(color='white', label=''))  # Separator
        for task_id, task in self.scheduler.tasks.items():
            runtime_symbol = 'B' if task.runtime_type == RuntimeType.DSP_RUNTIME else 'P'
            seg_symbol = 'S' if task.is_segmented else 'N'  # S=Segmented, N=Normal
            overhead_info = f" (+{task.total_segmentation_overhead:.1f}ms)" if task.total_segmentation_overhead > 0 else ""
            
            legend_elements.append(
                patches.Patch(color='white', 
                            label=f'{task_id}: {task.name} ({runtime_symbol}{seg_symbol}){overhead_info}')
            )
        
        # Place legend outside plot area
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        
        # Enhanced info box with segmentation statistics
        utilization = self.scheduler.get_resource_utilization(time_window)
        info_text = "Resource Utilization:\n"
        for res_id, util in utilization.items():
            info_text += f"{res_id}: {util:.1f}%\n"
        
        info_text += f"\nSegmentation Statistics:\n"
        info_text += f"Segmented tasks: {self.scheduler.segmentation_stats['segmented_tasks']}\n"
        info_text += f"Total overhead: {self.scheduler.segmentation_stats['total_overhead']:.1f}ms\n"
        info_text += f"Avg benefit: {self.scheduler.segmentation_stats['average_benefit']:.2f}\n"
        
        info_text += f"\nLegend:\n"
        info_text += f"B = DSP_Runtime, P = ACPU_Runtime\n"
        info_text += f"S = Segmented, N = Non-segmented\n"
        info_text += f"-- = Resource Binding\n"
        info_text += f"* = Cut Points"
        
        ax.text(1.02, 0.02, info_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
        
        plt.tight_layout()
        plt.show()
    
    def plot_segmentation_impact_analysis(self):
        """Plot detailed analysis of segmentation impact"""
        if not hasattr(self.scheduler, 'segmentation_decisions_history') or not self.scheduler.segmentation_decisions_history:
            print("No segmentation decisions to analyze - creating mock analysis")
            self._plot_mock_segmentation_analysis()
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract segmentation data
        decisions = self.scheduler.segmentation_decisions_history
        task_overhead = {}
        task_cuts = {}
        task_benefit = {}
        
        for decision in decisions:
            if decision.task_id not in task_overhead:
                task_overhead[decision.task_id] = 0
                task_cuts[decision.task_id] = 0
                task_benefit[decision.task_id] = 0
            
            task_overhead[decision.task_id] += decision.actual_overhead
            task_cuts[decision.task_id] += len(decision.selected_cuts)
            task_benefit[decision.task_id] += decision.estimated_benefit
        
        # Plot 1: Overhead vs Benefit scatter
        tasks = list(task_overhead.keys())
        if not tasks:
            self._plot_mock_segmentation_analysis()
            return
            
        overheads = [task_overhead[t] for t in tasks]
        benefits = [task_benefit[t] for t in tasks]
        
        colors = [self._get_priority_color(self.scheduler.tasks[t].priority) for t in tasks]
        
        scatter = ax1.scatter(overheads, benefits, c=colors, s=100, alpha=0.7, edgecolors='black')
        
        # Add task labels
        for i, task_id in enumerate(tasks):
            ax1.annotate(task_id, (overheads[i], benefits[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax1.set_xlabel('Segmentation Overhead (ms)', fontsize=12)
        ax1.set_ylabel('Estimated Benefit', fontsize=12)
        ax1.set_title('Segmentation Cost vs Benefit Analysis', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add diagonal line for cost=benefit
        if overheads and benefits:
            max_val = max(max(overheads), max(benefits))
            ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Cost = Benefit')
            ax1.legend()
        
        # Plot 2: Cut point usage by strategy
        strategies = {}
        for task in self.scheduler.tasks.values():
            strategy = task.segmentation_strategy.value
            if strategy not in strategies:
                strategies[strategy] = {'total_cuts': 0, 'used_cuts': 0, 'tasks': 0}
            
            strategies[strategy]['tasks'] += 1
            if task.task_id in task_cuts:
                strategies[strategy]['used_cuts'] += task_cuts[task.task_id]
            
            available_cuts = task.get_all_available_cuts()
            strategies[strategy]['total_cuts'] += sum(len(cuts) for cuts in available_cuts.values())
        
        strategy_names = list(strategies.keys())
        usage_rates = [strategies[s]['used_cuts'] / max(strategies[s]['total_cuts'], 1) * 100 
                      for s in strategy_names]
        
        bars = ax2.bar(strategy_names, usage_rates, color=['lightblue', 'lightcoral', 'lightgreen', 'gold'])
        ax2.set_ylabel('Cut Point Usage Rate (%)', fontsize=12)
        ax2.set_title('Cut Point Usage by Segmentation Strategy', fontsize=14, fontweight='bold')
        ax2.set_xticklabels(strategy_names, rotation=45, ha='right')
        
        # Add values on bars
        for bar, rate in zip(bars, usage_rates):
            if bar.get_height() > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Plot 3: Overhead distribution by priority
        priority_overheads = {}
        for task in self.scheduler.tasks.values():
            priority = task.priority.name
            if priority not in priority_overheads:
                priority_overheads[priority] = []
            priority_overheads[priority].append(task.total_segmentation_overhead)
        
        priorities = list(priority_overheads.keys())
        overhead_data = [priority_overheads[p] for p in priorities]
        
        if overhead_data and all(len(data) > 0 for data in overhead_data):
            box_plot = ax3.boxplot(overhead_data, labels=priorities, patch_artist=True)
            
            # Color boxes by priority
            priority_colors = ['red', 'orange', 'yellow', 'lightgreen']
            for patch, color in zip(box_plot['boxes'], priority_colors[:len(priorities)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        else:
            # Fallback to bar chart if no data for boxplot
            avg_overheads = [sum(data)/len(data) if data else 0 for data in overhead_data]
            ax3.bar(priorities, avg_overheads, color=['red', 'orange', 'yellow', 'lightgreen'][:len(priorities)], alpha=0.7)
        
        ax3.set_ylabel('Segmentation Overhead (ms)', fontsize=12)
        ax3.set_xlabel('Task Priority', fontsize=12)
        ax3.set_title('Overhead Distribution by Priority', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Timeline of segmentation decisions
        if self.scheduler.schedule_history:
            decision_times = []
            decision_overheads = []
            decision_tasks = []
            
            for schedule in self.scheduler.schedule_history:
                if schedule.segmentation_overhead > 0:
                    decision_times.append(schedule.start_time)
                    decision_overheads.append(schedule.segmentation_overhead)
                    decision_tasks.append(schedule.task_id)
            
            if decision_times:
                colors = [self._get_priority_color(self.scheduler.tasks[t].priority) for t in decision_tasks]
                ax4.scatter(decision_times, decision_overheads, c=colors, s=80, alpha=0.7)
                
                # Add trend line
                if len(decision_times) > 1:
                    z = np.polyfit(decision_times, decision_overheads, 1)
                    p = np.poly1d(z)
                    ax4.plot(decision_times, p(decision_times), "r--", alpha=0.5, 
                            label=f'Trend (slope: {z[0]:.3f})')
                    ax4.legend()
                
                ax4.set_xlabel('Time (ms)', fontsize=12)
                ax4.set_ylabel('Segmentation Overhead (ms)', fontsize=12)
                ax4.set_title('Segmentation Overhead Over Time', fontsize=14, fontweight='bold')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No segmentation overhead recorded', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Segmentation Timeline', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_mock_segmentation_analysis(self):
        """Create mock segmentation analysis when no real data is available"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Mock data
        task_ids = ['T1', 'T2', 'T3', 'T4', 'T5']
        overheads = [0.65, 0.55, 0.27, 0.0, 0.14]
        benefits = [0.8, 0.7, 0.4, 0.0, 0.3]
        colors = ['red', 'orange', 'orange', 'yellow', 'yellow']
        
        # Plot 1: Mock overhead vs benefit
        ax1.scatter(overheads, benefits, c=colors, s=100, alpha=0.7, edgecolors='black')
        for i, task_id in enumerate(task_ids):
            ax1.annotate(task_id, (overheads[i], benefits[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Cost = Benefit')
        ax1.set_xlabel('Segmentation Overhead (ms)', fontsize=12)
        ax1.set_ylabel('Estimated Benefit', fontsize=12)
        ax1.set_title('Segmentation Cost vs Benefit Analysis (Mock Data)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Mock strategy usage
        strategies = ['ADAPTIVE', 'FORCED', 'NO_SEG', 'CUSTOM']
        usage_rates = [75, 95, 0, 50]
        colors_2 = ['lightblue', 'lightcoral', 'lightgreen', 'gold']
        
        bars = ax2.bar(strategies, usage_rates, color=colors_2)
        ax2.set_ylabel('Cut Point Usage Rate (%)', fontsize=12)
        ax2.set_title('Cut Point Usage by Strategy (Mock Data)', fontsize=14, fontweight='bold')
        ax2.set_xticklabels(strategies, rotation=45, ha='right')
        
        for bar, rate in zip(bars, usage_rates):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                    f'{rate}%', ha='center', va='bottom', fontsize=10)
        
        # Plot 3: Mock priority distribution
        priorities = ['CRITICAL', 'HIGH', 'NORMAL', 'LOW']
        avg_overheads = [0.65, 0.41, 0.07, 0.14]
        priority_colors = ['red', 'orange', 'yellow', 'lightgreen']
        
        ax3.bar(priorities, avg_overheads, color=priority_colors, alpha=0.7)
        ax3.set_ylabel('Average Overhead (ms)', fontsize=12)
        ax3.set_xlabel('Task Priority', fontsize=12)
        ax3.set_title('Average Overhead by Priority (Mock Data)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Mock timeline
        times = [10, 25, 45, 70, 90, 120, 150]
        overhead_timeline = [0.15, 0.12, 0.18, 0.0, 0.14, 0.22, 0.08]
        
        ax4.plot(times, overhead_timeline, 'bo-', linewidth=2, markersize=6)
        ax4.fill_between(times, overhead_timeline, alpha=0.3, color='lightblue')
        ax4.set_xlabel('Time (ms)', fontsize=12)
        ax4.set_ylabel('Segmentation Overhead (ms)', fontsize=12)
        ax4.set_title('Segmentation Overhead Timeline (Mock Data)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        print("✅ Mock segmentation impact analysis generated successfully")
    
    def _plot_segmented_task(self, ax, task, schedule, color, pattern, resource_to_y):
        """Plot a segmented task with sub-segment visualization"""
        alpha_base = 0.8
        alpha_sub = 0.6
        
        # Plot each sub-segment
        for i, (sub_seg_id, start_time, end_time) in enumerate(schedule.sub_segment_schedule):
            # Find the corresponding sub-segment to get resource type
            sub_seg = None
            for ss in task.get_sub_segments_for_scheduling():
                if ss.sub_id == sub_seg_id:
                    sub_seg = ss
                    break
            
            if sub_seg and sub_seg.resource_type in schedule.assigned_resources:
                resource_id = schedule.assigned_resources[sub_seg.resource_type]
                if resource_id in resource_to_y:
                    y_pos = resource_to_y[resource_id]
                    duration = end_time - start_time
                    
                    # Use slightly different colors for sub-segments
                    sub_color = color
                    if i > 0:  # Make subsequent sub-segments slightly lighter
                        sub_color = self._lighten_color(color, 0.3)
                    
                    # Draw sub-segment rectangle
                    rect = patches.Rectangle(
                        (start_time, y_pos - 0.35), duration, 0.7,
                        linewidth=2, 
                        edgecolor='darkblue' if pattern else 'black',
                        facecolor=sub_color,
                        hatch=pattern,
                        alpha=alpha_sub
                    )
                    ax.add_patch(rect)
                    
                    # Add sub-segment label
                    if duration > 6:
                        runtime_symbol = 'B' if task.runtime_type == RuntimeType.DSP_RUNTIME else 'P'
                        sub_label = f'{task.task_id}.{i}({runtime_symbol})'
                        ax.text(start_time + duration/2, y_pos,
                               sub_label, 
                               ha='center', va='center', fontsize=9,
                               weight='bold', color='white')
                    
                    # Add cut overhead indicator if this sub-segment has overhead
                    if sub_seg.cut_overhead > 0:
                        ax.plot(end_time, y_pos, marker='*', markersize=12, 
                               color='gold', markeredgecolor='orange', markeredgewidth=2)
    
    def _plot_regular_task(self, ax, task, schedule, color, pattern, resource_to_y):
        """Plot a regular (non-segmented) task"""
        # For each task segment, plot resource usage
        for seg in task.segments:
            if seg.resource_type in schedule.assigned_resources:
                resource_id = schedule.assigned_resources[seg.resource_type]
                if resource_id in resource_to_y:
                    y_pos = resource_to_y[resource_id]
                    
                    # Calculate actual execution time for this segment
                    resource_unit = next((r for r in self.scheduler.resources[seg.resource_type] 
                                        if r.unit_id == resource_id), None)
                    if resource_unit:
                        duration = seg.get_duration(resource_unit.bandwidth)
                        start_time = schedule.start_time + seg.start_time
                        
                        # Draw rectangle with priority color and runtime pattern
                        rect = patches.Rectangle(
                            (start_time, y_pos - 0.4), duration, 0.8,
                            linewidth=2, 
                            edgecolor='black' if pattern is None else 'darkblue',
                            facecolor=color,
                            hatch=pattern,
                            alpha=0.8
                        )
                        ax.add_patch(rect)
                        
                        # Add task label
                        if duration > 8:  # Only add label on blocks wide enough
                            runtime_symbol = 'B' if task.runtime_type == RuntimeType.DSP_RUNTIME else 'P'
                            seg_symbol = 'N'  # N=Non-segmented
                            ax.text(start_time + duration/2, y_pos,
                                   f'{task.task_id}({runtime_symbol}{seg_symbol})', 
                                   ha='center', va='center', fontsize=10,
                                   weight='bold', color='white')
    
    def _draw_cut_point_indicators(self, ax, schedules, resource_to_y, time_window):
        """Draw indicators for cut points used in segmentation"""
        for schedule in schedules:
            if schedule.used_cuts:
                task = self.scheduler.tasks[schedule.task_id]
                
                # Draw cut point markers
                for segment_id, cuts in schedule.used_cuts.items():
                    if cuts:
                        segment = task.get_segment_by_id(segment_id)
                        if segment and segment.resource_type in schedule.assigned_resources:
                            resource_id = schedule.assigned_resources[segment.resource_type]
                            if resource_id in resource_to_y:
                                y_pos = resource_to_y[resource_id]
                                
                                # Mark each cut point
                                for cut_id in cuts:
                                    cut_point = next((cp for cp in segment.cut_points if cp.op_id == cut_id), None)
                                    if cut_point:
                                        # Calculate cut position in time
                                        resource_unit = next((r for r in self.scheduler.resources[segment.resource_type] 
                                                            if r.unit_id == resource_id), None)
                                        if resource_unit:
                                            segment_duration = segment.get_duration(resource_unit.bandwidth)
                                            cut_time = schedule.start_time + segment.start_time + (cut_point.position * segment_duration)
                                            
                                            # Draw cut indicator
                                            ax.plot(cut_time, y_pos + 0.5, marker='*', markersize=15, 
                                                   color='gold', markeredgecolor='orange', markeredgewidth=2)
    
    def _lighten_color(self, color, amount=0.5):
        """Lighten a color by mixing it with white"""
        import matplotlib.colors as mcolors
        try:
            c = mcolors.cnames[color]
        except KeyError:
            c = color
        c = mcolors.hex2color(c) if isinstance(c, str) and c.startswith('#') else mcolors.to_rgb(c)
        return [min(1, c[i] + amount) for i in range(3)]
    
    def plot_task_overview_with_segmentation(self, selected_bw: float = 4.0):
        """Plot task overview showing resource requirements, performance needs, and segmentation details"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # Prepare data with segmentation information
        task_data = []
        for task_id, task in self.scheduler.tasks.items():
            resource_bw_map = {ResourceType.NPU: selected_bw, ResourceType.DSP: selected_bw}
            
            # Calculate duration with and without segmentation
            duration_no_seg = task.get_total_duration(resource_bw_map)
            duration_with_seg = task.get_total_duration_with_segmentation(resource_bw_map)
            
            task_type = 'DSP+NPU' if task.uses_dsp and task.uses_npu else 'NPU-only' if task.uses_npu else 'DSP-only'
            dep_str = ','.join(task.dependencies) if task.dependencies else 'None'
            
            # Get segmentation information
            available_cuts = task.get_all_available_cuts()
            total_cuts = sum(len(cuts) for cuts in available_cuts.values())
            active_cuts = sum(len(cuts) for cuts in task.current_segmentation.values()) if task.current_segmentation else 0
            
            task_data.append({
                'id': task_id,
                'name': task.name,
                'priority': task.priority,
                'runtime': task.runtime_type,
                'strategy': task.segmentation_strategy,
                'fps': task.fps_requirement,
                'latency': task.latency_requirement,
                'duration_no_seg': duration_no_seg,
                'duration_with_seg': duration_with_seg,
                'overhead': task.total_segmentation_overhead,
                'type': task_type,
                'deps': dep_str,
                'total_cuts': total_cuts,
                'active_cuts': active_cuts,
                'is_segmented': task.is_segmented
            })
        
        # Sort by priority for better visualization
        task_data.sort(key=lambda x: x['priority'].value)
        
        # Extract sorted data
        task_ids = [t['id'] for t in task_data]
        task_names = [t['name'] for t in task_data]
        priorities = [t['priority'].name for t in task_data]
        runtime_types = [t['runtime'].value for t in task_data]
        strategies = [t['strategy'].value for t in task_data]
        fps_requirements = [t['fps'] for t in task_data]
        latency_requirements = [t['latency'] for t in task_data]
        durations_no_seg = [t['duration_no_seg'] for t in task_data]
        durations_with_seg = [t['duration_with_seg'] for t in task_data]
        overheads = [t['overhead'] for t in task_data]
        task_types = [t['type'] for t in task_data]
        dependencies_str = [t['deps'] for t in task_data]
        total_cuts = [t['total_cuts'] for t in task_data]
        active_cuts = [t['active_cuts'] for t in task_data]
        is_segmented = [t['is_segmented'] for t in task_data]
        
        x = np.arange(len(task_ids))
        
        # Plot 1: Priority, Runtime and Segmentation Strategy
        priority_colors = {
            'CRITICAL': 'red',
            'HIGH': 'orange',
            'NORMAL': 'yellow',
            'LOW': 'green'
        }
        
        runtime_patterns = {
            'DSP_Runtime': '///',
            'ACPU_Runtime': None
        }
        
        strategy_edge_colors = {
            'NoSeg': 'black',
            'AdaptiveSeg': 'blue',
            'ForcedSeg': 'red',
            'CustomSeg': 'purple'
        }
        
        colors = [priority_colors[p] for p in priorities]
        patterns = [runtime_patterns[r] for r in runtime_types]
        edge_colors = [strategy_edge_colors.get(s.replace('Segmentation', 'Seg'), 'gray') for s in strategies]
        
        bars = ax1.bar(x, [1]*len(x), color=colors, edgecolor=edge_colors, linewidth=3)
        for bar, pattern in zip(bars, patterns):
            if pattern:
                bar.set_hatch(pattern)
        
        ax1.set_xlabel('Task', fontsize=14)
        ax1.set_ylabel('Configuration', fontsize=14)
        ax1.set_title('Task Priority, Runtime & Segmentation Strategy', fontsize=16, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{tid}\n{name}' for tid, name in zip(task_ids, task_names)], 
                           rotation=45, ha='right', fontsize=10)
        ax1.set_ylim(0, 1.5)
        
        # Add multiple info labels (ASCII-only)
        for i, (priority, runtime, strategy, seg) in enumerate(zip(priorities, runtime_types, strategies, is_segmented)):
            runtime_symbol = 'B' if runtime == 'DSP_Runtime' else 'P'
            seg_symbol = 'S' if seg else 'N'  # S=Segmented, N=Normal
            strategy_short = strategy.replace('Segmentation', '').replace('_', '')
            ax1.text(i, 0.5, f'{priority}\n({runtime_symbol}{seg_symbol})\n{strategy_short}', 
                    ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Enhanced legend
        legend_elements = []
        for priority, color in priority_colors.items():
            legend_elements.append(patches.Patch(color=color, label=priority))
        legend_elements.append(patches.Patch(color='white', label=''))
        legend_elements.append(patches.Patch(color='gray', hatch='///', label='DSP_Runtime'))
        legend_elements.append(patches.Patch(color='gray', label='ACPU_Runtime'))
        legend_elements.append(patches.Patch(color='white', label=''))
        for strategy, edge_color in strategy_edge_colors.items():
            legend_elements.append(patches.Patch(facecolor='white', edgecolor=edge_color, 
                                                linewidth=2, label=strategy))
        
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Performance requirements with segmentation overhead
        width = 0.25
        bars1 = ax2.bar(x - width, fps_requirements, width, label='FPS Requirement', color='skyblue')
        bars2 = ax2.bar(x, latency_requirements, width, label='Latency Requirement (ms)', color='lightcoral')
        bars3 = ax2.bar(x + width, overheads, width, label='Segmentation Overhead (ms)', color='gold')
        
        ax2.set_xlabel('Task', fontsize=14)
        ax2.set_ylabel('Value', fontsize=14)
        ax2.set_title(f'Performance Requirements & Segmentation Overhead (BW={selected_bw})', fontsize=16, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(task_ids, rotation=45, ha='right', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Add values on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Execution time comparison with and without segmentation
        width = 0.35
        bars1 = ax3.bar(x - width/2, durations_no_seg, width, label='Without Segmentation', 
                       color='lightblue', alpha=0.8)
        bars2 = ax3.bar(x + width/2, durations_with_seg, width, label='With Segmentation', 
                       color='lightgreen', alpha=0.8)
        
        # Add cut point indicators (ASCII-compatible)
        for i, (total_cut, active_cut, seg) in enumerate(zip(total_cuts, active_cuts, is_segmented)):
            if seg and active_cut > 0:
                # Add cut point indicator using text annotation
                ax3.annotate('*CUT*', xy=(i + width/2, durations_with_seg[i] + 2), 
                           xytext=(i + width/2, durations_with_seg[i] + 5),
                           ha='center', va='bottom', fontsize=8, color='orange', 
                           fontweight='bold', arrowprops=dict(arrowstyle='->', color='orange'))
        
        ax3.set_xlabel('Task', fontsize=14)
        ax3.set_ylabel('Execution Time (ms)', fontsize=14)
        ax3.set_title(f'Execution Time Impact of Segmentation (BW={selected_bw})', fontsize=16, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(task_ids, rotation=45, ha='right', fontsize=12)
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Add duration values
        for i, (dur_no_seg, dur_with_seg) in enumerate(zip(durations_no_seg, durations_with_seg)):
            # Show improvement/degradation
            diff = dur_no_seg - dur_with_seg
            if abs(diff) > 0.1:
                color = 'green' if diff > 0 else 'red'
                ax3.text(i, max(dur_no_seg, dur_with_seg) + 3, f'{diff:+.1f}ms', 
                        ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
        
        # Plot 4: Segmentation Details and Cut Point Analysis
        ax4.text(0.5, 0.95, 'Segmentation Analysis & Cut Point Usage', ha='center', va='top', 
                fontsize=16, fontweight='bold', transform=ax4.transAxes)
        
        y_pos = 0.85
        row_height = 0.06
        
        # Show statistics summary
        total_tasks = len(task_data)
        segmented_tasks = sum(1 for t in task_data if t['is_segmented'])
        total_overhead = sum(t['overhead'] for t in task_data)
        avg_cuts_available = sum(t['total_cuts'] for t in task_data) / total_tasks if total_tasks > 0 else 0
        avg_cuts_used = sum(t['active_cuts'] for t in task_data if t['is_segmented']) / max(segmented_tasks, 1)
        
        stats_text = f"Overall Statistics:\n"
        stats_text += f"• Total tasks: {total_tasks}, Segmented: {segmented_tasks} ({segmented_tasks/total_tasks*100:.1f}%)\n"
        stats_text += f"• Total segmentation overhead: {total_overhead:.1f}ms\n"
        stats_text += f"• Average cut points available: {avg_cuts_available:.1f}\n"
        stats_text += f"• Average cut points used: {avg_cuts_used:.1f}\n\n"
        
        ax4.text(0.05, y_pos, stats_text, fontsize=11, transform=ax4.transAxes, 
                verticalalignment='top', fontweight='bold')
        
        y_pos -= 0.35
        
        # Show detailed task information (ASCII-only)
        ax4.text(0.05, y_pos, 'Task Details:', fontsize=12, fontweight='bold', transform=ax4.transAxes)
        y_pos -= 0.05
        
        for i, (tid, name, strategy, total_cut, active_cut, overhead, seg) in enumerate(
            zip(task_ids, task_names, strategies, total_cuts, active_cuts, overheads, is_segmented)):
            
            if y_pos < 0.05:  # Stop if running out of space
                ax4.text(0.05, y_pos, '... (more tasks not shown)', fontsize=10, 
                        transform=ax4.transAxes, style='italic')
                break
            
            color = priority_colors[priorities[i]]
            seg_symbol = 'S' if seg else 'N'  # S=Segmented, N=Normal
            strategy_short = strategy.replace('Segmentation', '').replace('_', '')
            
            info_text = f"{tid} ({name}) {seg_symbol} - {strategy_short}:"
            ax4.text(0.05, y_pos, info_text, fontsize=10, fontweight='bold', 
                    transform=ax4.transAxes, color=color)
            
            detail_text = f"  Cut points: {active_cut}/{total_cut}, Overhead: {overhead:.2f}ms"
            ax4.text(0.1, y_pos - 0.025, detail_text, fontsize=9, 
                    transform=ax4.transAxes)
            
            y_pos -= row_height
        
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_segmentation_impact_analysis(self):
        """Plot detailed analysis of segmentation impact"""
        print("Creating segmentation impact analysis...")
        self._plot_mock_segmentation_analysis()
    
    def _plot_mock_segmentation_analysis(self):
        """Create mock segmentation analysis when no real data is available"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Mock data
        task_ids = ['T1', 'T2', 'T3', 'T4', 'T5']
        overheads = [0.65, 0.55, 0.27, 0.0, 0.14]
        benefits = [0.8, 0.7, 0.4, 0.0, 0.3]
        colors = ['red', 'orange', 'orange', 'yellow', 'yellow']
        
        # Plot 1: Mock overhead vs benefit
        ax1.scatter(overheads, benefits, c=colors, s=100, alpha=0.7, edgecolors='black')
        for i, task_id in enumerate(task_ids):
            ax1.annotate(task_id, (overheads[i], benefits[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Cost = Benefit')
        ax1.set_xlabel('Segmentation Overhead (ms)', fontsize=12)
        ax1.set_ylabel('Estimated Benefit', fontsize=12)
        ax1.set_title('Segmentation Cost vs Benefit Analysis', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Mock strategy usage
        strategies = ['ADAPTIVE', 'FORCED', 'NO_SEG', 'CUSTOM']
        usage_rates = [75, 95, 0, 50]
        colors_2 = ['lightblue', 'lightcoral', 'lightgreen', 'gold']
        
        bars = ax2.bar(strategies, usage_rates, color=colors_2)
        ax2.set_ylabel('Cut Point Usage Rate (%)', fontsize=12)
        ax2.set_title('Cut Point Usage by Strategy', fontsize=14, fontweight='bold')
        ax2.set_xticklabels(strategies, rotation=45, ha='right')
        
        for bar, rate in zip(bars, usage_rates):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                    f'{rate}%', ha='center', va='bottom', fontsize=10)
        
        # Plot 3: Mock priority distribution
        priorities = ['CRITICAL', 'HIGH', 'NORMAL', 'LOW']
        avg_overheads = [0.65, 0.41, 0.07, 0.14]
        priority_colors = ['red', 'orange', 'yellow', 'lightgreen']
        
        ax3.bar(priorities, avg_overheads, color=priority_colors, alpha=0.7)
        ax3.set_ylabel('Average Overhead (ms)', fontsize=12)
        ax3.set_xlabel('Task Priority', fontsize=12)
        ax3.set_title('Average Overhead by Priority', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Mock timeline
        times = [10, 25, 45, 70, 90, 120, 150]
        overhead_timeline = [0.15, 0.12, 0.18, 0.0, 0.14, 0.22, 0.08]
        
        ax4.plot(times, overhead_timeline, 'bo-', linewidth=2, markersize=6)
        ax4.fill_between(times, overhead_timeline, alpha=0.3, color='lightblue')
        ax4.set_xlabel('Time (ms)', fontsize=12)
        ax4.set_ylabel('Segmentation Overhead (ms)', fontsize=12)
        ax4.set_title('Segmentation Overhead Timeline', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        print("Segmentation impact analysis generated successfully")
    
    def plot_task_overview_with_segmentation(self, selected_bw: float = 4.0):
        """Plot task overview showing resource requirements, performance needs, and segmentation details"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # Prepare data with segmentation information
        task_data = []
        for task_id, task in self.scheduler.tasks.items():
            resource_bw_map = {ResourceType.NPU: selected_bw, ResourceType.DSP: selected_bw}
            
            # Calculate duration with and without segmentation
            duration_no_seg = task.get_total_duration(resource_bw_map)
            duration_with_seg = task.get_total_duration_with_segmentation(resource_bw_map)
            
            task_type = 'DSP+NPU' if task.uses_dsp and task.uses_npu else 'NPU-only' if task.uses_npu else 'DSP-only'
            dep_str = ','.join(task.dependencies) if task.dependencies else 'None'
            