import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict
from enums import ResourceType, TaskPriority, RuntimeType
from scheduler import MultiResourceScheduler

class SchedulerVisualizer:
    """Visualization module for scheduler with runtime configuration support"""
    
    def __init__(self, scheduler: MultiResourceScheduler):
        self.scheduler = scheduler
    
    def plot_pipeline_schedule_with_runtime(self, time_window: float = None, show_first_n: int = None):
        """Plot pipeline schedule Gantt chart with runtime configuration visualization"""
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
        
        # Priority and runtime color mapping
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
        
        # Create figure
        fig, ax = plt.subplots(figsize=(20, max(12, len(all_resources) * 1.5)))
        
        # Plot scheduling blocks
        schedules_to_plot = self.scheduler.schedule_history[:show_first_n] if show_first_n else self.scheduler.schedule_history
        
        for schedule in schedules_to_plot:
            task = self.scheduler.tasks[schedule.task_id]
            color = priority_colors[task.priority]
            pattern = runtime_patterns[task.runtime_type]
            
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
                                ax.text(start_time + duration/2, y_pos,
                                       f'{task.task_id}({runtime_symbol})', 
                                       ha='center', va='center', fontsize=10,
                                       weight='bold', color='white')
        
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
                           'b--', linewidth=2, alpha=0.7)
                    ax.plot([binding.binding_end, binding.binding_end], 
                           [min_y - 0.5, max_y + 0.5], 
                           'b--', linewidth=2, alpha=0.7)
        
        # Set axes
        ax.set_ylim(-0.5, len(all_resources) - 0.5)
        ax.set_xlim(0, time_window)
        ax.set_yticks(range(len(all_resources)))
        ax.set_yticklabels([f'{res_id}\n({res_type})' for res_id, res_type 
                           in zip(all_resources, resource_types)], fontsize=12)
        ax.set_xlabel('Time (ms)', fontsize=14)
        ax.set_ylabel('Resource', fontsize=14)
        ax.set_title('Priority-Aware Task Scheduling with Runtime Configurations', fontsize=16, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        ax.tick_params(axis='x', labelsize=12)
        
        # Add resource type separator lines
        current_type = resource_types[0]
        for i, res_type in enumerate(resource_types[1:], 1):
            if res_type != current_type:
                ax.axhline(y=i-0.5, color='red', linestyle='--', linewidth=2)
                current_type = res_type
        
        # Add legend with priorities and runtime types
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
        
        # Task legend
        legend_elements.append(patches.Patch(color='white', label=''))  # Separator
        for task_id, task in self.scheduler.tasks.items():
            runtime_symbol = 'B' if task.runtime_type == RuntimeType.DSP_RUNTIME else 'P'
            legend_elements.append(
                patches.Patch(color='white', 
                            label=f'{task_id}: {task.name} ({runtime_symbol})')
            )
        
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)
        
        # Add resource utilization info
        utilization = self.scheduler.get_resource_utilization(time_window)
        util_text = "Resource Utilization:\n"
        for res_id, util in utilization.items():
            util_text += f"{res_id}: {util:.1f}%\n"
        
        util_text += f"\nRuntime Legend:\n"
        util_text += f"B = DSP_Runtime (Bound)\n"
        util_text += f"P = ACPU_Runtime (Pipelined)\n"
        util_text += f"-- = Resource Binding"
        
        ax.text(1.02, 0.02, util_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    def plot_task_overview_with_runtime(self, selected_bw: float = 4.0):
        """Plot task overview showing resource requirements, performance needs, and runtime types"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data
        task_data = []
        for task_id, task in self.scheduler.tasks.items():
            resource_bw_map = {ResourceType.NPU: selected_bw, ResourceType.DSP: selected_bw}
            duration = task.get_total_duration(resource_bw_map)
            
            task_type = 'DSP+NPU' if task.uses_dsp and task.uses_npu else 'NPU-only' if task.uses_npu else 'DSP-only'
            dep_str = ','.join(task.dependencies) if task.dependencies else 'None'
            
            task_data.append({
                'id': task_id,
                'name': task.name,
                'priority': task.priority,
                'runtime': task.runtime_type,
                'fps': task.fps_requirement,
                'latency': task.latency_requirement,
                'duration': duration,
                'type': task_type,
                'deps': dep_str
            })
        
        # Sort by priority for better visualization
        task_data.sort(key=lambda x: x['priority'].value)
        
        # Extract sorted data
        task_ids = [t['id'] for t in task_data]
        task_names = [t['name'] for t in task_data]
        priorities = [t['priority'].name for t in task_data]
        runtime_types = [t['runtime'].value for t in task_data]
        fps_requirements = [t['fps'] for t in task_data]
        latency_requirements = [t['latency'] for t in task_data]
        task_durations = [t['duration'] for t in task_data]
        task_types = [t['type'] for t in task_data]
        dependencies_str = [t['deps'] for t in task_data]
        
        x = np.arange(len(task_ids))
        
        # Plot 1: Priority and Runtime distribution
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
        
        colors = [priority_colors[p] for p in priorities]
        patterns = [runtime_patterns[r] for r in runtime_types]
        
        bars = ax1.bar(x, [1]*len(x), color=colors)
        for bar, pattern in zip(bars, patterns):
            if pattern:
                bar.set_hatch(pattern)
        
        ax1.set_xlabel('Task', fontsize=14)
        ax1.set_ylabel('Priority Level', fontsize=14)
        ax1.set_title('Task Priority & Runtime Distribution', fontsize=16, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{tid}\n{name}' for tid, name in zip(task_ids, task_names)], 
                           rotation=45, ha='right', fontsize=12)
        ax1.set_ylim(0, 1.5)
        
        # Add priority and runtime labels
        for i, (priority, runtime) in enumerate(zip(priorities, runtime_types)):
            runtime_symbol = 'B' if runtime == 'DSP_Runtime' else 'P'
            ax1.text(i, 0.5, f'{priority}\n({runtime_symbol})', ha='center', va='center', 
                    fontsize=10, fontweight='bold')
        
        # Create custom legend
        legend_elements = []
        for priority, color in priority_colors.items():
            legend_elements.append(patches.Patch(color=color, label=priority))
        legend_elements.append(patches.Patch(color='white', label=''))  # Separator
        legend_elements.append(patches.Patch(color='gray', hatch='///', label='DSP_Runtime'))
        legend_elements.append(patches.Patch(color='gray', label='ACPU_Runtime'))
        
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Performance requirements
        width = 0.35
        bars1 = ax2.bar(x - width/2, fps_requirements, width, label='FPS Requirement', color='skyblue')
        bars2 = ax2.bar(x + width/2, latency_requirements, width, label='Latency Requirement (ms)', color='lightcoral')
        
        ax2.set_xlabel('Task', fontsize=14)
        ax2.set_ylabel('Value', fontsize=14)
        ax2.set_title(f'Task Performance Requirements (BW={selected_bw})', fontsize=16, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(task_ids, rotation=45, ha='right', fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=11)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=11)
        
        # Plot 3: Execution time by task type and runtime
        type_colors = {'NPU-only': 'green', 'DSP+NPU': 'orange', 'DSP-only': 'blue'}
        bar_colors = [type_colors.get(t, 'gray') for t in task_types]
        
        bars3 = ax3.bar(x, task_durations, color=bar_colors)
        
        # Add runtime patterns
        for bar, runtime in zip(bars3, runtime_types):
            if runtime == 'DSP_Runtime':
                bar.set_hatch('///')
        
        ax3.set_xlabel('Task', fontsize=14)
        ax3.set_ylabel('Execution Time (ms)', fontsize=14)
        ax3.set_title(f'Task Execution Time, Resource Type & Runtime (BW={selected_bw})', fontsize=16, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(task_ids, rotation=45, ha='right', fontsize=12)
        
        # Add legend
        legend_elements = []
        for task_type, color in type_colors.items():
            legend_elements.append(patches.Patch(color=color, label=task_type))
        legend_elements.append(patches.Patch(color='white', label=''))  # Separator
        legend_elements.append(patches.Patch(color='gray', hatch='///', label='DSP_Runtime'))
        legend_elements.append(patches.Patch(color='gray', label='ACPU_Runtime'))
        
        ax3.legend(handles=legend_elements, loc='upper right', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, duration in zip(bars3, task_durations):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{duration:.1f}', ha='center', va='bottom', fontsize=11)
        
        # Plot 4: Dependencies and Runtime Info
        ax4.text(0.5, 0.95, 'Task Dependencies & Runtime Info', ha='center', va='top', 
                fontsize=16, fontweight='bold', transform=ax4.transAxes)
        
        y_pos = 0.85
        for i, (tid, deps, priority, runtime) in enumerate(zip(task_ids, dependencies_str, priorities, runtime_types)):
            color = priority_colors[priority]
            runtime_symbol = 'B' if runtime == 'DSP_Runtime' else 'P'
            
            ax4.text(0.05, y_pos - i*0.08, f'{tid} ({runtime_symbol}):', fontsize=12, fontweight='bold', 
                    transform=ax4.transAxes, color=color)
            ax4.text(0.35, y_pos - i*0.08, f'Depends on {deps}', fontsize=12, 
                    transform=ax4.transAxes)
        
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()