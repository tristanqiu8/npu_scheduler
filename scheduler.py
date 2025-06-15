from typing import List, Dict, Set, Optional
from collections import defaultdict
from enums import ResourceType, TaskPriority, RuntimeType
from models import ResourceUnit, TaskScheduleInfo, ResourceBinding
from task import NNTask
from priority_queue import ResourcePriorityQueues

class MultiResourceScheduler:
    """Multi-resource scheduler with runtime configuration support"""
    
    def __init__(self):
        self.tasks: Dict[str, NNTask] = {}
        self.resources: Dict[ResourceType, List[ResourceUnit]] = {
            ResourceType.NPU: [],
            ResourceType.DSP: []
        }
        # Priority queues for each resource
        self.resource_queues: Dict[str, ResourcePriorityQueues] = {}
        self.schedule_history: List[TaskScheduleInfo] = []
        # Track resource bindings for DSP_Runtime tasks
        self.active_bindings: List[ResourceBinding] = []
        
    def add_npu(self, npu_id: str, bandwidth: float):
        """Add NPU resource"""
        npu = ResourceUnit(npu_id, ResourceType.NPU, bandwidth)
        self.resources[ResourceType.NPU].append(npu)
        self.resource_queues[npu_id] = ResourcePriorityQueues(npu_id)
        
    def add_dsp(self, dsp_id: str, bandwidth: float):
        """Add DSP resource"""
        dsp = ResourceUnit(dsp_id, ResourceType.DSP, bandwidth)
        self.resources[ResourceType.DSP].append(dsp)
        self.resource_queues[dsp_id] = ResourcePriorityQueues(dsp_id)
    
    def add_task(self, task: NNTask):
        """Add task"""
        self.tasks[task.task_id] = task
    
    def cleanup_expired_bindings(self, current_time: float):
        """Clean up expired resource bindings"""
        # Remove expired bindings
        self.active_bindings = [binding for binding in self.active_bindings 
                               if binding.binding_end > current_time]
        
        # Release resource bindings
        for queue in self.resource_queues.values():
            if queue.bound_until <= current_time:
                queue.release_binding()
    
    def find_available_resources_for_task(self, task: NNTask, current_time: float) -> Optional[Dict[ResourceType, str]]:
        """Find available resources for a task based on its runtime type"""
        if task.runtime_type == RuntimeType.DSP_RUNTIME and task.requires_resource_binding():
            # For DSP_Runtime with multiple resource types, find resources that can be bound together
            return self.find_bound_resources(task, current_time)
        else:
            # For ACPU_Runtime or single-resource DSP_Runtime, use normal scheduling
            return self.find_pipelined_resources(task, current_time)
    
    def find_bound_resources(self, task: NNTask, current_time: float) -> Optional[Dict[ResourceType, str]]:
        """Find resources that can be bound together for DSP_Runtime tasks"""
        required_resource_types = set(seg.resource_type for seg in task.segments)
        
        # Check all combinations of resources to find a set that can be bound together
        for resource_combo in self.get_resource_combinations(required_resource_types):
            can_bind_all = True
            binding_start = current_time
            binding_end = current_time
            
            # Check if all resources in this combination are available
            for resource_type, resource_id in resource_combo.items():
                queue = self.resource_queues[resource_id]
                
                # Check if resource is available and not bound to another task
                if (queue.is_bound_to_other_task(task.task_id, current_time) or
                    queue.has_higher_priority_tasks(task.priority, current_time, task.task_id)):
                    can_bind_all = False
                    break
                
                # Calculate binding times
                binding_start = max(binding_start, queue.available_time)
            
            if can_bind_all:
                # Calculate total binding duration
                for seg in task.segments:
                    if seg.resource_type in resource_combo:
                        resource_id = resource_combo[seg.resource_type]
                        resource = next(r for r in self.resources[seg.resource_type] 
                                      if r.unit_id == resource_id)
                        seg_end = binding_start + seg.start_time + seg.get_duration(resource.bandwidth)
                        binding_end = max(binding_end, seg_end)
                
                # Bind all resources
                bound_resource_ids = set(resource_combo.values())
                for resource_id in bound_resource_ids:
                    self.resource_queues[resource_id].bind_resource(task.task_id, binding_end)
                
                # Record binding
                binding = ResourceBinding(
                    task_id=task.task_id,
                    bound_resources=bound_resource_ids,
                    binding_start=binding_start,
                    binding_end=binding_end
                )
                self.active_bindings.append(binding)
                
                return resource_combo
        
        return None
    
    def find_pipelined_resources(self, task: NNTask, current_time: float) -> Optional[Dict[ResourceType, str]]:
        """Find resources for ACPU_Runtime tasks (normal pipelined scheduling)"""
        assigned_resources = {}
        earliest_start = current_time
        
        # Check resource availability for all segments
        for seg in task.segments:
            best_resource = None
            best_start_time = float('inf')
            
            # Find best resource for this segment
            for resource in self.resources[seg.resource_type]:
                queue = self.resource_queues[resource.unit_id]
                
                # Check if higher priority tasks are waiting or resource is bound
                if (queue.has_higher_priority_tasks(task.priority, current_time, task.task_id) or
                    queue.is_bound_to_other_task(task.task_id, current_time)):
                    continue
                
                # Calculate when this resource could start this task
                resource_start = max(queue.available_time, earliest_start + seg.start_time)
                
                if resource_start < best_start_time:
                    best_start_time = resource_start
                    best_resource = resource
            
            if best_resource:
                assigned_resources[seg.resource_type] = best_resource.unit_id
                earliest_start = max(earliest_start, best_start_time - seg.start_time)
            else:
                return None
        
        return assigned_resources
    
    def get_resource_combinations(self, required_types: Set[ResourceType]) -> List[Dict[ResourceType, str]]:
        """Get all possible combinations of resources for the required types"""
        combinations = []
        
        def generate_combinations(types_list, current_combo, index):
            if index == len(types_list):
                combinations.append(current_combo.copy())
                return
            
            resource_type = types_list[index]
            for resource in self.resources[resource_type]:
                current_combo[resource_type] = resource.unit_id
                generate_combinations(types_list, current_combo, index + 1)
                del current_combo[resource_type]
        
        if required_types:
            generate_combinations(list(required_types), {}, 0)
        
        return combinations
    
    def priority_aware_schedule(self, time_window: float = 1000.0) -> List[TaskScheduleInfo]:
        """Priority-aware scheduling algorithm with runtime configuration support"""
        # Reset scheduling state
        for queue in self.resource_queues.values():
            queue.available_time = 0.0
            queue.release_binding()
            for p in TaskPriority:
                queue.queues[p].clear()
        
        for task in self.tasks.values():
            task.schedule_info = None
            task.last_execution_time = -float('inf')
            task.ready_time = 0
        
        self.schedule_history.clear()
        self.active_bindings.clear()
        
        # Track task execution counts for FPS calculation
        task_execution_count = defaultdict(int)
        current_time = 0.0
        
        while current_time < time_window:
            # Clean up expired bindings
            self.cleanup_expired_bindings(current_time)
            
            # Phase 1: Check which tasks are ready (dependencies satisfied)
            for task in self.tasks.values():
                # Skip if task was recently executed
                if task.last_execution_time + task.min_interval_ms > current_time:
                    continue
                
                # Check dependencies
                deps_satisfied = True
                max_dep_end_time = 0.0
                
                for dep_id in task.dependencies:
                    dep_task = self.tasks.get(dep_id)
                    if dep_task:
                        if task_execution_count[dep_id] <= task_execution_count[task.task_id]:
                            deps_satisfied = False
                            break
                        if dep_task.schedule_info:
                            max_dep_end_time = max(max_dep_end_time, dep_task.schedule_info.end_time)
                
                if deps_satisfied:
                    task.ready_time = max(current_time, max_dep_end_time)
            
            # Phase 2: Schedule ready tasks
            scheduled_any = False
            
            for task in self.tasks.values():
                if (task.ready_time <= current_time and 
                    task.last_execution_time + task.min_interval_ms <= current_time):
                    
                    # Try to find available resources based on runtime type
                    assigned_resources = self.find_available_resources_for_task(task, current_time)
                    
                    if assigned_resources:
                        # Execute the scheduling
                        actual_start = current_time
                        actual_end = actual_start
                        
                        # Update resource availability times
                        for seg in task.segments:
                            resource_id = assigned_resources[seg.resource_type]
                            resource = next(r for r in self.resources[seg.resource_type] 
                                          if r.unit_id == resource_id)
                            
                            seg_start = actual_start + seg.start_time
                            seg_duration = seg.get_duration(resource.bandwidth)
                            seg_end = seg_start + seg_duration
                            
                            # Update resource availability (but don't override binding for DSP_Runtime)
                            if task.runtime_type == RuntimeType.ACPU_RUNTIME:
                                self.resource_queues[resource_id].available_time = seg_end
                            
                            actual_end = max(actual_end, seg_end)
                        
                        # Create schedule info
                        schedule_info = TaskScheduleInfo(
                            task_id=task.task_id,
                            start_time=actual_start,
                            end_time=actual_end,
                            assigned_resources=assigned_resources,
                            actual_latency=actual_end - current_time,
                            runtime_type=task.runtime_type
                        )
                        
                        task.schedule_info = schedule_info
                        task.last_execution_time = actual_start
                        self.schedule_history.append(schedule_info)
                        task_execution_count[task.task_id] += 1
                        scheduled_any = True
            
            # Phase 3: Advance time
            if not scheduled_any:
                # Find next event time
                next_time = current_time + 1.0
                
                # Check when resources become available or bindings expire
                for queue in self.resource_queues.values():
                    if queue.available_time > current_time:
                        next_time = min(next_time, queue.available_time)
                    if queue.bound_until > current_time:
                        next_time = min(next_time, queue.bound_until)
                
                # Check when tasks can be scheduled again
                for task in self.tasks.values():
                    next_schedule_time = task.last_execution_time + task.min_interval_ms
                    if next_schedule_time > current_time:
                        next_time = min(next_time, next_schedule_time)
                
                current_time = min(next_time, time_window)
            else:
                # Small time advance to check for new opportunities
                current_time += 0.1
        
        return self.schedule_history
    
    def get_resource_utilization(self, time_window: float) -> Dict[str, float]:
        """Calculate resource utilization"""
        utilization = {}
        
        for resource_type, resources in self.resources.items():
            for resource in resources:
                busy_time = 0.0
                for schedule in self.schedule_history:
                    if resource.unit_id in schedule.assigned_resources.values():
                        # Simplified: assume resource is busy for entire task duration
                        busy_time += (schedule.end_time - schedule.start_time)
                
                utilization[resource.unit_id] = min(busy_time / time_window * 100, 100)
        
        return utilization
    
    def print_schedule_summary(self):
        """Print scheduling summary with runtime information"""
        print("=== Scheduling Summary ===")
        print(f"NPU Resources: {len(self.resources[ResourceType.NPU])}")
        print(f"DSP Resources: {len(self.resources[ResourceType.DSP])}")
        print(f"Total Tasks: {len(self.tasks)}")
        print(f"Total Scheduled Events: {len(self.schedule_history)}")
        
        # Count by priority and runtime type
        priority_counts = defaultdict(int)
        runtime_counts = defaultdict(int)
        for task in self.tasks.values():
            priority_counts[task.priority.name] += 1
            runtime_counts[task.runtime_type.value] += 1
        
        print("\nTasks by Priority:")
        for priority in TaskPriority:
            count = priority_counts[priority.name]
            print(f"  {priority.name}: {count} tasks")
        
        print("\nTasks by Runtime Type:")
        for runtime_type in RuntimeType:
            count = runtime_counts[runtime_type.value]
            print(f"  {runtime_type.value}: {count} tasks")
        
        # Count task scheduling
        task_schedule_count = defaultdict(int)
        task_latencies = defaultdict(list)
        
        for schedule in self.schedule_history:
            task_schedule_count[schedule.task_id] += 1
            task_latencies[schedule.task_id].append(schedule.actual_latency)
        
        print("\nTask Scheduling Details:")
        # Group by priority
        for priority in TaskPriority:
            priority_tasks = [t for t in self.tasks.values() if t.priority == priority]
            if priority_tasks:
                print(f"\n  {priority.name} Priority Tasks:")
                for task in priority_tasks:
                    count = task_schedule_count[task.task_id]
                    avg_latency = sum(task_latencies[task.task_id]) / len(task_latencies[task.task_id]) if task_latencies[task.task_id] else 0
                    achieved_fps = count / (self.schedule_history[-1].end_time / 1000) if self.schedule_history else 0
                    
                    print(f"    {task.task_id} ({task.name}) [{task.runtime_type.value}]:")
                    print(f"      Scheduled Count: {count}")
                    print(f"      Average Latency: {avg_latency:.1f}ms (Required: {task.latency_requirement}ms)")
                    print(f"      Achieved FPS: {achieved_fps:.1f} (Required: {task.fps_requirement})")
        
        # Resource utilization
        if self.schedule_history:
            time_window = self.schedule_history[-1].end_time
            utilization = self.get_resource_utilization(time_window)
            print("\nResource Utilization:")
            for resource_id, util in utilization.items():
                print(f"  {resource_id}: {util:.1f}%")
        
        # Resource binding statistics
        binding_count = defaultdict(int)
        for schedule in self.schedule_history:
            if schedule.runtime_type == RuntimeType.DSP_RUNTIME:
                binding_count[schedule.task_id] += 1
        
        if binding_count:
            print("\nDSP_Runtime Binding Statistics:")
            for task_id, count in binding_count.items():
                task = self.tasks[task_id]
                print(f"  {task_id} ({task.name}): {count} bound executions")