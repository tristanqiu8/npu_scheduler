from typing import Optional, Tuple
from collections import deque
from enums import TaskPriority
from task import NNTask

class ResourcePriorityQueues:
    """Priority queues for a single resource"""
    
    def __init__(self, resource_id: str):
        self.resource_id = resource_id
        # Create a queue for each priority level
        self.queues = {
            priority: deque() for priority in TaskPriority
        }
        self.available_time = 0.0  # When resource becomes available
    
    def add_task(self, task: NNTask, ready_time: float):
        """Add task to appropriate priority queue"""
        self.queues[task.priority].append((task, ready_time))
    
    def get_next_task(self, current_time: float) -> Optional[Tuple[NNTask, float]]:
        """Get next task to execute based on priority"""
        # Check queues from highest to lowest priority
        for priority in TaskPriority:
            queue = self.queues[priority]
            # Remove tasks that are not ready yet
            ready_tasks = []
            while queue:
                task, ready_time = queue.popleft()
                if ready_time <= current_time and task.last_execution_time + task.min_interval_ms <= current_time:
                    ready_tasks.append((task, ready_time))
                else:
                    # Put back if not ready
                    queue.append((task, ready_time))
                    break
            
            # If we found ready tasks at this priority level, return the first one
            if ready_tasks:
                # Put remaining ready tasks back
                for t in ready_tasks[1:]:
                    queue.append(t)
                return ready_tasks[0]
        
        return None
    
    def has_higher_priority_tasks(self, priority: TaskPriority, current_time: float) -> bool:
        """Check if there are higher priority tasks waiting"""
        for p in TaskPriority:
            if p < priority:  # Higher priority
                # Check if any task in this queue is ready
                for task, ready_time in self.queues[p]:
                    if (ready_time <= current_time and 
                        task.last_execution_time + task.min_interval_ms <= current_time):
                        return True
        return False
