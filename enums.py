from enum import Enum

class ResourceType(Enum):
    """Resource type enumeration"""
    DSP = "DSP"
    NPU = "NPU"

class TaskPriority(Enum):
    """Task priority levels (from high to low)"""
    CRITICAL = 0     # Highest priority - safety critical tasks
    HIGH = 1         # High priority - real-time tasks
    NORMAL = 2       # Normal priority - regular tasks
    LOW = 3          # Low priority - background tasks
    
    def __lt__(self, other):
        """Enable priority comparison"""
        return self.value < other.value
