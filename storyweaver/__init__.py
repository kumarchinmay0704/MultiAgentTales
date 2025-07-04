from .workers import (
    ProtagonistAgent,
    AntagonistAgent,
    SceneBuilder,
    DialogueGenerator,
    generate_story
)
from .orchestrator import Orchestrator

__all__ = [
    'ProtagonistAgent',
    'AntagonistAgent',
    'SceneBuilder',
    'DialogueGenerator',
    'Orchestrator',
    'generate_story'
]

__version__ = '1.0.0' 