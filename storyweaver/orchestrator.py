from typing import Dict, List
from .workers import (
    ProtagonistAgent,
    AntagonistAgent,
    SceneBuilder,
    DialogueGenerator
)

class Orchestrator:
    def __init__(self, context: Dict):
        self.context = context
        self.agents = {
            'protagonist': ProtagonistAgent(context),
            'antagonist': AntagonistAgent(context),
            'scene': SceneBuilder(context),
            'dialogue': DialogueGenerator(context)
        }
        self.story_state = {
            'current_scene': None,
            'last_action': None,
            'story_progress': 0,
            'mood': 'neutral',
            'tension_level': 0,
            'plot_points': []
        }
    
    def update_story_state(self, new_element: str):
        """Update story state based on new content"""
        self.story_state['last_action'] = new_element
        self.story_state['story_progress'] += 1
        
        # Update mood and tension based on content
        if any(word in new_element.lower() for word in ['danger', 'threat', 'attack']):
            self.story_state['tension_level'] += 1
            self.story_state['mood'] = 'tense'
        elif any(word in new_element.lower() for word in ['peace', 'calm', 'safe']):
            self.story_state['tension_level'] = max(0, self.story_state['tension_level'] - 1)
            self.story_state['mood'] = 'peaceful'
    
    def generate_next_story_element(self, user_input: str) -> str:
        # Determine which agent should respond based on story state
        if self.story_state['current_scene'] is None:
            response = self.agents['scene'].build(user_input, self.story_state)
            self.story_state['current_scene'] = response
        elif self.story_state['story_progress'] % 2 == 0:
            response = self.agents['protagonist'].respond(user_input, self.story_state)
        else:
            response = self.agents['antagonist'].respond(user_input, self.story_state)
        
        # Add dialogue
        dialogue = self.agents['dialogue'].generate(user_input, self.story_state)
        
        # Update story state
        self.update_story_state(response)
        
        return f"{response}\n{dialogue}"
