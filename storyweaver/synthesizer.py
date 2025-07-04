class Synthesizer:
    def __init__(self, context=None):
        self.context = context or {
            'characters': {
                'protagonist': 'Alex',
                'antagonist': 'Morgan',
                'support': 'Jamie'
            }
        }

    def synthesize(self, *args):
        # Filter out None values and duplicates while preserving order
        seen = set()
        valid_segments = []
        for arg in args:
            if arg is not None and arg not in seen:
                valid_segments.append(arg)
                seen.add(arg)
        
        # Get character names from context
        protagonist = self.context['characters']['protagonist']
        antagonist = self.context['characters']['antagonist']
        support = self.context['characters']['support']
        
        # Group segments by type
        scene_segments = [s for s in valid_segments if "scene unfolds" in s.lower() or "scene is set" in s.lower()]
        character_segments = [s for s in valid_segments if any(name in s for name in [protagonist, antagonist, support])]
        dialogue_segments = [s for s in valid_segments if '"' in s]
        mood_segments = [s for s in valid_segments if any(word in s.lower() for word in ["mood", "tension", "fear", "wonder", "mystery"])]
        world_segments = [s for s in valid_segments if s not in scene_segments + character_segments + dialogue_segments + mood_segments]
        
        # Combine segments in a narrative flow
        story = []
        
        # Start with scene setting (only one)
        if scene_segments:
            story.append(scene_segments[0])
        
        # Add character actions (ensure each character appears only once)
        seen_characters = set()
        for segment in character_segments:
            for char in [protagonist, antagonist, support]:
                if char in segment and char not in seen_characters:
                    story.append(segment)
                    seen_characters.add(char)
                    break
        
        # Add world details (max 1)
        if world_segments:
            story.append(world_segments[0])
        
        # Add dialogue (max 1)
        if dialogue_segments:
            story.append(dialogue_segments[0])
        
        # End with mood (max 1)
        if mood_segments:
            story.append(mood_segments[0])
        
        return "\n".join(story)

def generate_visual(story_segment):
    # In a real system, call an AI art API here
    return "/static/placeholder.jpg"
