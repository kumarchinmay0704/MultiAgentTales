import logging
from pathlib import Path
import os
import google.generativeai as genai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys
GOOGLE_API_KEY = "AIzaSyC16i9PUmcvu8LnACcBrxJ4tmtrNmOq0qM"

# Configure APIs
genai.configure(api_key=GOOGLE_API_KEY)

class ModelManager:
    """Manages model access through multiple AI providers"""
    _models = {}
    _last_request_time = {}  # Track last request time for each model
    _request_counts = {}     # Track request counts for rate limiting
    
    # Model configurations for different roles
    MODEL_CONFIGS = {
        'narrator': {
            'provider': 'gemini',
            'model': 'gemini-2.0-flash-lite',  # Highest RPM (30) and RPD (1,500)
            'temperature': 0.7,
            'top_p': 0.95,
            'top_k': 40,
            'rpm_limit': 30,
            'rpd_limit': 1500
        },
        'protagonist': {
            'provider': 'gemini',
            'model': 'gemini-2.0-flash',  # Good balance (15 RPM, 1,500 RPD)
            'temperature': 0.8,
            'top_p': 0.95,
            'top_k': 40,
            'rpm_limit': 15,
            'rpd_limit': 1500
        },
        'antagonist': {
            'provider': 'gemini',
            'model': 'gemini-1.5-flash',  # Separate model (15 RPM, 500 RPD)
            'temperature': 0.8,
            'top_p': 0.95,
            'top_k': 40,
            'rpm_limit': 15,
            'rpd_limit': 500
        },
        'scene': {
            'provider': 'gemini',
            'model': 'gemini-2.0-flash-lite',  # Highest RPM (30) and RPD (1,500)
            'temperature': 0.7,
            'top_p': 0.95,
            'top_k': 40,
            'rpm_limit': 30,
            'rpd_limit': 1500
        },
        'dialogue': {
            'provider': 'gemini',
            'model': 'gemini-2.0-flash',  # Good balance (15 RPM, 1,500 RPD)
            'temperature': 0.9,
            'top_p': 0.95,
            'top_k': 40,
            'rpm_limit': 15,
            'rpd_limit': 1500
        }
    }
    
    @classmethod
    def _check_rate_limits(cls, role: str) -> bool:
        """Check if we're within rate limits for a given role"""
        config = cls.MODEL_CONFIGS[role]
        model = config['model']
        current_time = time.time()
        
        # Initialize tracking if needed
        if model not in cls._last_request_time:
            cls._last_request_time[model] = []
            cls._request_counts[model] = 0
        
        # Clean up old requests (older than 1 minute)
        cls._last_request_time[model] = [t for t in cls._last_request_time[model] 
                                       if current_time - t < 60]
        
        # Check RPM limit
        if len(cls._last_request_time[model]) >= config['rpm_limit']:
            # Calculate wait time
            wait_time = 60 - (current_time - cls._last_request_time[model][0])
            if wait_time > 0:
                logger.info(f"Rate limit reached for {model}. Waiting {wait_time:.2f} seconds.")
                time.sleep(wait_time)
            return True
        
        # Check RPD limit
        if cls._request_counts[model] >= config['rpd_limit']:
            logger.error(f"Daily rate limit reached for {model}")
            return False
        
        return True
    
    @classmethod
    def _update_rate_limits(cls, role: str):
        """Update rate limit tracking after a request"""
        config = cls.MODEL_CONFIGS[role]
        model = config['model']
        current_time = time.time()
        
        cls._last_request_time[model].append(current_time)
        cls._request_counts[model] += 1
    
    @classmethod
    def get_model(cls, role: str):
        """Get or create a model for a specific role"""
        if role not in cls._models:
            if role not in cls.MODEL_CONFIGS:
                raise ValueError(f"Unknown role: {role}")
            
            config = cls.MODEL_CONFIGS[role]
            try:
                model = genai.GenerativeModel(
                    model_name=config['model'],
                    generation_config=genai.types.GenerationConfig(
                        temperature=config['temperature'],
                        top_p=config['top_p'],
                        top_k=config['top_k'],
                        max_output_tokens=512
                    )
                )
                cls._models[role] = model
            except Exception as e:
                logger.error(f"Error initializing model for role {role}: {str(e)}")
                raise
            
        return cls._models[role]

class BaseAgent:
    def __init__(self, context: dict, role: str):
        self.context = context
        self.role = role
        self.model = ModelManager.get_model(role)
        self.provider = ModelManager.MODEL_CONFIGS[role]['provider']
        
    def generate(self, prompt: str) -> str:
        """Generate text using the model"""
        max_retries = 3
        retry_delay = 5  # Increased delay between retries
        
        for attempt in range(max_retries):
            try:
                # Add delay between requests
                time.sleep(2)  # Minimum delay between requests
                
                # Check rate limits
                if not ModelManager._check_rate_limits(self.role):
                    wait_time = 60  # Wait a full minute if rate limit is hit
                    logger.info(f"Rate limit reached. Waiting {wait_time} seconds.")
                    time.sleep(wait_time)
                
                # Configure generation parameters
                generation_config = {
                    "temperature": ModelManager.MODEL_CONFIGS[self.role]['temperature'],
                    "top_p": ModelManager.MODEL_CONFIGS[self.role]['top_p'],
                    "top_k": ModelManager.MODEL_CONFIGS[self.role]['top_k'],
                    "max_output_tokens": 512,
                }
                
                # Generate content
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                # Update rate limit tracking
                ModelManager._update_rate_limits(self.role)
                
                if response and hasattr(response, 'text'):
                    return response.text.strip()
                
                logger.error(f"Invalid response from model for {self.role}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                return f"Error: Invalid response from model"
                    
            except Exception as e:
                logger.error(f"Error generating text for {self.role} (attempt {attempt + 1}): {str(e)}")
                if "429" in str(e) and attempt < max_retries - 1:
                    # Rate limit hit, wait and retry with exponential backoff
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Rate limit error. Waiting {wait_time} seconds before retry.")
                    time.sleep(wait_time)
                    continue
                return f"Error generating text for {self.role}: {str(e)}"
        
        return f"Failed to generate text after {max_retries} attempts"

class NarratorAgent(BaseAgent):
    def __init__(self, context: dict):
        super().__init__(context, 'narrator')
        self.prompt_template = """Write a narrative passage for a {genre} story.

Setting: {setting}
Characters: {protagonist} (protagonist) and {antagonist} (antagonist)
Context: {context}
Story state: {story_state}

Write a vivid and engaging narrative that sets the scene and advances the story."""

class ProtagonistAgent(BaseAgent):
    def __init__(self, context: dict):
        super().__init__(context, 'protagonist')
        self.prompt_template = """As {protagonist}, describe your next action in this story.

Context: {context}
Story state: {story_state}

Write from {protagonist}'s perspective, describing their thoughts, feelings, and actions."""

class AntagonistAgent(BaseAgent):
    def __init__(self, context: dict):
        super().__init__(context, 'antagonist')
        self.prompt_template = """As {antagonist}, describe your next action in this story.

Context: {context}
Story state: {story_state}

Write from {antagonist}'s perspective, describing their thoughts, feelings, and actions."""

class SceneBuilder(BaseAgent):
    def __init__(self, context: dict):
        super().__init__(context, 'scene')
        self.prompt_template = """Write a scene for a {genre} story.

Setting: {setting}
Characters: {protagonist} and {antagonist}
Context: {context}
Story state: {story_state}

Create a vivid and atmospheric scene that advances the story."""

class DialogueGenerator(BaseAgent):
    def __init__(self, context: dict):
        super().__init__(context, 'dialogue')
        self.prompt_template = """What does {speaker} say in this moment?

Context: {context}
Story state: {story_state}

Write natural and engaging dialogue that reveals character and advances the plot."""

class StoryOrchestrator:
    def __init__(self, context: dict):
        self.context = context
        self.agents = {
            'narrator': NarratorAgent(context),
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
    
    def generate_next_story_element(self, user_input: str) -> str:
        try:
            # Determine which agent should respond based on story state
            if self.story_state['current_scene'] is None:
                # Generate initial scene
                prompt = self.agents['scene'].prompt_template.format(
                    genre=self.context['genre'],
                    setting=self.context['setting'],
                    protagonist=self.context['characters']['protagonist'],
                    antagonist=self.context['characters']['antagonist'],
                    context=user_input,
                    story_state=self.story_state
                )
                response = self.agents['scene'].generate(prompt)
                self.story_state['current_scene'] = response
                
                # Add narrator's perspective
                narrator_prompt = self.agents['narrator'].prompt_template.format(
                    genre=self.context['genre'],
                    setting=self.context['setting'],
                    protagonist=self.context['characters']['protagonist'],
                    antagonist=self.context['characters']['antagonist'],
                    context=user_input,
                    story_state=self.story_state
                )
                narrator_response = self.agents['narrator'].generate(narrator_prompt)
                
                return f"{narrator_response}\n\n{response}"
            else:
                # Alternate between protagonist and antagonist
                if self.story_state['story_progress'] % 2 == 0:
                    agent = self.agents['protagonist']
                    character = self.context['characters']['protagonist']
                else:
                    agent = self.agents['antagonist']
                    character = self.context['characters']['antagonist']
                
                # Generate character action
                action_prompt = agent.prompt_template.format(
                    protagonist=self.context['characters']['protagonist'],
                    antagonist=self.context['characters']['antagonist'],
                    context=user_input,
                    story_state=self.story_state
                )
                action_response = agent.generate(action_prompt)
                
                # Generate dialogue
                dialogue_prompt = self.agents['dialogue'].prompt_template.format(
                    speaker=character,
                    context=user_input,
                    story_state=self.story_state
                )
                dialogue_response = self.agents['dialogue'].generate(dialogue_prompt)
                
                # Update story state
                self.story_state['last_action'] = action_response
                self.story_state['story_progress'] += 1
                
                return f"{action_response}\n\n{dialogue_response}"
        except Exception as e:
            logger.error(f"Error generating story element: {str(e)}")
            return "The story continues with an unexpected turn of events."

def get_ending_prompt(ending_type: str, story_context: dict) -> str:
    """Generate an appropriate prompt based on the ending type"""
    ending_prompts = {
        'suspense': """Create a suspenseful ending that leaves readers on the edge of their seats. 
Build tension and create a sense of anticipation for what might happen next.""",
        
        'dramatic': """Write a dramatic and emotionally powerful ending. 
Include intense emotions, high stakes, and a climactic resolution.""",
        
        'happy': """Create a satisfying happy ending where the main characters achieve their goals 
and find resolution to their conflicts.""",
        
        'tragic': """Write a tragic ending that evokes strong emotions. 
The characters should face significant loss or failure, but in a meaningful way.""",
        
        'twist': """Create an unexpected plot twist ending that surprises readers 
while still making sense within the story's context.""",
        
        'cliffhanger': """Write a cliffhanger ending that leaves a major question unanswered 
and creates anticipation for what might happen next.""",
        
        'mysterious': """Create a mysterious ending that leaves some elements unexplained 
and allows readers to interpret the conclusion in different ways.""",
        
        'bittersweet': """Write a bittersweet ending that combines both positive and negative elements, 
showing that victory comes with a cost or that loss brings some form of growth."""
    }
    
    return ending_prompts.get(ending_type, "Write a satisfying conclusion to the story.")

def generate_story(initial_prompt: str, num_elements: int = 5, ending_type: str = None) -> list:
    """Generate a complete story with multiple elements
    
    Args:
        initial_prompt (str): The starting point for the story
        num_elements (int): Number of story elements to generate (default: 5)
        ending_type (str): Type of ending to generate (e.g., 'suspense', 'dramatic', etc.)
    """
    try:
        # Extract characters from the prompt
        characters = {
            'protagonist': 'Character 1',
            'antagonist': 'Character 2'
        }
        
        # Try to extract character names from the prompt
        import re
        character_match = re.search(r'featuring\s+([^and]+)\s+and\s+([^in\.]+)', initial_prompt, re.IGNORECASE)
        if character_match:
            characters['protagonist'] = character_match.group(1).strip()
            characters['antagonist'] = character_match.group(2).strip()
        
        context = {
            'characters': characters,
            'genre': 'story',  # Default genre
            'setting': 'an interesting location'  # Default setting
        }
        
        # Try to extract genre and setting
        genre_match = re.search(r'Begin a (\w+) story', initial_prompt, re.IGNORECASE)
        if genre_match:
            context['genre'] = genre_match.group(1).strip()
        
        setting_match = re.search(r'in\s+([^\.]+)', initial_prompt, re.IGNORECASE)
        if setting_match:
            context['setting'] = setting_match.group(1).strip()
        
        orchestrator = StoryOrchestrator(context)
        story_elements = []
        
        # Generate initial scene
        intro_prompt = f"""Write the opening scene of a {context['genre']} story.
Setting: {context['setting']}
Characters: {context['characters']['protagonist']} (protagonist) and {context['characters']['antagonist']} (antagonist)
Focus on setting the scene and introducing the characters. Write in third person narrative style."""
        
        first_element = orchestrator.generate_next_story_element(intro_prompt)
        if first_element and isinstance(first_element, str):
            story_elements.append(first_element)
        else:
            logger.error("Failed to generate initial scene")
            return ["Failed to generate the story. Please try again."]
        
        # Generate subsequent elements
        for i in range(num_elements - 1):
            try:
                if i % 2 == 0:
                    # Protagonist's turn
                    continuation_prompt = f"""Continue the story from {context['characters']['protagonist']}'s perspective.
Previous events: {story_elements[-1]}
Focus on {context['characters']['protagonist']}'s thoughts, feelings, and actions.
Write in first person from {context['characters']['protagonist']}'s point of view."""
                else:
                    # Antagonist's turn
                    continuation_prompt = f"""Continue the story from {context['characters']['antagonist']}'s perspective.
Previous events: {story_elements[-1]}
Focus on {context['characters']['antagonist']}'s thoughts, feelings, and actions.
Write in first person from {context['characters']['antagonist']}'s point of view."""
                
                next_element = orchestrator.generate_next_story_element(continuation_prompt)
                if next_element and isinstance(next_element, str):
                    story_elements.append(next_element)
                else:
                    logger.error(f"Invalid story element generated: {next_element}")
                    story_elements.append("The story continues with an unexpected turn of events.")
            except Exception as e:
                logger.error(f"Error generating story element {i+1}: {str(e)}")
                story_elements.append("The story takes an unexpected turn.")
        
        # Add ending if specified
        if ending_type:
            ending_prompt = f"""Write the final scene of this {context['genre']} story.
Previous events: {story_elements[-1]}
Characters: {context['characters']['protagonist']} and {context['characters']['antagonist']}
Setting: {context['setting']}

{get_ending_prompt(ending_type, context)}

Write in third person narrative style. Make this a satisfying conclusion that wraps up the story."""

            try:
                final_element = orchestrator.generate_next_story_element(ending_prompt)
                if final_element and isinstance(final_element, str):
                    # Add a conclusion header to the final element
                    story_elements.append(f"Conclusion\n\n{final_element}")
                else:
                    logger.error("Failed to generate ending")
                    story_elements.append("Conclusion\n\nThe story concludes with an unexpected ending.")
            except Exception as e:
                logger.error(f"Error generating ending: {str(e)}")
                story_elements.append("Conclusion\n\nThe story concludes with an unexpected ending.")
        
        return story_elements
    except Exception as e:
        logger.error(f"Error generating story: {str(e)}")
        return ["The story generation encountered an unexpected error. Please try again."]

if __name__ == "__main__":
    # Example prompts for different genres
    example_prompts = [
        "Begin a horror story featuring Alex and Maya in an abandoned hospital",
        "Begin a fantasy story featuring Elara and Thorne in a magical forest",
        "Begin a mystery story featuring Detective Smith and Sarah in a small town",
        "Begin a sci-fi story featuring Captain Nova and Dr. Chen on a space station"
    ]
    
    # Print example prompts
    print("\n=== Example Story Prompts ===\n")
    for i, prompt in enumerate(example_prompts, 1):
        print(f"{i}. {prompt}")
    
    print("\n=== Generated Story ===\n")
    # Use the first example prompt with a custom ending
    user_ending = "Alex and Maya finally escape the hospital, forever changed by their experience."
    story = generate_story(example_prompts[0], num_elements=5, ending_type='happy')
    
    for i, element in enumerate(story, 1):
        print(f"\n--- Part {i} ---")
        print(element)
        print("-" * 50) 