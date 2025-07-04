import random
import logging
from typing import Dict, List, Optional
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
import gc
from huggingface_hub import snapshot_download
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_gpu_memory():
    """Clear GPU memory cache"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as e:
        logger.error(f"Error clearing GPU memory: {str(e)}")

def get_gpu_memory_usage():
    """Get current GPU memory usage"""
    try:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        return 0
    except Exception as e:
        logger.error(f"Error getting GPU memory usage: {str(e)}")
        return 0

class ModelManager:
    """Manages model loading and caching"""
    _models = {}
    _tokenizers = {}
    
    # Map model names to their roles - using specialized models for each role
    MODEL_MAPPING = {
        'narrator': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',    # Good for storytelling and narrative
        'protagonist': 'PygmalionAI/pygmalion-2-7b',         # Good for hero's perspective
        'antagonist': 'PygmalionAI/pygmalion-2-7b',          # Good for villain's perspective
        'scene': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',       # Good for scene descriptions
        'dialogue': 'PygmalionAI/pygmalion-2-7b',            # Good for character interaction
        'emotion': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'      # Good for emotional tone
    }
    
    @classmethod
    def download_models(cls):
        """Download all required models to local directory"""
        base_path = Path("models")
        base_path.mkdir(exist_ok=True)
        
        for role, model_name in cls.MODEL_MAPPING.items():
            model_path = base_path / role
            model_path.mkdir(exist_ok=True)
            
            if not (model_path / "config.json").exists():
                logger.info(f"Downloading {model_name} for {role}...")
                try:
                    # Download model and tokenizer with optimizations
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,     # Use half precision
                        low_cpu_mem_usage=True,        # Reduce CPU memory usage
                        device_map="auto",             # Automatic device mapping
                        max_memory={0: "1GB"},         # Limit GPU memory usage
                        cache_dir=base_path,           # Force download to project directory
                        local_files_only=False         # Allow downloads
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        padding_side="left",           # Better for batch processing
                        truncation_side="left",        # Better for long sequences
                        cache_dir=base_path,           # Force download to project directory
                        local_files_only=False         # Allow downloads
                    )
                    
                    # Save to local directory
                    model.save_pretrained(model_path)
                    tokenizer.save_pretrained(model_path)
                    logger.info(f"Successfully downloaded {model_name} for {role}")
                except Exception as e:
                    logger.error(f"Error downloading {model_name} for {role}: {str(e)}")
                    raise
    
    @classmethod
    def get_model(cls, model_name: str):
        if model_name not in cls._models:
            logger.info(f"Loading model: {model_name}")
            
            # Get the role for this model
            role = None
            for r, m in cls.MODEL_MAPPING.items():
                if m == model_name:
                    role = r
                    break
            
            if not role:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Path to the local model
            model_dir = Path("models") / role
            
            if not model_dir.exists():
                logger.info(f"Model directory not found: {model_dir}. Downloading models...")
                cls.download_models()
            
            try:
                # Force CUDA device
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available. Please check your GPU installation.")
                
                device = "cuda"
                gpu_memory = get_gpu_memory_usage()
                logger.info(f"Current GPU memory usage: {gpu_memory:.2f} MB")
                
                # Clear GPU memory if usage is high
                if gpu_memory > 500:  # If more than 500MB is used
                    logger.info("Clearing GPU memory cache...")
                    clear_gpu_memory()
                
                logger.info(f"Using device: {device}")
                
                # Load tokenizer and model from local directory with optimizations
                cls._tokenizers[model_name] = AutoTokenizer.from_pretrained(
                    model_dir,
                    padding_side="left",
                    truncation_side="left"
                )
                cls._models[model_name] = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float16,     # Use half precision
                    device_map="auto",             # Automatic device mapping
                    low_cpu_mem_usage=True,        # Reduce CPU memory usage
                    max_memory={0: "1GB"}          # Limit GPU memory usage
                )
                
                logger.info(f"Successfully loaded model from {model_dir}")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                # Clear GPU memory on error
                clear_gpu_memory()
                raise
                
        return cls._models[model_name], cls._tokenizers[model_name]

    @classmethod
    def clear_all_models(cls):
        """Clear all loaded models from memory"""
        cls._models.clear()
        cls._tokenizers.clear()
        clear_gpu_memory()
        logger.info("Cleared all models from memory")

class BaseAgent:
    def __init__(self, context: Dict, model_name: str):
        self.context = context
        self.model_name = model_name
        self.model, self.tokenizer = ModelManager.get_model(model_name)
        
        # Force CUDA device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your GPU installation.")
        
        self.device = "cuda"
        
        # Create the pipeline with optimized settings
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=32,          # Reduced for faster generation
            do_sample=True,             # Enable sampling
            temperature=0.7,            # Balanced creativity
            top_p=0.95,                 # Nucleus sampling
            repetition_penalty=1.15,    # Prevent repetition
            batch_size=1,               # Single batch for stability
            torch_dtype=torch.float16,  # Use half precision
            device_map="auto",          # Automatic device mapping
            max_memory={0: "1GB"}       # Limit GPU memory usage
        )
        
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        self.memory = ConversationBufferMemory()
        
        if not isinstance(context, dict):
            logger.error(f"Invalid context type: {type(context)}")
            raise ValueError("Context must be a dictionary")
        if 'characters' not in context:
            logger.error("No characters found in context")
            raise ValueError("Context must contain 'characters' key")

    def __del__(self):
        """Cleanup when agent is destroyed"""
        try:
            if hasattr(self, 'pipe'):
                del self.pipe
            if hasattr(self, 'llm'):
                del self.llm
            if hasattr(self, 'device') and self.device == "cuda":
                clear_gpu_memory()
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")

class NarratorAgent(BaseAgent):
    def __init__(self, context: Dict): 
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        super().__init__(context, model_name)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "story_state", "genre", "setting"],
            template="""Write a narrative passage for a {genre} story.

Setting: {setting}
Context: {context}
Story state: {story_state}

Narrative:"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

class ProtagonistAgent(BaseAgent):
    def __init__(self, context: Dict): 
        model_name = "PygmalionAI/pygmalion-2-7b"
        super().__init__(context, model_name)
        self.prompt_template = PromptTemplate(
            input_variables=["protagonist", "context", "story_state"],
            template="""As {protagonist}, describe your next action in this story.

Context: {context}
Story state: {story_state}

Your action:"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

class AntagonistAgent(BaseAgent):
    def __init__(self, context: Dict): 
        model_name = "PygmalionAI/pygmalion-2-7b"
        super().__init__(context, model_name)
        self.prompt_template = PromptTemplate(
            input_variables=["antagonist", "context", "story_state"],
            template="""As {antagonist}, describe your next action in this story.

Context: {context}
Story state: {story_state}

Your action:"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

class SceneBuilder(BaseAgent):
    def __init__(self, context: Dict): 
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        super().__init__(context, model_name)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "story_state", "genre", "setting", "characters"],
            template="""Write a scene for a {genre} story.

Setting: {setting}
Characters: {characters}
Context: {context}
Story state: {story_state}

Scene:"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

class DialogueGenerator(BaseAgent):
    def __init__(self, context: Dict): 
        model_name = "PygmalionAI/pygmalion-2-7b"
        super().__init__(context, model_name)
        self.prompt_template = PromptTemplate(
            input_variables=["speaker", "context", "story_state"],
            template="""What does {speaker} say in this moment?

Context: {context}
Story state: {story_state}

{speaker}:"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

class EmotionGenerator(BaseAgent):
    def __init__(self, context: Dict): 
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        super().__init__(context, model_name)
        self.prompt_template = PromptTemplate(
            input_variables=["character", "emotion", "context", "story_state"],
            template="""How does {character} feel {emotion} in this moment?

Context: {context}
Story state: {story_state}

Emotional response:"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

class StoryOrchestrator:
    def __init__(self, context: Dict):
        self.context = context
        self.agents = {}
        self.story_state = {
            'current_scene': None,
            'last_action': None,
            'story_progress': 0,
            'mood': 'neutral',
            'tension_level': 0,
            'plot_points': []
        }
        
        # Initialize agents lazily to save memory
        self._agent_types = {
            'protagonist': ProtagonistAgent,
            'antagonist': AntagonistAgent,
            'scene': SceneBuilder,
            'dialogue': DialogueGenerator,
            'emotion': EmotionGenerator
        }
    
    def _get_agent(self, agent_type: str):
        """Lazy loading of agents"""
        if agent_type not in self.agents:
            self.agents[agent_type] = self._agent_types[agent_type](self.context)
        return self.agents[agent_type]
    
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
        try:
            # Determine which agent should respond based on story state
            if self.story_state['current_scene'] is None:
                response = self._get_agent('scene').build(user_input, self.story_state)
                self.story_state['current_scene'] = response
            elif self.story_state['story_progress'] % 2 == 0:
                response = self._get_agent('protagonist').respond(user_input, self.story_state)
            else:
                response = self._get_agent('antagonist').respond(user_input, self.story_state)
            
            # Add dialogue
            dialogue = self._get_agent('dialogue').generate(user_input, self.story_state)
            
            # Update story state
            self.update_story_state(response)
            
            return f"{response}\n{dialogue}"
        except Exception as e:
            logger.error(f"Error generating story element: {str(e)}")
            # Clear GPU memory on error
            if torch.cuda.is_available():
                clear_gpu_memory()
            return "The story continues with an unexpected turn of events."
    
    def __del__(self):
        """Cleanup when orchestrator is destroyed"""
        self.agents.clear()
        if torch.cuda.is_available():
            clear_gpu_memory()

def generate_story(initial_prompt: str, num_elements: int = 5) -> List[str]:
    """Generate a complete story with multiple elements
    
    Args:
        initial_prompt (str): The starting point for the story. Should be in the format:
            "Begin a [genre] story featuring [character1] and [character2]. [optional setting/context]"
            Example: "Begin a horror story featuring Alex and Maya in an abandoned hospital"
        num_elements (int): Number of story elements to generate (default: 5)
    
    Returns:
        List[str]: List of story elements
    """
    # Validate and format the initial prompt
    if not initial_prompt.strip():
        initial_prompt = "Begin a story featuring two characters in an interesting setting."
    
    # Extract characters from the prompt
    characters = {
        'protagonist': 'Character 1',
        'antagonist': 'Character 2',
        'support': 'Character 3'
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
    
    try:
        orchestrator = StoryOrchestrator(context)
        story_elements = []
        
        # Generate initial scene
        story_elements.append(orchestrator.generate_next_story_element(initial_prompt))
        
        # Generate subsequent elements
        for i in range(num_elements - 1):
            # Create a continuation prompt that references the previous element
            continuation_prompt = f"Continue the story from: {story_elements[-1]}"
            next_element = orchestrator.generate_next_story_element(continuation_prompt)
            story_elements.append(next_element)
            
            # Clear GPU memory periodically
            if i % 2 == 0 and torch.cuda.is_available():
                clear_gpu_memory()
        
        return story_elements
    except Exception as e:
        logger.error(f"Error generating story: {str(e)}")
        if torch.cuda.is_available():
            clear_gpu_memory()
        return ["The story generation encountered an unexpected error."]
    finally:
        # Cleanup
        if 'orchestrator' in locals():
            del orchestrator
        ModelManager.clear_all_models()

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
    # Use the first example prompt
    story = generate_story(example_prompts[0], num_elements=5)
    
    for i, element in enumerate(story, 1):
        print(f"\n--- Part {i} ---")
        print(element)
        print("-" * 50)
