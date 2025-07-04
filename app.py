from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify, send_file
from storyweaver.orchestrator import Orchestrator
from storyweaver.visuals import generate_visual
from storyweaver import generate_story
from storyweaver.workers import StoryOrchestrator
import os
from datetime import timedelta
import logging
from setup_models import generate_story, StoryOrchestrator
import time
import threading
from collections import deque
from fpdf import FPDF
import base64
import tempfile
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.permanent_session_lifetime = timedelta(minutes=60)

# Global variable to store the orchestrator
orchestrator = None

# Ensure the Hugging Face API key is set
# os.environ["HUGGINGFACE_API_KEY"] = "your_huggingface_api_key_here"

# Simple RPM tracker for Gemini 2.0 Flash (15 RPM)
class RPMTracker:
    def __init__(self, rpm_limit=15):
        self.rpm_limit = rpm_limit
        self.timestamps = deque()
        self.lock = threading.Lock()

    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            # Remove timestamps older than 60 seconds
            while self.timestamps and now - self.timestamps[0] > 60:
                self.timestamps.popleft()
            if len(self.timestamps) >= self.rpm_limit:
                wait_time = 60 - (now - self.timestamps[0]) + 0.1
                time.sleep(max(wait_time, 0))
            self.timestamps.append(time.time())

gemini_rpm_tracker = RPMTracker(rpm_limit=15)

class StoryGenerator:
    def __init__(self, prompt, num_elements, ending):
        self.prompt = prompt
        self.num_elements = num_elements
        self.ending = ending
        self.orchestrator = None
        self.context = self._initialize_context()
        
    def _initialize_context(self):
        # Extract characters from the prompt
        characters = {
            'protagonist': 'Character 1',
            'antagonist': 'Character 2'
        }
        
        # Try to extract character names from the prompt
        import re
        character_match = re.search(r'featuring\s+([^and]+)\s+and\s+([^in\.]+)', self.prompt, re.IGNORECASE)
        if character_match:
            characters['protagonist'] = character_match.group(1).strip()
            characters['antagonist'] = character_match.group(2).strip()
        
        context = {
            'characters': characters,
            'genre': 'story',  # Default genre
            'setting': 'an interesting location'  # Default setting
        }
        
        # Try to extract genre and setting
        genre_match = re.search(r'Begin a (\w+) story', self.prompt, re.IGNORECASE)
        if genre_match:
            context['genre'] = genre_match.group(1).strip()
        
        setting_match = re.search(r'in\s+([^\.]+)', self.prompt, re.IGNORECASE)
        if setting_match:
            context['setting'] = setting_match.group(1).strip()
            
        return context
    
    def generate_narration(self):
        # Only fallback for narration (main story) using gemini-2.0-flash
        try:
            if not self.orchestrator:
                self.orchestrator = StoryOrchestrator(self.context)
            # Try with gemini-2.0-flash
            return self._generate_with_model_fallback(self.prompt)
        except Exception as e:
            logger.error(f"Narration generation failed: {e}")
            return "[Story generation failed. Please try again.]"
    
    def _generate_with_model_fallback(self, prompt):
        try:
            # Try primary model
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            if hasattr(response, 'text'):
                return response.text
            return str(response)
        except Exception as e:
            if '429' in str(e) or 'quota' in str(e).lower():
                logger.warning("gemini-2.0-flash hit rate limit, using gemma-3n-e4b-it as fallback.")
                try:
                    model = genai.GenerativeModel('gemma-3n-e4b-it')
                    response = model.generate_content(prompt)
                    if hasattr(response, 'text'):
                        return response.text
                    return str(response)
                except Exception as e2:
                    logger.error(f"Fallback model also failed: {e2}")
                    return "[Story generation failed. Please try again.]"
            else:
                raise
    
    def generate_dialogue(self):
        if not self.orchestrator:
            self.orchestrator = StoryOrchestrator(self.context)
        return self.orchestrator.generate_next_story_element(self.prompt)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_story_endpoint():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        num_elements = int(data.get('num_elements', 5))
        ending = data.get('ending', 'happy')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Initialize story generator
        generator = StoryGenerator(prompt, num_elements, ending)
        
        story = []
        images = []
        # Example meta for demonstration; replace with real meta extraction if available
        def extract_meta(text, idx):
            # Dummy meta extraction; replace with real logic
            return {
                "current_scene": f"Scene {idx+1}",
                "last_action": None,
                "story_progress": idx+1,
                "mood": "unease" if idx == 0 else "tense",
                "tension_level": 3 + idx,
                "plot_points": [f"Plot point {idx+1}"],
                "scene_description": text[:200],
                "characters": [
                    {"name": "Protagonist", "description": "Main hero"},
                    {"name": "Antagonist", "description": "Main villain"}
                ]
            }
        for i in range(num_elements):
            try:
                gemini_rpm_tracker.wait_if_needed()
                logger.info(f"Generating story part {i+1}")
                part = generator.generate_narration()  # Use narration as the main part
                meta = extract_meta(part, i)
                story.append({"text": part, "meta": meta})
                # Generate image for this part
                gemini_rpm_tracker.wait_if_needed()
                image_url = generate_visual(part)
                images.append(image_url)
            except Exception as e:
                logger.error(f"Error generating story part {i+1}: {str(e)}")
                story.append({"text": f"Part {i+1} could not be generated due to rate limits. Please try again.", "meta": {}})
                images.append("/static/placeholder.png")
        # Generate and append the conclusion
        gemini_rpm_tracker.wait_if_needed()
        conclusion = f"Conclusion\n\n{generator.generate_narration()}"
        meta = extract_meta(conclusion, num_elements)
        story.append({"text": conclusion, "meta": meta})
        gemini_rpm_tracker.wait_if_needed()
        images.append(generate_visual(conclusion))
        story_texts = [part["text"] for part in story]
        return jsonify({
            'story': story_texts,
            'images': images,
            'status': 'success',
            'message': 'Story generated successfully'
        })
    except Exception as e:
        logger.error(f"Error generating story: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'message': 'Failed to generate story'
        }), 500

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    logger.debug(f"Setup route called with method: {request.method}")
    if request.method == 'POST':
        try:
            # Log form data
            logger.debug(f"Form data received: {request.form}")
            
            # Save character and genre choices
            session['characters'] = {
                'protagonist': request.form.get('protagonist', 'Alex'),
                'antagonist': request.form.get('antagonist', 'Morgan'),
                'support': request.form.get('support', 'Jamie')
            }
            session['genre'] = request.form.get('genre', 'Adventure')
            session['story_context'] = {
                'story': [],
                'emotions': {'happiness': 0.5, 'tension': 0.5},
                'characters': session['characters'],
                'genre': session['genre'],
                'current_scene': 0,
                'plot_points': [],
                'used_actions': [],
                'story_arc': 'setup',
                'discovered_artifacts': [],
                'active_clues': [],
                'current_location': 'general',
                'previous_location': None,
                'current_scene_type': 'general'
            }
            session.permanent = True
            logger.debug("Session data saved successfully")
            return redirect(url_for('story'))
        except Exception as e:
            logger.error(f"Setup error: {str(e)}", exc_info=True)
            flash('An error occurred during setup. Please try again.', 'error')
            return redirect(url_for('setup'))
    return render_template('setup.html')

@app.route('/story', methods=['GET', 'POST'])
def story():
    try:
        logger.debug("Story route called")
        if 'story_context' not in session:
            logger.debug("No story context found, redirecting to setup")
            return redirect(url_for('setup'))
        
        orchestrator = Orchestrator(session['story_context'])
        story_segment = None
        visual_url = None
        actions = []
        moods = []
        
        if request.method == 'POST':
            # Check if a button was clicked
            user_input = request.form.get('user_input') or request.form.get('action') or request.form.get('mood')
            logger.debug(f"User input received: {user_input}")
            
            if not user_input:
                flash('Please provide some input or choose an action.', 'warning')
                return redirect(url_for('story'))
            
            try:
                story_segment, context, actions, moods = orchestrator.generate_next(user_input)
                session['story_context'] = context
                visual_url = generate_visual(story_segment)
                logger.debug("Story segment generated successfully")
            except Exception as e:
                logger.error(f"Story generation error: {str(e)}", exc_info=True)
                flash('An error occurred while generating the story. Please try again.', 'error')
                return redirect(url_for('story'))
                
        elif not session['story_context']['story']:
            # Generate introduction
            try:
                logger.debug("Generating introduction")
                intro, context, actions, moods = orchestrator.generate_next('start')
                session['story_context'] = context
                story_segment = intro
                visual_url = generate_visual(story_segment)
                logger.debug("Introduction generated successfully")
            except Exception as e:
                logger.error(f"Introduction generation error: {str(e)}", exc_info=True)
                flash('An error occurred while starting the story. Please try again.', 'error')
                return redirect(url_for('setup'))
        
        return render_template('story.html', 
                             story=story_segment, 
                             visual=visual_url, 
                             actions=actions, 
                             moods=moods)
                             
    except Exception as e:
        logger.error(f"General story error: {str(e)}", exc_info=True)
        flash('An unexpected error occurred. Please start over.', 'error')
        return redirect(url_for('start'))

@app.route('/initialize_story', methods=['POST'])
def initialize_story():
    global orchestrator
    
    data = request.json
    protagonist = data.get('protagonist', 'Alex')
    antagonist = data.get('antagonist', 'The Shadow King')
    category = data.get('category', 'fantasy')
    
    # Create context with user inputs
    context = {
        'characters': {
            'protagonist': protagonist,
            'antagonist': antagonist,
            'support': 'Maya'  # Default supporting character
        },
        'category': category
    }
    
    # Initialize orchestrator
    orchestrator = StoryOrchestrator(context)
    
    # Generate initial scene
    initial_prompt = f"Begin a {category} story featuring {protagonist} and {antagonist}."
    story_element = orchestrator.generate_next_story_element(initial_prompt)
    
    return jsonify({
        'status': 'success',
        'story_element': story_element
    })

@app.route('/continue_story', methods=['POST'])
def continue_story():
    global orchestrator
    
    if not orchestrator:
        return jsonify({
            'status': 'error',
            'message': 'Story not initialized. Please start a new story.'
        }), 400
    
    data = request.json
    user_prompt = data.get('prompt', '')
    
    try:
        next_element = orchestrator.generate_next_story_element(user_prompt)
        return jsonify({
            'status': 'success',
            'story_element': next_element
        })
    except Exception as e:
        logger.error(f"Error generating story: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def sanitize_text(text):
    # Replace non-latin1 characters with a safe replacement (e.g., '?')
    return text.encode('latin-1', 'replace').decode('latin-1')

@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    data = request.get_json()
    story = data.get('story', [])
    images = data.get('images', [])
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    for idx, (text, img_data) in enumerate(zip(story, images)):
        # Add image if not placeholder
        if img_data and img_data.startswith('data:image'):
            try:
                header, b64data = img_data.split(',', 1)
                img_bytes = base64.b64decode(b64data)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
                    tmp_img.write(img_bytes)
                    tmp_img.flush()
                    pdf.image(tmp_img.name, w=120)
            except Exception as e:
                pass
        pdf.ln(2)
        # Add story part (sanitize text for PDF)
        pdf.multi_cell(0, 10, sanitize_text(text))
        pdf.ln(8)
    # Save PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
        pdf.output(tmp_pdf.name)
        tmp_pdf.flush()
        return send_file(tmp_pdf.name, as_attachment=True, download_name='story.pdf', mimetype='application/pdf')

@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"404 error: {request.url}")
    return render_template('start.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"500 error: {str(e)}", exc_info=True)
    return render_template('start.html'), 500

if __name__ == '__main__':
    app.run(debug=True)