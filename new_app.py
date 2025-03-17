import os
import torch
import gradio as gr
import torchaudio
import numpy as np
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple
import tempfile

# Import the CSM model
try:
    from generator import load_csm_1b, Segment
except ImportError:
    # Define a mock Segment class for local testing if the actual model is not installed
    @dataclass
    class Segment:
        text: str
        speaker: int
        audio: Optional[torch.Tensor] = None

    def load_csm_1b(device="cuda"):
        class MockGenerator:
            def __init__(self):
                self.sample_rate = 24000
                
            def generate(self, text, speaker, context, max_audio_length_ms):
                # Return dummy audio for testing UI without model
                return torch.zeros(int(self.sample_rate * max_audio_length_ms / 1000))
        
        print("WARNING: Using mock CSM model for UI testing")
        return MockGenerator()

# Set up device
if torch.backends.mps.is_available():
    #device = "mps"
    device = "cpu"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

# Initialize the model
try:
    generator = load_csm_1b(device=device)
    MODEL_LOADED = True
    print("CSM model loaded successfully!")
except Exception as e:
    MODEL_LOADED = False
    print(f"Failed to load CSM model: {e}")
    print("UI will run in demonstration mode without generating actual audio")

# Global variables to store conversation history
conversation_history = []

def get_temp_file_path(suffix):
    """Generate a temporary path for files"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    return tmp.name

def get_temp_wav_path():
    """Generate a temporary path for wav files"""
    return get_temp_file_path(".wav")

def get_temp_txt_path():
    """Generate a temporary path for txt files"""
    return get_temp_file_path(".txt")

def convert_ebook_to_text(ebook_path):
    """Convert an ebook file to text using Calibre's ebook-convert tool"""
    if not ebook_path:
        return None, "No ebook file provided."
    
    try:
        txt_path = get_temp_txt_path()
        
        # Use Calibre's ebook-convert tool to convert the ebook to text
        result = subprocess.run(
            ["ebook-convert", ebook_path, txt_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Read the converted text
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean up
        os.remove(txt_path)
        
        return text, "Ebook successfully converted to text!"
    except subprocess.CalledProcessError as e:
        return None, f"Error converting ebook: {e.stderr}"
    except Exception as e:
        return None, f"Error processing ebook: {str(e)}"

def generate_speech(text, speaker_id, max_length_ms=10000):
    """Generate speech without context"""
    if not MODEL_LOADED:
        return "Model not loaded. This is a UI demonstration only.", None
    
    if not text.strip():
        return "Please enter text to generate speech.", None
    
    try:
        # Generate audio
        audio = generator.generate(
            text=text,
            speaker=speaker_id,
            context=[],
            max_audio_length_ms=max_length_ms,
        )
        
        # Convert to numpy
        audio_np = audio.cpu().numpy()
        
        # Create temporary wav file
        out_path = get_temp_wav_path()
        torchaudio.save(out_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
        
        # Add to conversation history
        global conversation_history
        conversation_history.append(Segment(text=text, speaker=speaker_id, audio=audio))
        
        return f"Generated speech for: {text}", out_path
    except Exception as e:
        return f"Error generating speech: {str(e)}", None

def load_audio_file(audio_path):
    """Load and preprocess audio file"""
    if not audio_path:
        return None, "No audio file provided."
    
    try:
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        audio_tensor = torchaudio.functional.resample(
            audio_tensor.squeeze(0), 
            orig_freq=sample_rate, 
            new_freq=generator.sample_rate if MODEL_LOADED else 24000
        )
        return audio_tensor, "Audio loaded successfully!"
    except Exception as e:
        return None, f"Error loading audio: {str(e)}"

def add_context(text, speaker_id, audio_path):
    """Add a context segment to the conversation history"""
    if not text.strip():
        return "Please enter text for the context segment."
    
    if audio_path:
        audio_tensor, message = load_audio_file(audio_path)
        if audio_tensor is None:
            return message
    else:
        audio_tensor = None
        
    global conversation_history
    conversation_history.append(Segment(text=text, speaker=speaker_id, audio=audio_tensor))
    
    return f"Added context: Speaker {speaker_id} said '{text}'"

def generate_with_context(text, speaker_id, max_length_ms=10000):
    """Generate speech with the current conversation history as context"""
    if not MODEL_LOADED:
        return "Model not loaded. This is a UI demonstration only.", None
    
    if not text.strip():
        return "Please enter text to generate speech.", None
    
    if not conversation_history:
        return "No conversation history. Please add context first or use the basic generation tab.", None
    
    try:
        # Generate audio with context
        audio = generator.generate(
            text=text,
            speaker=speaker_id,
            context=conversation_history,
            max_audio_length_ms=max_length_ms,
        )
        
        # Convert to numpy
        audio_np = audio.cpu().numpy()
        
        # Create temporary wav file
        out_path = get_temp_wav_path()
        torchaudio.save(out_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
        
        # Add to conversation history
        conversation_history.append(Segment(text=text, speaker=speaker_id, audio=audio))
        
        return f"Generated speech with context for: {text}", out_path
    except Exception as e:
        return f"Error generating speech: {str(e)}", None

def clear_history():
    """Clear the conversation history"""
    global conversation_history
    conversation_history = []
    return "Conversation history cleared!"

def view_history():
    """View the current conversation history"""
    if not conversation_history:
        return "No conversation history."
    
    history_text = "Current Conversation History:\n\n"
    for i, segment in enumerate(conversation_history):
        history_text += f"{i+1}. Speaker {segment.speaker}: \"{segment.text}\"\n"
        history_text += f"   Audio: {'Available' if segment.audio is not None else 'Not available'}\n"
    
    return history_text

def process_input_method(input_method, text_input, ebook_file, speaker_id, max_length_ms):
    """Process either text input or ebook file based on the selected input method"""
    if input_method == "text":
        if not text_input.strip():
            return "Please enter text to generate speech.", None
        return generate_speech(text_input, speaker_id, max_length_ms)
    else:  # ebook method
        if not ebook_file:
            return "Please upload an ebook file.", None
        
        # Convert ebook to text
        text, msg = convert_ebook_to_text(ebook_file)
        if text is None:
            return msg, None
        
        # Generate speech from the converted text
        return generate_speech(text, speaker_id, max_length_ms)

def create_ui():
    """Create the Gradio UI"""
    with gr.Blocks(title="CSM - Conversational Speech Model") as app:
        gr.Markdown("""
        # CSM - Conversational Speech Model Demo
        
        Generate conversational speech using Sesame's CSM model. This app allows you to:
        1. Generate speech from text or ebook files with different speakers
        2. Build a conversational context
        3. Generate speech that matches the conversation style/flow
        
        *Note: The model works best when provided with context.*
        """)
        
        with gr.Tabs():
            with gr.TabItem("Basic Generation"):
                with gr.Row():
                    with gr.Column():
                        # Input method selection
                        input_method = gr.Radio(
                            ["text", "ebook"], 
                            label="Input Method", 
                            value="text"
                        )
                        
                        # Text input (shown when text method is selected)
                        with gr.Group(visible=True) as text_group:
                            basic_text = gr.Textbox(
                                label="Text to generate", 
                                placeholder="Enter text here...",
                                lines=3
                            )
                        
                        # Ebook input (shown when ebook method is selected)
                        with gr.Group(visible=False) as ebook_group:
                            ebook_file = gr.File(
                                label="Upload Ebook File",
                                file_types=[".epub", ".mobi", ".azw", ".azw3", ".fb2", ".docx", ".pdf"]
                            )
                            ebook_info = gr.Textbox(
                                label="Ebook Info", 
                                value="Supported formats: .epub, .mobi, .azw, .azw3, .fb2, .docx, .pdf", 
                                interactive=False
                            )
                        
                        basic_speaker = gr.Slider(
                            minimum=0, 
                            maximum=5, 
                            value=0, 
                            step=1, 
                            label="Speaker ID"
                        )
                        basic_length = gr.Slider(
                            minimum=1000, 
                            maximum=30000, 
                            value=10000, 
                            step=1000, 
                            label="Max Audio Length (ms)"
                        )
                        basic_btn = gr.Button("Generate Speech")
                    
                    with gr.Column():
                        basic_output = gr.Textbox(label="Output Message")
                        basic_audio = gr.Audio(label="Generated Audio")
            
            with gr.TabItem("Contextual Generation"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Add Context")
                        context_text = gr.Textbox(
                            label="Context Text", 
                            placeholder="Enter what was said...",
                            lines=2
                        )
                        context_speaker = gr.Slider(
                            minimum=0, 
                            maximum=5, 
                            value=0, 
                            step=1, 
                            label="Speaker ID"
                        )
                        context_audio = gr.Audio(
                            label="Context Audio (Optional)", 
                            type="filepath"
                        )
                        add_context_btn = gr.Button("Add to Conversation")
                        context_output = gr.Textbox(label="Context Status")
                        
                    with gr.Column():
                        gr.Markdown("### Generate with Context")
                        gen_text = gr.Textbox(
                            label="Text to generate", 
                            placeholder="Enter response text...",
                            lines=2
                        )
                        gen_speaker = gr.Slider(
                            minimum=0, 
                            maximum=5, 
                            value=0, 
                            step=1, 
                            label="Speaker ID"
                        )
                        gen_length = gr.Slider(
                            minimum=1000, 
                            maximum=30000, 
                            value=10000, 
                            step=1000, 
                            label="Max Audio Length (ms)"
                        )
                        gen_btn = gr.Button("Generate with Context")
                        gen_output = gr.Textbox(label="Output Message")
                        gen_audio = gr.Audio(label="Generated Audio")
                
                with gr.Row():
                    history_btn = gr.Button("View Conversation History")
                    clear_btn = gr.Button("Clear Conversation History")
                
                history_output = gr.Textbox(label="Conversation History", lines=10)
        
        # Set up event handlers for input method radio button
        input_method.change(
            fn=lambda x: [gr.update(visible=(x == "text")), gr.update(visible=(x == "ebook"))],
            inputs=[input_method],
            outputs=[text_group, ebook_group]
        )
        
        # Set up other event handlers
        basic_btn.click(
            process_input_method, 
            inputs=[input_method, basic_text, ebook_file, basic_speaker, basic_length], 
            outputs=[basic_output, basic_audio]
        )
        
        add_context_btn.click(
            add_context, 
            inputs=[context_text, context_speaker, context_audio], 
            outputs=[context_output]
        )
        
        gen_btn.click(
            generate_with_context, 
            inputs=[gen_text, gen_speaker, gen_length], 
            outputs=[gen_output, gen_audio]
        )
        
        history_btn.click(
            view_history, 
            inputs=[], 
            outputs=[history_output]
        )
        
        clear_btn.click(
            clear_history, 
            inputs=[], 
            outputs=[history_output]
        )
        
    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch(share=False)  # Set share=False if you don't want a public link
