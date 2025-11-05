"""
Mistral 7B Web Chat Interface
A Flask web service for interacting with the Mistral 7B model through a browser
"""

from flask import Flask, render_template, request, jsonify, stream_template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import threading
import time
import json
from datetime import datetime
import logging

# Suppress warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with memory efficient attention.*")

app = Flask(__name__)

# Reduce Flask logging - only show errors
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Global variables for model and tokenizer
model = None
tokenizer = None
model_loading_status = {"status": "not_loaded", "message": "Model not loaded"}

def load_model():
    """Load the Mistral model and tokenizer"""
    global model, tokenizer, model_loading_status
    
    try:
        model_loading_status = {"status": "loading", "message": "Loading model..."}
        print("🔄 Loading Mistral 7B model...")
        
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,  # Use cached files
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True
        )
        
        # Set up tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model_loading_status = {"status": "loaded", "message": "Model loaded successfully"}
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        model_loading_status = {"status": "error", "message": f"Error loading model: {str(e)}"}
        print(f"❌ Error loading model: {e}")

def generate_response(messages, max_new_tokens=300, temperature=0.7, do_sample=True):
    """Generate response from the model"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return "❌ Model not loaded. Please wait for model to load."
    
    try:
        # Apply chat template
        model_inputs = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            padding=True
        )
        
        # Create attention mask
        attention_mask = (model_inputs != tokenizer.pad_token_id).long()
        
        # Move to GPU if available
        device = next(model.parameters()).device
        model_inputs = model_inputs.to(device)
        attention_mask = attention_mask.to(device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract only the new generated part
        original_length = len(tokenizer.batch_decode(model_inputs, skip_special_tokens=True)[0])
        new_response = response[original_length:].strip()
        
        return new_response
        
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

@app.route('/')
def index():
    """Main chat interface"""
    return render_template('chat.html')

@app.route('/status')
def status():
    """Get model loading status"""
    return jsonify(model_loading_status)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        conversation_history = data.get('history', [])
        
        # Generation parameters
        max_tokens = data.get('max_tokens', 300)
        temperature = data.get('temperature', 0.7)
        do_sample = data.get('do_sample', True)
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Add user message to conversation
        messages = conversation_history + [{"role": "user", "content": user_message}]
        
        # Generate response
        bot_response = generate_response(messages, max_tokens, temperature, do_sample)
        
        # Add bot response to conversation
        messages.append({"role": "assistant", "content": bot_response})
        
        return jsonify({
            'response': bot_response,
            'history': messages,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear')
def clear_conversation():
    """Clear conversation history"""
    return jsonify({'message': 'Conversation cleared'})

if __name__ == '__main__':
    # Start model loading in background
    loading_thread = threading.Thread(target=load_model)
    loading_thread.daemon = True
    loading_thread.start()
    
    print("🚀 Starting Mistral Web Chat Service...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("⏳ Model is loading in the background...")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)