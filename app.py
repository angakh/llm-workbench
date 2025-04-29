"""
app.py - Main Application Entry Point for LLM-Workbench

This file serves as the main entry point for the LLM-Workbench application, a Streamlit-based
tool that allows users to interact with multiple LLM providers and models.

Key responsibilities:
1. User Interface Management: Creates and manages the Streamlit UI, including provider/model selection,
   chat interface, and settings sidebar.
2. Session State Management: Maintains application state across interactions, storing model selections,
   chat history, and user preferences.
3. Provider & Resource Management: Initializes and provides access to core services:
   - ProviderManager: Handles communication with various LLM providers (Ollama, OpenAI, Anthropic)
   - PromptManager: Manages prompt templates for AI interactions
4. Workflow Coordination: Orchestrates the main application workflows:
   - Model selection and configuration
   - Chat interaction with selected LLMs
   - Prompt template management

The application follows a single-page design with a sidebar for settings and a main area for
either the chat interface or custom user interfaces built on top of the framework.

Dependencies:
- External Python packages: streamlit, dotenv, logging
- Custom modules: llm_providers, prompt_manager
- Environment variables: Configuration loaded from .env file
"""

import streamlit as st
import os
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
import warnings
from concurrent.futures import ThreadPoolExecutor

# Import our custom modules
from llm_providers import ProviderManager
from prompt_manager import PromptManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize streamlit
st.set_page_config(page_title="LLM-Workbench", page_icon="🔬", layout="wide")

# Add custom CSS for chat styling
st.markdown("""
<style>
    /* User message styling */
    .user-message {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 10px 15px;
        margin-bottom: 10px;
    }
    
    /* AI message styling */
    .ai-message {
        background-color: #e6f2ff;
        border-radius: 10px;
        padding: 10px 15px;
        margin-bottom: 10px;
    }
    
    /* Message headers */
    .message-header {
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    /* Separator */
    .message-separator {
        margin: 15px 0;
        border-top: 1px solid #ddd;
    }
    
    /* Model header */
    .model-header {
        font-weight: bold;
        margin-top: 5px;
        margin-bottom: 5px;
        color: #555;
    }
    
    /* Response time */
    .response-time {
        font-size: 0.8em;
        color: #888;
        text-align: right;
    }
    
    /* Error message */
    .error-message {
        color: #d9534f;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state if not already initialized
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_prompt_template' not in st.session_state:
    st.session_state.current_prompt_template = "default.txt"
if 'editing_prompt' not in st.session_state:
    st.session_state.editing_prompt = False
if 'generating_response' not in st.session_state:
    st.session_state.generating_response = False
if 'selected_provider' not in st.session_state:
    st.session_state.selected_provider = None
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = {}

# Initialize our managers
@st.cache_resource
def get_provider_manager():
    config_path = os.environ.get("CONFIG_PATH", None)
    return ProviderManager(config_path)

@st.cache_resource
def get_prompt_manager():
    prompts_dir = os.environ.get("PROMPTS_DIR", "prompts")
    return PromptManager(prompts_dir)

# Get our manager instances
provider_manager = get_provider_manager()
prompt_manager = get_prompt_manager()

# App title and description
st.title("🔬 LLM-Workbench")
st.markdown("An integrated workspace for interacting with multiple LLM providers and models")

# Create a sidebar for settings
st.sidebar.title("Settings")

# Select LLM provider
available_providers = provider_manager.get_available_providers()
if available_providers:
    if st.session_state.selected_provider not in available_providers:
        st.session_state.selected_provider = available_providers[0]
        
    provider_name = st.sidebar.selectbox(
        "Select LLM Provider",
        options=available_providers,
        index=available_providers.index(st.session_state.selected_provider),
        key="provider_selectbox"
    )
    
    st.session_state.selected_provider = provider_name
    
    # Get models for the selected provider
    available_models = provider_manager.get_provider_models(provider_name)
    
    if available_models:
        st.sidebar.markdown("### Available Models")
        st.sidebar.markdown("Select models to use in the comparison:")
        
        # Initialize the selected models dictionary for this provider if it doesn't exist
        if provider_name not in st.session_state.selected_models:
            st.session_state.selected_models[provider_name] = set()
        
        # Create checkboxes for each model
        for model in available_models:
            model_selected = model in st.session_state.selected_models.get(provider_name, set())
            
            if st.sidebar.checkbox(
                model, 
                value=model_selected,
                key=f"model_{provider_name}_{model}"
            ):
                # Add to selected models
                provider_manager.select_model(provider_name, model, True)
                st.session_state.selected_models[provider_name].add(model)
            else:
                # Remove from selected models
                provider_manager.select_model(provider_name, model, False)
                if model in st.session_state.selected_models[provider_name]:
                    st.session_state.selected_models[provider_name].remove(model)
    else:
        st.sidebar.warning(f"No models available for {provider_name}")
else:
    st.sidebar.warning("No LLM providers available. Please check your installation and API keys.")
    provider_name = None

# Prompt template selection
st.sidebar.markdown("---")
st.sidebar.subheader("Prompt Templates")

# Get available prompt templates
prompt_files = prompt_manager.get_prompt_files()

if prompt_files:
    # Select prompt template
    st.session_state.current_prompt_template = st.sidebar.selectbox(
        "Select Prompt Template",
        options=prompt_files,
        index=prompt_files.index("default.txt") if "default.txt" in prompt_files else 0,
        key="prompt_template_selectbox"
    )
    
    # Display the current prompt template
    current_prompt_content = prompt_manager.load_prompt(st.session_state.current_prompt_template)
    
    if st.sidebar.button("Edit Prompt Template", key="edit_prompt_button"):
        st.session_state.editing_prompt = True
    
    if st.sidebar.button("Create New Prompt", key="create_prompt_button"):
        st.session_state.editing_prompt = True
        st.session_state.current_prompt_template = "new_prompt.txt"
        current_prompt_content = ""
else:
    st.sidebar.warning("No prompt templates available")
    current_prompt_content = ""

# Prompt template editor
if st.session_state.editing_prompt:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Edit Prompt Template")
    
    prompt_name = st.sidebar.text_input("Prompt Name", value=st.session_state.current_prompt_template, key="prompt_name_input")
    prompt_content = st.sidebar.text_area("Prompt Content", value=current_prompt_content, height=300, key="prompt_content_area")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Save", key="save_prompt"):
            if prompt_manager.save_prompt(prompt_name, prompt_content):
                st.session_state.current_prompt_template = prompt_name
                st.session_state.editing_prompt = False
                st.rerun()
            else:
                st.error("Failed to save prompt template")
    
    with col2:
        if st.button("Cancel", key="cancel_edit"):
            st.session_state.editing_prompt = False
            st.rerun()

# Generate response parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Generation Parameters")

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01,
    help="Controls randomness: Lower values are more focused, higher values more creative"
)

max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=100,
    max_value=4000,
    value=1000,
    step=100,
    help="Maximum length of the generated response"
)

# Main content area

# Display selected models summary
selected_models = provider_manager.get_selected_models()
if selected_models:
    st.markdown("### Selected Models")
    
    models_text = []
    for provider, models in selected_models.items():
        models_text.append(f"**{provider}**: {', '.join(models)}")
    
    st.markdown(" • ".join(models_text))
else:
    st.warning("No models selected. Please select at least one model in the sidebar.")

# Chat interface
st.markdown("---")
st.markdown("### Chat")

# Display chat history with improved styling
for prompt, responses in st.session_state.chat_history:
    # User message with light gray background
    st.markdown(f"""
    <div class="user-message">
        <div class="message-header">You:</div>
        {prompt}
    </div>
    """, unsafe_allow_html=True)
    
    # Display each model's response
    for response in responses:
        # Determine the message class based on error status
        message_class = "ai-message"
        
        # Format the response
        if response.get("error"):
            response_text = f"""
            <div class="model-header">{response["provider"]} / {response["model"]}</div>
            <div class="error-message">Error: {response["error"]}</div>
            <div class="response-time">Time: {response["time"]:.2f}s</div>
            """
        else:
            response_text = f"""
            <div class="model-header">{response["provider"]} / {response["model"]}</div>
            {response["text"]}
            <div class="response-time">Time: {response["time"]:.2f}s</div>
            """
        
        # AI response with light blue background
        st.markdown(f"""
        <div class="{message_class}">
            {response_text}
        </div>
        """, unsafe_allow_html=True)
    
    # Add a separator
    st.markdown('<div class="message-separator"></div>', unsafe_allow_html=True)

# Input for new messages
user_prompt = st.text_area("Enter your prompt:", height=100, key="user_prompt_input")

# Check if any models are selected
any_models_selected = any(len(models) > 0 for models in st.session_state.selected_models.values())

# Send button - disabled if no models selected
send_button = st.button(
    "Send", 
    key="send_button", 
    disabled=not any_models_selected or st.session_state.generating_response
)

if send_button and user_prompt:
    # Start generating response
    st.session_state.generating_response = True
    st.rerun()

# Trigger response generation if the flag is set
if st.session_state.generating_response and user_prompt:
    with st.spinner("Generating responses..."):
        try:
            # Get the current prompt template
            template = prompt_manager.load_prompt(st.session_state.current_prompt_template)
            if not template:
                st.error(f"Could not load prompt template: {st.session_state.current_prompt_template}")
                formatted_prompt = user_prompt
            else:
                # Format the prompt with the template
                formatted_prompt = prompt_manager.format_prompt(template, user_prompt)
            
            # Generate responses from all selected models
            responses = provider_manager.generate_responses_from_all_selected(
                prompt=formatted_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Sort responses by provider and model name for consistent ordering
            responses.sort(key=lambda x: (x["provider"], x["model"]))
            
            # Add to chat history
            st.session_state.chat_history.append((user_prompt, responses))
            
            # Reset the input and generation flag
            user_prompt = ""
            st.session_state.generating_response = False
            
            # Rerun to update the UI
            st.rerun()
            
        except Exception as e:
            logger.error(f"Error generating responses: {e}")
            st.error(f"Error generating responses: {str(e)}")
            st.session_state.generating_response = False

# Extension point for custom applications
st.markdown("---")
st.markdown("### Build Your Own Application")
st.markdown("""
This is where you can extend the LLM-Workbench to build your own application.
The sidebar provides model selection and prompt management, while you can implement
your own logic here.

Example starter code:
```python
# Access the provider manager
provider_manager = get_provider_manager()

# Access the prompt manager
prompt_manager = get_prompt_manager()

# Get selected models
selected_models = provider_manager.get_selected_models()

# Generate a response using selected models
def my_custom_function(text_input):
    responses = provider_manager.generate_responses_from_all_selected(
        prompt=text_input,
        temperature=0.7,
        max_tokens=1000
    )
    return responses
```
""")

# Footer
st.markdown("---")
st.markdown("LLM-Workbench - A flexible framework for working with multiple LLM providers and models")