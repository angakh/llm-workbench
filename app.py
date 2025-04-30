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

# Add custom CSS for styling
st.markdown("""
<style>
    /* Message separator */
    .message-separator {
        margin: 15px 0;
        border-top: 1px solid #ddd;
    }
    
    /* Provider model header styling */
    .provider-model-header {
        font-weight: bold;
        color: #555;
        margin-bottom: 8px;
    }
    
    /* Custom styling for the chat container */
    .stChatMessage {
        padding: 8px !important;
    }
    
    /* Improve spacing between chat messages */
    .stChatMessageContent {
        margin: 4px 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state if not already initialized
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'messages' not in st.session_state:
    st.session_state.messages = []
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

# Build Your Own Application accordion at the top of the page
with st.expander("Build Your Own Application", expanded=False):
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
    
    # Optional: Add some more interactive elements here
    if st.checkbox("Show simple example"):
        st.code("""
# Example: Generate a response and extract key points
custom_prompt = "Summarize the key features of LLMs"
        
# Get responses from all selected models
responses = provider_manager.generate_responses_from_all_selected(
    prompt=custom_prompt,
    temperature=0.7,
    max_tokens=500
)

# Display the responses
for response in responses:
    st.subheader(f"{response['provider']} / {response['model']}")
    st.write(response['text'])
    st.caption(f"Response time: {response['time']:.2f}s")
        """, language="python")

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

# Define provider-to-emoji mapping for avatars
def get_provider_emoji(provider_name):
    """Return an emoji for the provider to use as avatar"""
    provider_emojis = {
        "openai": "🟢",    # Green circle for OpenAI
        "anthropic": "🟣",  # Purple circle for Anthropic
        "ollama": "🟠",     # Orange circle for Ollama
        # Add more providers as needed
    }
    # Default to a blue circle if provider not in mapping
    return provider_emojis.get(provider_name.lower(), "🔵")

# Display existing chat history with native Streamlit components
for i, (prompt, responses) in enumerate(st.session_state.chat_history):
    # User message
    with st.chat_message("user"):
        st.write(prompt)
    
    # AI responses
    for response in responses:
        # Get appropriate emoji for the provider
        provider_emoji = get_provider_emoji(response['provider'])
        
        # Show response with provider emoji as avatar
        with st.chat_message("assistant", avatar=provider_emoji):
            # Add provider/model header
            st.markdown(f"**{response['provider']} / {response['model']}**")
            
            if response.get("error"):
                st.error(f"Error: {response['error']}")
            else:
                st.write(response["text"])
            
            st.caption(f"Response time: {response['time']:.2f}s")

# Check if any models are selected
any_models_selected = any(len(models) > 0 for models in st.session_state.selected_models.values())

# Chat input using Streamlit's native chat_input
user_prompt = st.chat_input(
    "Enter your prompt:", 
    key="user_prompt_chat_input", 
    disabled=not any_models_selected or st.session_state.generating_response
)

if user_prompt and not st.session_state.generating_response:
    # Start generating response
    st.session_state.generating_response = True
    
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
            
            # Show the user's message immediately 
            with st.chat_message("user"):
                st.write(user_prompt)
            
            # Generate responses from all selected models
            responses = provider_manager.generate_responses_from_all_selected(
                prompt=formatted_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Sort responses by provider and model name for consistent ordering
            responses.sort(key=lambda x: (x["provider"], x["model"]))
            
            # Display each model's response as it comes in
            for response in responses:
                # Get appropriate emoji for the provider
                provider_emoji = get_provider_emoji(response['provider'])
                
                # Show response with provider emoji as avatar
                with st.chat_message("assistant", avatar=provider_emoji):
                    # Add provider/model header
                    st.markdown(f"**{response['provider']} / {response['model']}**")
                    
                    if response.get("error"):
                        st.error(f"Error: {response['error']}")
                    else:
                        st.write(response["text"])
                    
                    st.caption(f"Response time: {response['time']:.2f}s")
            
            # Add to chat history
            st.session_state.chat_history.append((user_prompt, responses))
            
            # Reset the generation flag
            st.session_state.generating_response = False
            
        except Exception as e:
            logger.error(f"Error generating responses: {e}")
            st.error(f"Error generating responses: {str(e)}")
            st.session_state.generating_response = False
            
    # Use rerun to update the UI
    st.rerun()