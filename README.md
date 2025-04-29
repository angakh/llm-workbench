# LLM-Workbench

LLM-Workbench is a flexible framework for working with multiple Large Language Model (LLM) providers and models. It provides a user-friendly interface for comparing different models, managing prompt templates, and building custom applications.

## Features

- **Multiple Provider Support**: Connect to OpenAI, Anthropic, Ollama, and more
- **Model Selection**: Select and compare multiple models across providers
- **Prompt Template Management**: Create, edit, and use templates for consistent prompting
- **Chat Interface**: Built-in chat interface for comparing model responses
- **Extensible Framework**: Build your own applications on top of the workbench
- **Concurrent Processing**: Generate responses from multiple models in parallel

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/llm-workbench.git
cd llm-workbench
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys:
Create a `.env` file in the root directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Usage

1. Launch the application:
```bash
streamlit run app.py
```

2. Select providers and models in the sidebar
3. Use the built-in chat interface or extend the app with your own code

## Components

- **llm_providers.py**: Connects to various LLM providers and manages model selection
- **prompt_manager.py**: Handles prompt template creation, editing, and formatting
- **app.py**: Main Streamlit application with UI for model selection and chat

## Extending the Framework

LLM-Workbench is designed to be easily extended. Here's how to build on top of it:

1. Access the provider manager:
```python
provider_manager = get_provider_manager()
```

2. Get selected models:
```python
selected_models = provider_manager.get_selected_models()
```

3. Generate responses:
```python
responses = provider_manager.generate_responses_from_all_selected(
    prompt="Your prompt here",
    temperature=0.7,
    max_tokens=1000
)
```

4. Process responses:
```python
for response in responses:
    provider = response["provider"]
    model = response["model"]
    text = response["text"]
    time_taken = response["time"]
    error = response["error"]
    
    # Do something with the response
```

## Supported Providers

- **Ollama**: Local LLM serving platform
- **OpenAI**: GPT models via API
- **Anthropic**: Claude models via API

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.