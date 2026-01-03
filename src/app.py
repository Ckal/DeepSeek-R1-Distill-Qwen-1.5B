import gradio as gr
import os
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch

@dataclass
class AppConfig:
    """Configuration for the chat application"""
    MODEL_NAME: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    MAX_LENGTH: int = 4096
    DEFAULT_TEMP: float = 0.7
    CHAT_HEIGHT: int = 450
    PAD_TOKEN: str = "[PAD]"

CHAT_TEMPLATE = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}
{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}
{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
        <｜User｜>{{message['content']}}
    {%- endif -%}
    {%- if message['role'] == 'assistant' and message['content'] is not none -%}
        {% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}<｜Assistant｜>{{content}}<｜end▁of▁sentence｜>
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt %}<｜Assistant｜>{% endif -%}"""

CSS = """
:root {
    --primary-color: #1565c0;
    --secondary-color: #1976d2;
    --text-primary: rgba(0, 0, 0, 0.87);
    --text-secondary: rgba(0, 0, 0, 0.65);
    --spacing-lg: 30px;
    --border-radius: 100vh;
    --shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: var(--spacing-lg);
}

.header {
    text-align: center;
    margin-bottom: var(--spacing-lg);
    padding: 20px;
    background: var(--primary-color);
    color: white;
    border-radius: 8px;
}

.header h1 {
    font-size: 28px;
    margin-bottom: 8px;
}

.header p {
    font-size: 18px;
    opacity: 0.9;
}

#chatbot {
    border-radius: 8px;
    background: white;
    box-shadow: var(--shadow);
}

.message {
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
}

.user-message {
    background: var(--primary-color);
    color: white;
}

.assistant-message {
    background: #f5f5f5;
}
"""

class ChatBot:
    def __init__(self, config: AppConfig):
        self.config = config
        self.setup_model()

    def setup_model(self):
        """Initialize the model and tokenizer with proper configuration"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.config.PAD_TOKEN})
            
        self.tokenizer.chat_template = CHAT_TEMPLATE

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16  # Use half precision for better memory efficiency
        )
        
        # Resize token embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))

    def _convert_history_to_messages(self, history: List[tuple]) -> List[Dict[str, str]]:
        """Convert tuple history to message format"""
        messages = []
        for user, assistant in history:
            messages.extend([
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ])
        return messages

    def generate_response(self, 
                         message: str,
                         history: List[tuple],
                         temperature: float,
                         max_new_tokens: int) -> str:
        """Generate streaming response with improved error handling and attention mask"""
        try:
            # Convert history to messages format
            conversation = self._convert_history_to_messages(history)
            conversation.append({"role": "user", "content": message})

            # Prepare input with attention mask
            inputs = self.tokenizer.apply_chat_template(
                conversation,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(self.model.device)
            
            attention_mask = torch.ones_like(inputs)
            
            streamer = TextIteratorStreamer(
                self.tokenizer,
                timeout=10.0,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generate_kwargs = {
                "input_ids": inputs,
                "attention_mask": attention_mask,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "temperature": temperature,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
            thread.start()

            return "".join([chunk for chunk in self._process_stream(streamer)])

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _process_stream(self, streamer) -> str:
        """Process the streaming output with improved text cleaning"""
        outputs = []
        for text in streamer:
            # Clean special tokens and normalize whitespace
            text = (text.replace("<think>", "[think]")
                      .replace("</think>", "[/think]")
                      .replace("<｜end▁of▁sentence｜>", "")
                      .strip())
            outputs.append(text)
            yield "".join(outputs)

def create_gradio_interface(chatbot: ChatBot):
    """Create the Gradio interface with improved layout and modern message format"""
    examples = [
        ['Tell me about artificial intelligence.'],
        ['What are neural networks?'],
        ['Explain machine learning in simple terms.']
    ]

    with gr.Blocks(css=CSS) as demo:
        with gr.Column(elem_classes="container"):
            with gr.Column(elem_classes="header"):
                gr.Markdown("# DeepSeek R1 Chat Interface")
                gr.Markdown("An efficient and responsive chat interface powered by DeepSeek R1 Distill")

            chatbot_interface = gr.Chatbot(
                height=chatbot.config.CHAT_HEIGHT,
                container=True,
                elem_id="chatbot",
                type="messages"  # Use modern message format
            )

            interface = gr.ChatInterface(
                fn=chatbot.generate_response,
                chatbot=chatbot_interface,
                additional_inputs=[
                    gr.Slider(
                        minimum=0, maximum=1,
                        value=chatbot.config.DEFAULT_TEMP,
                        label="Temperature",
                        info="Higher values make the output more random"
                    ),
                    gr.Slider(
                        minimum=128, maximum=chatbot.config.MAX_LENGTH,
                        value=1024,
                        label="Max new tokens",
                        info="Maximum length of the generated response"
                    ),
                ],
                examples=examples,
                cache_examples=False,
                #retry_btn="Regenerate Response",
                #undo_btn="Undo Last",
                #clear_btn="Clear Chat",
            )

    return demo

if __name__ == "__main__":
    config = AppConfig()
    chatbot = ChatBot(config)
    demo = create_gradio_interface(chatbot)
    demo.launch(
        debug=True,
        share=False,  # Set to True to create a public link
        server_name="0.0.0.0",
        server_port=7860,
       # ssr=False  # Disable SSR to avoid experimental features
    )