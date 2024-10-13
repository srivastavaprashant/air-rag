from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig, StoppingCriteria
from biokb.settings import MAX_LENGTH, DEVICE_NAME, MAX_TIME, MODEL_CACHE_DIR
from biokb.helpers import get_generation_config
from biokb.utils import get_logger
from torch import Tensor, float16
import re
from typing import Tuple
from pathlib import Path
import logging
root = Path(__file__).parent.parent

logger = get_logger('air-llm', debug=True)
logger.info("Starting AIR llm...")

def get_4bitconfig():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=float16,  # Use float16 for computation
        bnb_4bit_use_double_quant=True,  # Use double quantization
        bnb_4bit_quant_type="nf4"  # Use NormalFloat4 quantization
    )

class AIRLLM:    
    def __init__(
        self, 
        model_name: str,
        logger: logging.Logger,
        generation_config: GenerationConfig = get_generation_config(t=0.1, p=1.5),
        stopping_criteria: StoppingCriteria = None,
        load_in_4bit: bool = True,
    ):  
        self.generation_config = generation_config
        self.stopping_criteria = stopping_criteria
        self.max_length = MAX_LENGTH
        self.load_in_4bit = load_in_4bit
        self.logger = logger or logging.getLogger(__name__)
        self.model, self.tokenizer = self.initialize_model_and_tokenizer(
            model_name
        )

    def initialize_model_and_tokenizer(
        self,
        model_name: str,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Initialize the model and tokenizer for the given model name.
        Args:
            model_name (str): The name of the model to load.
            device_name (str): The name of the device to use.
            max_length (int): The maximum length of the input sequence.
            load_in_4bit (bool): Whether to load the model in 4-bit mode.
        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: The model and tokenizer.
        """
        # Initialize the model
        self.logger.info(f"Loading model: {model_name}")

        if self.load_in_4bit:
            self.logger.info("Loading model in 4-bit mode")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                cache_dir=MODEL_CACHE_DIR,
                quantization_config=get_4bitconfig(),
                max_length=self.max_length
            )
        else:
            bnb_config = None
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                cache_dir=MODEL_CACHE_DIR,
                quantization_config=bnb_config,
                max_length=self.max_length
            ).to(DEVICE_NAME)
        self.logger.info(f"Model loaded: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=MODEL_CACHE_DIR
        )
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def generate_text(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        repetition_penalty: float = 1.5,
        stopping_criteria: StoppingCriteria = None,
        max_length: int = MAX_LENGTH,
    ):  
        if max_length is None:
            max_length = self.max_length
        output = self.model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_length=max_length, 
            stopping_criteria=stopping_criteria,
            max_time=MAX_TIME,
            generation_config=self.generation_config,
            repetition_penalty=repetition_penalty,
            do_sample=False,
        )
        return output

    def decode_output(self, output: Tensor, input_cutoff:int = None):
        if input_cutoff is None:
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        else:
            response = self.tokenizer.decode(output[0][input_cutoff:], skip_special_tokens=True)

        return response

    def tokenize_input(self, input_text: str):
        output = self.tokenizer.encode_plus(
            input_text, 
            return_tensors="pt", 
            add_special_tokens=True,
            padding=True,
            truncation=True
        ).to(DEVICE_NAME)
        return output['input_ids'], output['attention_mask']

    def tokenize_chat_input(self, messages: list[dict], tools: list[callable] = None):
        output = self.tokenizer.apply_chat_template(
            conversation = messages,
            tools = tools,
            return_tensors="pt", 
            tokenize=True, 
            add_generation_prompt=True,
            return_dict=True
        )
        input_ids = output["input_ids"].to(DEVICE_NAME)
        attn_mask = output["attention_mask"].to(DEVICE_NAME)
        return input_ids, attn_mask

    def get_chat_response(self, messages: list[dict], tools: list[callable] = None):
        self.logger.debug(f"Calling get_chat_response: {messages}")
        input_ids, attn_mask = self.tokenize_chat_input(messages, tools)
        output = self.generate_text(input_ids, attn_mask)
        response = self.decode_output(output, len(input_ids[0]))
        
        return response
    
    def get_response(self, input_text: str):
        self.logger.debug(f"Calli with input_text: {input_text}")
        input_ids, attn_mask = self.tokenize_input(input_text)
        output = self.generate_text(input_ids, attn_mask)
        response = self.decode_output(output, len(input_ids[0]))
        return response