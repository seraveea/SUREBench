class QwenPipeline:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def __call__(self, inputs, temperature, max_new_tokens, enable_thinking=False):
        messages = inputs
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
    
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
        # Parse content (skip thinking content)
        # Sometimes the thinking trace exceeds max_new_tokens and misses the </think> token,
        # so thinking content may leak into the final output.
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)  # </think> token
        except ValueError:
            index = 0
        
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        # print(thinking_content)
        # print(content)
        return thinking_content, content
    

class GemmaPipeline:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    def __call__(self, inputs, temperature, max_new_tokens, enable_thinking=False):
        messages = inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        model_inputs = self.processor(text=text, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        
        input_len = model_inputs["input_ids"].shape[-1]

        outputs = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens, temperature=temperature)
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        parse_response = self.processor.parse_response(response)

        return parse_response


class QwenVLLMPipeline:
    """Qwen vLLM pipeline with enable_thinking support"""
    
    def __init__(self, llm, tokenizer):
        """
        Args:
            llm: vLLM LLM instance
            tokenizer: AutoTokenizer instance
        """
        self.llm = llm
        self.tokenizer = tokenizer
    
    def __call__(self, inputs, temperature, max_new_tokens, enable_thinking=False):
        """
        Generate response with optional thinking content
        
        Args:
            inputs: Chat messages (list of dicts with role/content)
            temperature: Sampling temperature
            max_new_tokens: Max tokens to generate
            enable_thinking: Whether to enable thinking mode
        
        Returns:
            (thinking_content, content) tuple
        """
        from vllm import SamplingParams
        
        messages = inputs
        # Build prompt with enable_thinking support
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        # Use vLLM to generate
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
        )
        
        outputs = self.llm.generate(
            [text],
            sampling_params,
            use_tqdm=False
        )
        
        # Extract generated text
        generated_text = outputs[0].outputs[0].text
        
        # Parse thinking content and actual content
        # When enable_thinking=True, the response includes <think>...</think> tags
        if enable_thinking and "<think>" in generated_text:
            try:
                # Find thinking end tag
                think_start = generated_text.find("<think>")
                think_end = generated_text.find("</think>")
                
                if think_start != -1 and think_end != -1:
                    thinking_content = generated_text[think_start + 7:think_end].strip("\n")
                    # Content is everything after </think>
                    content = generated_text[think_end + 8:].strip("\n")
                else:
                    # Fallback if tags are malformed
                    thinking_content = ""
                    content = generated_text
            except Exception:
                thinking_content = ""
                content = generated_text
        else:
            # No thinking content
            thinking_content = ""
            content = generated_text.strip("\n")
        
        return thinking_content, content