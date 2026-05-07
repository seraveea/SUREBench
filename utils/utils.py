import os
import re
from utils.llm_utils import QwenPipeline, GemmaPipeline, QwenVLLMPipeline


def _build_eos_ids(tokenizer):
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)

    for tok in ["<|eot_id|>", "<|endoftext|>", "<end_of_turn>"]:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is not None and tok_id not in eos_ids:
            eos_ids.append(tok_id)

    return eos_ids


def agent_reply(llm, message, temperature=1, enable_thinking=False, num_return_sequences=1, max_new_token=256):
    # Check if llm is an instance of QwenPipeline or QwenVLLMPipeline
    num_return_sequences = int(num_return_sequences)
    if isinstance(llm, (QwenPipeline, QwenVLLMPipeline)):
        # For Qwen models - loop to get multiple results
        outputs = []
        for _ in range(num_return_sequences):
            tnking_output, output = llm(message, temperature=temperature, max_new_tokens=max_new_token, enable_thinking=enable_thinking)
            outputs.append(output)
        return outputs if num_return_sequences > 1 else outputs[0]
    
    if isinstance(llm, GemmaPipeline):
        # For Gemma models, loop calls to get multiple outputs.
        outputs = []
        for _ in range(num_return_sequences):
            output = llm(message, temperature=temperature, max_new_tokens=max_new_token, enable_thinking=enable_thinking)
            print(output)
            outputs.append(output)
        return outputs if num_return_sequences > 1 else outputs[0]
    
    # Check whether llm is a VLLMTextGenerationWrapper.
    elif llm.__class__.__name__ == 'VLLMTextGenerationWrapper':
        # vLLM supports generating multiple outputs in one call.
        eos_pair = _build_eos_ids(llm.tokenizer)
        
        outputs = llm(message, max_new_tokens=max_new_token, 
                      temperature=temperature, 
                      do_sample=True, 
                      top_p=0.9, 
                      n=num_return_sequences)
        
        prompt = llm._build_prompt(message)
        results = [o["generated_text"][len(prompt):] for o in outputs]
        return results if num_return_sequences > 1 else results[0]

    elif llm.__class__.__name__ == 'OllamaChatWrapper':
        outputs = llm(
            message,
            max_new_tokens=max_new_token,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            n=num_return_sequences,
            enable_thinking=enable_thinking,
        )
        results = [o.get("generated_text", "") for o in outputs]
        return results if num_return_sequences > 1 else results[0]

    elif not hasattr(llm, "tokenizer"):
        # For API wrappers without tokenizer (e.g., School/OpenRouter wrappers).
        outputs = llm(
            message,
            max_new_tokens=max_new_token,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            n=num_return_sequences,
            enable_thinking=enable_thinking,
        )
        results = [o.get("generated_text", "") if isinstance(o, dict) else str(o) for o in outputs]
        return results if num_return_sequences > 1 else results[0]
    
    else:
        # For transformers pipelines, loop calls to get multiple outputs.
        eos_pair = _build_eos_ids(llm.tokenizer)
   
        prompt = llm.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        prompt_token_len = len(llm.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        max_length = prompt_token_len + max_new_token
        outputs = []
        for _ in range(num_return_sequences):
            output = llm(prompt,
                         max_length=max_length,
                         eos_token_id=eos_pair, 
                         do_sample=True, 
                         temperature=temperature, 
                         top_p=0.9, 
                         pad_token_id=llm.tokenizer.eos_token_id)
            outputs.append(output[0]["generated_text"][len(prompt):])
        
        return outputs if num_return_sequences > 1 else outputs[0]



def remove_extra_spaces(text):
    # Remove spaces before punctuation marks.
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    # Remove redundant spaces around brackets.
    text = re.sub(r'\(\s+', '(', text)  # Spaces right after '('
    text = re.sub(r'\s+\)', ')', text)  # Spaces right before ')'
    text = re.sub(r'\[\s+', '[', text)  # Spaces right after '['
    text = re.sub(r'\s+\]', ']', text)  # Spaces right before ']'
    text = re.sub(r'\{\s+', '{', text)  # Spaces right after '{'
    text = re.sub(r'\s+\}', '}', text)  # Spaces right before '}'
    # Collapse repeated spaces into a single space.
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_NQ(sample):
    # Some annotations may not contain long answers, and may also miss short answers.
    question = sample['question']['text']
    tokens = sample['document']['tokens']['token']
    is_html = sample['document']['tokens']['is_html']
    question_id = sample['id']

    # document = sample['document_tokens']
    answer_set = []
    annotation_list = [dict(zip(sample['annotations'].keys(), values))
                       for values in zip(*sample['annotations'].values())]
    # annotation_list contains all human annotations.
    for anno in annotation_list:
        # Extract long answer.
        if anno['long_answer']['start_token'] != -1:
            start = anno['long_answer']['start_token']
            end = anno['long_answer']['end_token']
            long_answer = remove_extra_spaces(' '.join([token for token, html_flag in zip(tokens[start:end], is_html[start:end]) if not html_flag]))
        else:
            long_answer = None

        answer_set.append(
            {'id': anno['id'],
             'long_answer': long_answer,
             'short_answer': anno['short_answers']['text']}
        )

    candidate_set = []
    candidate_list = [dict(zip(sample['long_answer_candidates'].keys(), values)) for values in zip(*sample['long_answer_candidates'].values())]
    for candi in candidate_list:
        text = remove_extra_spaces(' '.join([token for token, html_flag in zip(tokens[candi['start_token']:candi['end_token']], is_html[candi['start_token']:candi['end_token']]) if not html_flag]))
        candidate_set.append(
            {'text': text,
             'top_level': candi['top_level']
             }
        )
    document = remove_extra_spaces(' '.join([token for token, html_flag in zip(tokens, is_html) if not html_flag]))
    # documents is the raw document, without human filtering

    return question, answer_set, candidate_set, document, question_id