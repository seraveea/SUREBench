from datasets import load_dataset
import torch
import json
import argparse
from tqdm import tqdm
from utils.utils import agent_reply, extract_NQ
from transformers import pipeline
import os
# import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, PreTrainedTokenizerBase
from utils.llm_utils import QwenPipeline, GemmaPipeline, QwenVLLMPipeline
from path_config import data_path, dataset_cache_dir, get_model_root, model_path
try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None

import warnings
from transformers import logging as transformers_logging

# Suppress duplicated transformers max_length/max_new_tokens warnings.
transformers_logging.set_verbosity_error()
warnings.filterwarnings('ignore', message='.*max_new_tokens.*max_length.*')


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")



if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    @property
    def _all_special_tokens_extended(self):
        return self.all_special_tokens

    PreTrainedTokenizerBase.all_special_tokens_extended = _all_special_tokens_extended


def _sanitize_vllm_rope_scaling(model_path: str) -> None:
    """Normalize rope_scaling schema for older vLLM compatibility."""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return

    changed = False

    # 1) Top-level rope_scaling: add rope_type if only type exists.
    rope_scaling = cfg.get("rope_scaling")
    if isinstance(rope_scaling, dict) and "rope_type" not in rope_scaling and "type" in rope_scaling:
        rope_scaling["rope_type"] = rope_scaling["type"]
        cfg["rope_scaling"] = rope_scaling
        changed = True

    # 2) Gemma4 often stores nested rope_scaling in text_config (full/sliding attention).
    # Older vLLM expects a flat dict with rope_type at the first level.
    text_cfg = cfg.get("text_config")
    if isinstance(text_cfg, dict):
        text_rope = text_cfg.get("rope_scaling")
        if isinstance(text_rope, dict) and "rope_type" not in text_rope:
            candidate = None
            if isinstance(text_rope.get("sliding_attention"), dict):
                candidate = dict(text_rope["sliding_attention"])
            elif isinstance(text_rope.get("full_attention"), dict):
                candidate = dict(text_rope["full_attention"])
            else:
                for v in text_rope.values():
                    if isinstance(v, dict):
                        candidate = dict(v)
                        break

            if isinstance(candidate, dict):
                if "rope_type" not in candidate:
                    if "type" in candidate:
                        candidate["rope_type"] = candidate["type"]
                    else:
                        candidate["rope_type"] = "default"
                if "type" not in candidate:
                    candidate["type"] = candidate["rope_type"]
                text_cfg["rope_scaling"] = candidate
                cfg["text_config"] = text_cfg
                changed = True

    if changed:
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            print(f"[vllm] patched rope_scaling schema in {config_path}")
        except Exception:
            return


class VLLMTextGenerationWrapper:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        trust_remote_code: bool = False,
    ):
        if LLM is None or SamplingParams is None:
            raise ImportError("vLLM is required for VLLMTextGenerationWrapper. Please install vllm in this environment.")
        self.model_path = model_path
        _sanitize_vllm_rope_scaling(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        llm_kwargs = dict(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            disable_log_stats=True,
        )
        try:
            self.llm = LLM(**llm_kwargs)
        except Exception as e:
            err_msg = str(e)
            if "rope_scaling should have a 'rope_type' key" not in err_msg:
                raise

            # Fallback for vLLM/transformers config mismatch on some Gemma configs.
            override_cfg = {"rope_scaling": {"rope_type": "default", "type": "default"}}
            llm_kwargs_hf = dict(llm_kwargs)
            llm_kwargs_hf["hf_overrides"] = override_cfg
            try:
                print("[vllm] retry with hf_overrides rope_scaling")
                self.llm = LLM(**llm_kwargs_hf)
            except TypeError as e2:
                if "hf_overrides" in str(e2):
                    raise RuntimeError(
                        "Current vLLM version does not support rope_scaling or hf_overrides overrides. "
                        "Please upgrade vLLM, or patch model config.json to include rope_scaling.rope_type."
                    ) from e2
                raise

    def _build_prompt(self, prompt):
        # Support the current input format: [{"role": ..., "content": ...}, ...].
        if isinstance(prompt, list):
            return self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def __call__(
        self,
        prompt,
        max_new_tokens=256,
        temperature=1,
        top_p=0.9,
        do_sample=True,
        n=1,
        **kwargs,
    ):
        text_prompt = self._build_prompt(prompt)
        # vLLM generation is controlled via SamplingParams.
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p,
            n=n,
        )

        outputs = self.llm.generate([text_prompt], sampling_params, use_tqdm=False)
        texts = [o.text for o in outputs[0].outputs]

        # Mimic transformers pipeline return format.
        # vLLM returns only newly generated text, so prepend the prompt for compatibility.
        return [{"generated_text": text_prompt + t} for t in texts]


def get_questions(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


os.environ["HF_HOME"] = get_model_root()
huggingface_cache_path = get_model_root()
os.environ['HF_HUB_OFFLINE'] = '1'



def _get_pipeline_tokenizer(gen_pipeline):
    return getattr(gen_pipeline, "tokenizer", None)


def truncate_input_text(text: str, gen_pipeline, max_input_tokens: int, sample_tag: str = "") -> str:
    """Truncate long input text before generation to avoid context overflow."""
    if not isinstance(text, str) or max_input_tokens <= 0:
        return text

    tokenizer = _get_pipeline_tokenizer(gen_pipeline)
    if tokenizer is None:
        # Fallback when tokenizer is unavailable (e.g., pure API wrappers).
        approx_char_limit = max_input_tokens * 4
        if len(text) <= approx_char_limit:
            return text
        print(f"[truncate] {sample_tag} chars {len(text)} -> {approx_char_limit} (approx)")
        return text[:approx_char_limit]

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_len = len(token_ids)
    if token_len <= max_input_tokens:
        return text

    truncated_text = tokenizer.decode(token_ids[:max_input_tokens], skip_special_tokens=False)
    print(f"[truncate] {sample_tag} tokens {token_len} -> {max_input_tokens}")
    return truncated_text


def pipeline_instance(args):
    if args.model == 'qwen3-1.7b':
        model_dir = model_path('Qwen3-1.7B', args=args)
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map=args.device
        )
        qwen_pipeline = QwenPipeline(model, tokenizer)
        return qwen_pipeline
    elif args.model == 'qwen3-8b':
        model_dir = model_path('Qwen3-8B', args=args)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map=args.device
        )
        qwen_pipeline = QwenPipeline(model, tokenizer)
        return qwen_pipeline
    elif args.model == 'qwen3-8b-vllm':
        model_path_str = model_path('Qwen3-8B', args=args)
        # Use vLLM for Qwen3-8B with enable_thinking support
        tokenizer = AutoTokenizer.from_pretrained(
            model_path_str,
            local_files_only=True,
            trust_remote_code=True,
        )
        llm = LLM(
            model=model_path_str,
            tokenizer=model_path_str,
            tensor_parallel_size=getattr(args, "tp_size", 1),
            gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.9),
            max_model_len=getattr(args, "max_model_len", None),
            trust_remote_code=True,
            disable_log_stats=True,
        )
        qwen_vllm_pipeline = QwenVLLMPipeline(llm, tokenizer)
        return qwen_vllm_pipeline
    elif args.model == 'llama-70b':
        model_dir = model_path('llama-3-70B-Instruct', args=args)
        gen_pipeline = pipeline(
            "text-generation",
            model=model_dir,
            device_map=args.device,
            torch_dtype=torch.float16
        )
        return gen_pipeline
    elif args.model == 'llama-405b':
        model_path_str = model_path('models--meta-llama--Llama-3.1-405B-Instruct-FP8/snapshots/64a54b704768dfd589a3e4ac05d546052f67f4fd', args=args)
        gen_pipeline = VLLMTextGenerationWrapper(
            model_path=model_path_str,
            tensor_parallel_size=getattr(args, "tp_size", 8),
            gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.9),
            max_model_len=getattr(args, "max_model_len", None),
            trust_remote_code=False,
        )
        return gen_pipeline
    elif args.model == 'deepseek-r1-8b':
        model_path_str = model_path('deepseek-r1-distill-llama-8B', args=args)
        gen_pipeline = VLLMTextGenerationWrapper(
            model_path=model_path_str,
            tensor_parallel_size=getattr(args, "tp_size", 1),
            gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.9),
            max_model_len=getattr(args, "max_model_len", None),
            trust_remote_code=False,
        )
        return gen_pipeline
    elif args.model == 'qwen3-235b':
        model_path_str = model_path('models--Qwen--Qwen3-235B-A22B-FP8/snapshots/39eb2b067ea6b8e3e1dd97d3cd0c7ffeaf3e1a35', args=args)
        # Use vLLM for better FP8 support and memory optimization
        gen_pipeline = VLLMTextGenerationWrapper(
            model_path=model_path_str,
            tensor_parallel_size=max(4, getattr(args, "tp_size", 4)),
            gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.9),
            max_model_len=getattr(args, "max_model_len", None),
            trust_remote_code=True,
        )
        return gen_pipeline
    else:
        model_dir = model_path('Meta-Llama-3.1-8B-Instruct_hf', args=args)
        gen_pipeline = pipeline(
            "text-generation",
            model=model_dir,
            device_map=args.device,
            torch_dtype=torch.float16
            )
        return gen_pipeline


def trivia_pipeline(args):
    """Pipeline for the TriviaQA dataset.""" 
    question_set = get_questions(args.question_file)['Data']
    # First LLM instance created here
    qa_pipeline = pipeline_instance(args)  
    answers = []

    for question_instance in tqdm(question_set):
        query = question_instance['Question']
        qid = question_instance['QuestionId']
        # Get all repeated samples in one call.
        current_reply = agent_reply(qa_pipeline, [{"role":"system","content":"Output a short answer with minimum words"},{"role": "user", "content": query}], 
                                    temperature=args.temperature,
                                    enable_thinking=args.enable_thinking,
                                    num_return_sequences=args.repeat,
                                    max_new_token=args.max_new_tokens)
        current_question = {
                'qid':qid,
                'question': query,
                'answers': current_reply
        }
        answers.append(current_question)

    output = {
        "generaion_args": vars(args),
        "answers": answers
    }
    json.dump(output, open(args.result_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'结果已保存到 {args.result_file}')
    return None


def nq_pipeline(args):
    """Pipeline for the Natural Questions dataset."""
    nq_data = load_dataset(
        "google-research-datasets/natural_questions",
        "dev",
        cache_dir=dataset_cache_dir("NQ", args=args),
    )
    qa_pipeline = pipeline_instance(args)
    
    answers = []
    for question_instance in tqdm(nq_data['validation']):
        query, answer_set, candidate_set, document, qid = extract_NQ(question_instance)
        # Get all repeated samples in one call.
        current_reply = agent_reply(
            qa_pipeline, 
            [{"role":"system","content":"Output a short answer with minimum words"},{"role": "user", "content": query}], 
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
            num_return_sequences=args.repeat,
            max_new_token=args.max_new_tokens)
        
        current_question = {
            'qid':qid,
            'question': query,
            'answers': current_reply,
        }
        answers.append(current_question)
    
    output = {
        "generaion_args": vars(args),
        "answers": answers
    }
    json.dump(output, open(args.result_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'结果已保存到 {args.result_file}')
    return None


def coqa_pipeline(args):
    question_set = get_questions(args.question_file)['data']
    qa_pipeline = pipeline_instance(args)
    answers = []
    for story_instance in tqdm(question_set):
        story_id  = story_instance['id']
        story = story_instance['story']
        for question in story_instance['questions']:
            question_text = question['input_text']
            question_id = question['turn_id']
            # Get all repeated samples in one call.
            current_reply = agent_reply(
                qa_pipeline, 
                [{"role":"system","content":"Output a short answer with minimum words"},{"role": "user", "content": f'story: {story}\n question: {question_text}'}], 
                temperature=args.temperature,
                enable_thinking=args.enable_thinking,
                num_return_sequences=args.repeat,
                max_new_token=args.max_new_tokens)
            current_question = {
                'qid': f"{story_id}/{str(question_id)}",
                'question': question_text,
                'story_id': story_id,
                'turn_id': question_id,
                'answers': current_reply
            }
            answers.append(current_question)
    output = {
        "generaion_args": vars(args),
        "answers": answers 
    }
    json.dump(output, open(args.result_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'结果已保存到 {args.result_file}')
    return None


def squad_pipeline(args):
    question_set = get_questions(args.question_file)
    qa_pipeline = pipeline_instance(args)
    answers = []
    
    for article in tqdm(question_set['data']):
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question_id = qa['id']
                question_text = qa['question']
                
                # Get all repeated samples in one call.
                current_reply = agent_reply(qa_pipeline, [
                    {"role": "system", "content": "Output a short answer with minimum words"},
                    {"role": "user", "content": f"Context: {context}\nQuestion: {question_text}"}
                ], temperature=args.temperature,
                enable_thinking=args.enable_thinking,
                num_return_sequences=args.repeat,
                max_new_token=args.max_new_tokens)
                
                current_question = {
                    'qid': question_id,
                    'question': question_text,
                    'answers': current_reply
                }
                answers.append(current_question)
    
    output = {
        "generaion_args": vars(args),
        "answers": answers
    }
    json.dump(output, open(args.result_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'结果已保存到 {args.result_file}')
    return None


def BioASQ_pipeline(args):
    """Pipeline for the BioASQ dataset."""
    question_path = [
        data_path('BioASQ_Task12BGE', '12B1_golden.json', args=args),
        data_path('BioASQ_Task12BGE', '12B2_golden.json', args=args),
        data_path('BioASQ_Task12BGE', '12B3_golden.json', args=args),
        data_path('BioASQ_Task12BGE', '12B4_golden.json', args=args),
    ]
    question_set = []
    for path in question_path:
        question_set.extend(get_questions(path)['questions'])
    qa_pipeline = pipeline_instance(args)
    answers = []

    for question_instance in tqdm(question_set):
        query = question_instance['body']
        qid = question_instance['id']
        # Get all repeated samples in one call.
        current_reply = agent_reply(
            qa_pipeline, 
            [{"role":"system","content":"Output a short answer with minimum words"},{"role": "user", "content": query}], 
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
            num_return_sequences=args.repeat,
            max_new_token=args.max_new_tokens)
        
        current_question = {
                'qid':qid,
                'question': query,
                'answers': current_reply
        }
        answers.append(current_question)

    output = {
        "generaion_args": vars(args),
        "answers": answers
    }

    json.dump(output, open(args.result_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'结果已保存到 {args.result_file}')
    return None


def SVAMP_pipeline(args):
    # SVAMP reference answers are numeric, so use a numeric-output prompt.
    question_set = get_questions(args.question_file)
    # First LLM instance created here
    qa_pipeline = pipeline_instance(args)  
    answers = []
    for question_instance in tqdm(question_set):
        query = question_instance['Body'] + '\n' + question_instance['Question']
        qid = question_instance['ID']
        # Get all repeated samples in one call.
        current_reply = agent_reply(
            qa_pipeline, 
            [{"role":"system","content":"Only output a number answer"},{"role": "user", "content": query}], 
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
            num_return_sequences=args.repeat, 
            max_new_token=args.max_new_tokens)
        
        current_question = {
                'qid':qid,
                'question': query,
                'answers': current_reply
        }
        answers.append(current_question)

    output = {
        "generaion_args": vars(args),
        "answers": answers
    }
    json.dump(output, open(args.result_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'结果已保存到 {args.result_file}')
    return None


def gsm8k_pipeline(args):
    """Pipeline for the GSM8K dataset."""
    # Load GSM8K from Hugging Face.
    gsm8k_data = load_dataset("gsm8k", "main", cache_dir=dataset_cache_dir("gsm8k", args=args))
    qa_pipeline = pipeline_instance(args)
    
    # Use test split.
    test_data = gsm8k_data['test']
    
    answers = []
    for idx, question_instance in enumerate(tqdm(test_data)):
        query = question_instance['question']
        qid = str(idx)
        # Get all repeated samples in one call.
        current_reply = agent_reply(
            qa_pipeline, 
            [{"role":"system","content":"Solve this math problem step by step and provide the final numerical answer after ####."},
             {"role": "user", "content": query}], 
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
            num_return_sequences=args.repeat,            
            max_new_token=args.max_new_tokens)
        
        current_question = {
            'qid': qid,
            'question': query,
            'answers': current_reply
        }
        answers.append(current_question)

    
    output = {
        "generaion_args": vars(args),
        "answers": answers
    }
    json.dump(output, open(args.result_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'结果已保存到 {args.result_file}')
    return None


def xsum_pipeline(args):
    """Pipeline for the XSUM dataset."""
    xsum_data = load_dataset("xsum", cache_dir=dataset_cache_dir("xsum", args=args))
    qa_pipeline = pipeline_instance(args)

    # Use test split.
    test_data = xsum_data['test']

    answers = []
    for idx, sample in enumerate(tqdm(test_data)):
        document = sample['document']
        qid = sample.get('id', str(idx))
        document = truncate_input_text(document, qa_pipeline, args.max_input_tokens, sample_tag=f"xsum:{qid}")
 
        # Get all repeated samples in one call.
        current_reply = agent_reply(
            qa_pipeline,
            [
                {"role": "system", "content": "Summarize the given article in one short sentence."},
                {"role": "user", "content": document}
            ],
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
            num_return_sequences=args.repeat,
            max_new_token=args.max_new_tokens
        )

        current_question = {
            'qid': qid,
            'question': document,
            'answers': current_reply
        }
        answers.append(current_question)

    output = {
        "generaion_args": vars(args),
        "answers": answers
    }
    json.dump(output, open(args.result_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'结果已保存到 {args.result_file}')
    return None


def aeslc_pipeline(args):
    """Pipeline for the AESLC dataset."""
    aeslc_data = load_dataset("aeslc", cache_dir=dataset_cache_dir("aeslc", args=args))
    qa_pipeline = pipeline_instance(args)

    # Use test split.
    test_data = aeslc_data['test']

    answers = []
    for idx, sample in enumerate(tqdm(test_data)):
        email_body = sample.get('email_body', '')
        qid = sample.get('id', str(idx))
        email_body = truncate_input_text(email_body, qa_pipeline, args.max_input_tokens, sample_tag=f"aeslc:{qid}")

        # Get all repeated samples in one call.
        current_reply = agent_reply(
            qa_pipeline,
            [
                {"role": "system", "content": "Generate a concise email subject line."},
                {"role": "user", "content": email_body}
            ],
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
            num_return_sequences=args.repeat,            
            max_new_token=args.max_new_tokens
        )

        current_question = {
            'qid': qid,
            'question': email_body,
            'answers': current_reply
        }
        answers.append(current_question)

    output = {
        "generaion_args": vars(args),
        "answers": answers
    }
    json.dump(output, open(args.result_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'结果已保存到 {args.result_file}')
    return None


def cnn_dailymail_pipeline(args):
    """Pipeline for the CNN/DailyMail dataset."""
    cnndm_data = load_dataset("cnn_dailymail", "3.0.0", cache_dir=dataset_cache_dir("cnn_dailymail", args=args))
    qa_pipeline = pipeline_instance(args)

    # Use test split.
    test_data = cnndm_data['test']

    answers = []
    for idx, sample in enumerate(tqdm(test_data)):
        article = sample.get('article', '')
        qid = sample.get('id', str(idx))
        article = truncate_input_text(article, qa_pipeline, args.max_input_tokens, sample_tag=f"cnn_dailymail:{qid}")
 
        # Get all repeated samples in one call.
        current_reply = agent_reply(
            qa_pipeline,
            [
                {"role": "system", "content": "Summarize the given news article in 2-3 concise sentences."},
                {"role": "user", "content": article}
            ],
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
            num_return_sequences=args.repeat,
            max_new_token=args.max_new_tokens
        )

        current_question = {
            'qid': qid,
            'question': article,
            'answers': current_reply
        }
        answers.append(current_question)

    output = {
        "generaion_args": vars(args),
        "answers": answers
    }
    json.dump(output, open(args.result_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'结果已保存到 {args.result_file}')
    return None


def multi_news_pipeline(args):
    """Pipeline for the MultiNews dataset."""

    multi_news_data = load_dataset("Awesome075/multi_news_parquet", cache_dir=dataset_cache_dir("multi_news", args=args))
    qa_pipeline = pipeline_instance(args)

    # Use test split.
    test_data = multi_news_data['test']

    answers = []
    for idx, sample in enumerate(tqdm(test_data)):
        document = sample.get('document', '')
        qid = str(idx)
        document = truncate_input_text(document, qa_pipeline, args.max_input_tokens, sample_tag=f"multi_news:{qid}")

        # Get all repeated samples in one call.
        current_reply = agent_reply(
            qa_pipeline,
            [
                {"role": "system", "content": "Summarize the following multi-document news in 2-3 concise sentences."},
                {"role": "user", "content": document}
            ],
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
            num_return_sequences=args.repeat,
            max_new_token=args.max_new_tokens
        )

        current_question = {
            'qid': qid,
            'question': document,
            'answers': current_reply
        }
        answers.append(current_question)

    output = {
        "generaion_args": vars(args),
        "answers": answers
    }
    json.dump(output, open(args.result_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'结果已保存到 {args.result_file}')
    return None


def wmt19_pipeline(args):
    """Pipeline for WMT19 (validation split only)."""
    if '-' not in args.wmt19_subset:
        raise ValueError("wmt19_subset 格式应为 src-tgt，例如 zh-en")
    source_lang, target_lang = args.wmt19_subset.split('-', 1)
    wmt19_validation = load_dataset(
        "wmt19",
        args.wmt19_subset,
        split="validation",
        cache_dir=dataset_cache_dir("wmt19", args=args)
    )
    qa_pipeline = pipeline_instance(args)

    answers = []
    for idx, sample in enumerate(tqdm(wmt19_validation)):
        translation = sample.get('translation', {})
        source_text = translation.get(source_lang, '')
        qid = str(idx)
        # Get all repeated samples in one call.
        current_reply = agent_reply(
            qa_pipeline,
            [
                {"role": "system", "content": f"Translate from {source_lang} to {target_lang}. Output only the translated text."},
                {"role": "user", "content": source_text}
            ],
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
            num_return_sequences=args.repeat,
            max_new_token=args.max_new_tokens
        )

        current_question = {
            'qid': qid,
            'question': source_text,
            'answers': current_reply
        }
        answers.append(current_question)

    output = {
        "generaion_args": vars(args),
        "answers": answers
    }
    json.dump(output, open(args.result_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'结果已保存到 {args.result_file}')
    return None



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='', help='Dataset root directory. If empty, use SUREBENCH_DATA_ROOT or ./data')
    parser.add_argument('--model_root', default='', help='Model root directory. If empty, use SUREBENCH_MODEL_ROOT or ./models')
    parser.add_argument('--model', default='qwen3-235b', 
                        help="""选择模型: gemma-4-31b-it, gemma-4-31b-it-vllm, llama-8b, llama-70b, llama-405b, qwen3-235b, qwen3-8b, qwen3-32b, m25""")
    parser.add_argument('--dataset', default='triviaqa', 
                        help="""triviaqa, nq, coqa, squad, bioasq, svamp, gsm8k, xsum, aeslc, cnn_dailymail, multi_news, wmt19""")
    parser.add_argument('--enable_thinking', default=False, help='是否启用思考过程')
    parser.add_argument('--question_file', default='', help='Path to question json; if empty, inferred from --dataset and --data_root')
    parser.add_argument('--temperature', type=float, default=1.0, help='生成答案的温度')
    parser.add_argument('--result_file', default='output/triviaqa/triviaqa_qwen3-235b.json', help='结果保存路径')
    parser.add_argument('--wmt19_subset', default='cs-en', help='WMT19子集，如 cs-en, de-en, fi-en, gu-en, kk-en, lt-en, ru-en, zh-en')
    parser.add_argument('--device', default="auto")
    parser.add_argument('--repeat', type=int, default=10, help='每个问题需要回答的次数')
    parser.add_argument('--max_input_tokens', type=int, default=131000, help='截断输入文本到最大token数；0表示不截断')
    parser.add_argument('--tp_size', type=int, default=1, help='vLLM tensor parallel size，仅对vLLM分支生效')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.6, help='vLLM GPU显存利用率，仅对vLLM分支生效')
    parser.add_argument('--max_model_len', type=int, default=None, help='vLLM最大上下文长度，仅对vLLM分支生效')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='生成答案的最大token数')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.dataset == 'nq':
        nq_pipeline(args)
    elif args.dataset == 'triviaqa':
        if not args.question_file:
            args.question_file = data_path('triviaqa-rc', 'qa', 'wikipedia-dev.json', args=args)
        trivia_pipeline(args)
    elif args.dataset == 'coqa':
        if not args.question_file:
            args.question_file = data_path('coqa', 'coqa-dev-v1.0.json', args=args)
        coqa_pipeline(args)
    elif args.dataset == 'squad':
        if not args.question_file:
            args.question_file = data_path('SQuAD', 'dev-v2.0.json', args=args)
        squad_pipeline(args)
    elif args.dataset == 'bioasq':
        BioASQ_pipeline(args)
    elif args.dataset == 'svamp':
        if not args.question_file:
            args.question_file = data_path('SVAMP', 'test.json', args=args)
        SVAMP_pipeline(args)
    elif args.dataset == 'gsm8k':
        gsm8k_pipeline(args)
    elif args.dataset == 'xsum':
        xsum_pipeline(args)
    elif args.dataset == 'aeslc':
        aeslc_pipeline(args)
    elif args.dataset == 'cnn_dailymail':
        cnn_dailymail_pipeline(args)
    elif args.dataset == 'multi_news':
        multi_news_pipeline(args)
    elif args.dataset == 'wmt19':
        wmt19_pipeline(args)
    else:
        print('please specify a valid dataset: nq, triviaqa, or coqa')
