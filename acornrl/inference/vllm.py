import torch
import concurrent.futures
import re, os, sys, subprocess 
import time, requests, json, logging, os
from typing import Dict, Tuple, List, Optional, Any

import torch
import concurrent.futures
import logging
import os
import time
import requests
import subprocess
import sys
from typing import Optional, List, Dict, Any



STANDARD_GAME_PROMPT = "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format."


class VLLMInferenceClient:
    def __init__(
        self, 
        model_name: str,
        url: str = "http://localhost:8000",  # Changed to http
        max_length: int = 2048,
        standard_prompt: str = STANDARD_GAME_PROMPT,
        device: str = "cuda",
        **kwargs
    ):
        self.model_name = model_name
        self.url = url  # Consistent naming
        self.max_length = max_length
        self.standard_prompt = standard_prompt
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Test connection to vLLM server 
        try:
            health_url = f"{url}/health"
            response = requests.get(health_url)
            if response.status_code == 200:
                self.logger.info(f"Successfully connected to vLLM server at {url}")
            else:
                self.logger.warning(f"vLLM server at {url} responded with status code {response.status_code}")
        except Exception as e:
            self.logger.error(f"Failed to connect to vLLM server at {url}: {str(e)}")
        
    def format_observation(self, observation: str) -> str:
        begin_token = "<｜begin▁of▁sentence｜>"
        if not begin_token in observation:
            formatted_input = f"{begin_token}Please reason step by step, and put your final answer within [ ].\n{self.standard_prompt}\n<｜User｜>{observation}\n<｜Assistant｜><think>\n"
        else:
            # Observation already has formatting
            formatted_input = observation
        return formatted_input 

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> Dict:  # Updated return type
        formatted_input = self.format_observation(prompt)

        # Prepare request to vLLM
        url = f"{self.url}/generate"  # Fixed variable
        payload = {
            "prompt": formatted_input,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": [],
            "n": 1,
            "stream": False,
        }

        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in payload:  # Fixed typo
                payload[key] = value 


        return_dict = {
            "reasoning": None,
            "answer": None,
            "metrics": {}
        }

        # Make the API call
        try:
            start_time = time.time()
            response = requests.post(url, json=payload)
            generation_time = time.time() - start_time 

            if response.status_code == 200:
                data = response.json()
                generated_text = data["text"][0]

                # Get token count from response if available, otherwise estimate
                if "usage" in data:
                    # Some vLLM APIs return token counts directly
                    tokens_generated = data["usage"]["completion_tokens"]
                else:
                    # Rough estimation (can be replaced with a proper tokenizer)
                    tokens_generated = len(generated_text.split()) * 1.3  # Rough estimate
                

                # Extract reasoning and answer with regex
                # if no answer, return full response as answer
                reasoning, answer = self._extract_reasoning_and_answer(generated_text)
                return_dict["reasoning"] = reasoning 
                return_dict["answer"] = answer 
                return_dict["metrics"]["completion_tokens"] = tokens_generated
                return_dict["metrics"]["completion_time"] = generation_time

                self.logger.info(f"Generated response in {generation_time:.2f} seconds")
                
                return return_dict 
            else:
                self.logger.error(f"Request failed with status code {response.status_code}: {response.text}")
                return return_dict
        except Exception as e:
            self.logger.error(f"Error during text generation: {str(e)}")
            return return_dict



    def _extract_reasoning_and_answer(self, text: str) -> Tuple[str, str]:
        # Pattern to match content within <think>...</think> tags
        think_pattern = r'<think>(.*?)</think>'
        # Pattern to match content within <answer>...</answer> tags
        answer_pattern = r'<answer>(.*?)</answer>'
        # Pattern to match content within square brackets [...]
        bracket_pattern = r'\[(.*?)\]'
        
        # Search for reasoning within <think> tags
        think_match = re.search(think_pattern, text, re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else ""
        
        # Search for answer within <answer> tags
        answer_match = re.search(answer_pattern, text, re.DOTALL)
        
        if answer_match:
            # If <answer> tags are found, extract the content
            answer = answer_match.group(1).strip()
        elif think_match:
            # If </think> tag is present but no <answer> tag, return everything after </think>
            think_end_pos = text.find('</think>') + len('</think>')
            answer = text[think_end_pos:].strip()
        else:
            # If no <think> or <answer> tags, look for content in square brackets
            bracket_matches = re.findall(bracket_pattern, text, re.DOTALL)
            if bracket_matches:
                # If square brackets are found, take the last occurrence
                answer = bracket_matches[-1].strip()
            else:
                # If no square brackets either, return the full text as the answer
                answer = text.strip()
                            
        return reasoning, answer


class VLLMServerManager:
    def __init__(
        self,
        model_path: str,
        max_seq_len: int = 2048,
        gpus: Optional[List[int]] = None,
        tensor_parallel_size: int = 1,
        base_port: int = 8000,
        lora_modules: Optional[Dict[str, Any]] = None  # New parameter for LoRA modules
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.base_port = base_port 
        self.max_seq_len = max_seq_len
        self.lora_modules = lora_modules or {}

        # Use all GPUs if none are specified
        if gpus is None:
            self.gpus = list(range(torch.cuda.device_count()))
        else:
            self.gpus = gpus 

        self.num_servers = len(self.gpus) // tensor_parallel_size
        self.processes = [] 
        self.logger = logging.getLogger(__name__)

    def start_servers(self):
        """Start vLLM servers in parallel with LoRA support."""
        self.logger.info(f"Starting {self.num_servers} vLLM servers with tensor_parallel_size {self.tensor_parallel_size}")
        
        log_dir = os.path.join(os.getcwd())
        os.makedirs(log_dir, exist_ok=True)
        
        def start_single_server(i):
            gpu_ids = self.gpus[i*self.tensor_parallel_size:(i+1)*self.tensor_parallel_size]
            port = self.base_port + i
            
            # Set environment variables
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            
            # Create log file
            log_path = os.path.join(log_dir, f"vllm_server_gpu{gpu_ids[0]}_port{port}.log")
            log_file = open(log_path, "w")
            
            # Command to start the server
            # if not self.enable_lora:
            cmd = [
                sys.executable, "-m", "vllm.entrypoints.api_server",
                "--model", self.model_path,
                "--tensor-parallel-size", str(self.tensor_parallel_size),
                "--port", str(port),
                "--host", "0.0.0.0",
                "--gpu-memory-utilization", "0.9",
                "--max-model-len", str(self.max_seq_len),
                "--trust-remote-code",
                "--max-seq-len-to-capture", "48000"
            ]
            # else:
            #     cmd = [
            #         sys.executable, "-m", "vllm.entrypoints.api_server",
            #         "--model", "./merged_model",
            #         "--tensor-parallel-size", str(self.tensor_parallel_size),
            #         "--port", str(port),
            #         "--host", "0.0.0.0",
            #         "--gpu-memory-utilization", "0.9",
            #         "--max-model-len", str(self.max_seq_len),
            #         "--trust-remote-code",
            #         "--max-seq-len-to-capture", "48000"
            #     ]

            
            self.logger.info(f"Starting vLLM server on GPUs {gpu_ids} at port {port}")
            self.logger.info(f"Command: {' '.join(cmd)}")
            
            # Clear GPU cache before starting
            try:
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'memory_stats'):
                    self.logger.info(f"GPU {gpu_ids[0]} memory stats: {torch.cuda.memory_stats(device=gpu_ids[0])}")
            except Exception as e:
                self.logger.warning(f"Failed to clear CUDA cache: {e}")
            
            process = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file, text=True)
            
            # Wait for server to start
            if not self._verify_server(port):
                self.logger.warning(f"Server on port {port} failed initial startup check")
                with open(log_path, 'r') as f:
                    last_lines = f.readlines()[-50:]
                    self.logger.error(f"Last 50 lines of server log:\n{''.join(last_lines)}")
                    
            return (process, port, log_file, gpu_ids)

        # Use a thread pool to start all servers in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_servers) as executor:
            futures = [executor.submit(start_single_server, i) for i in range(self.num_servers)]
            for future in concurrent.futures.as_completed(futures):
                server_tuple = future.result()
                self.processes.append(server_tuple)



    def _verify_server(self, port: int, max_attempts: int = 12, timeout: int = 5) -> bool:
        """Verify a single server is running."""
        url = f"http://localhost:{port}/health"
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    self.logger.info(f"vLLM server on port {port} is ready")
                    return True
            except Exception:
                pass
            
            self.logger.info(f"Waiting for vLLM server on port {port}... ({attempt+1}/{max_attempts})")
            time.sleep(timeout)
        
        self.logger.error(f"vLLM server on port {port} failed to start")
        return False


    def get_client(self, server_idx: int = 0) -> VLLMInferenceClient:
        """Get a client for a specific server."""
        if not self.processes:
            raise RuntimeError("No vLLM servers are running")

        idx = server_idx % len(self.processes)
        # Extract only the port, ignore other elements
        _, port, *_ = self.processes[idx]

        return VLLMInferenceClient(model_name=self.model_path, url=f"http://localhost:{port}")


    def stop_servers(self):
        """Stop all vLLM servers."""
        self.logger.info(f"Processes list before stopping: {self.processes}")

        for process_tuple in self.processes:
            # Handle the new 4-element tuple structure
            process, port, log_file, _ = process_tuple
            
            self.logger.info(f"Stopping vLLM server on port {port}")
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Server on port {port} did not terminate, killing it")
                process.kill()
            except Exception as e:
                self.logger.error(f"Error stopping server on port {port}: {e}")
                
            # Close log file
            try:
                log_file.close()
            except:
                pass

        self.processes = []