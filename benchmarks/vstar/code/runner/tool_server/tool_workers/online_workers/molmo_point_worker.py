import torch
import numpy as np
from PIL import Image, ImageDraw
import base64
import uuid
import os
import traceback
import re
from io import BytesIO
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor


from transformers import HfArgumentParser, AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig, PreTrainedModel
from dataclasses import dataclass, field
from typing import Optional
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse

from tool_server.utils.server_utils import build_logger
from tool_server.utils.worker_arguments import WorkerArguments
from tool_server.utils.error_codes import *
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"molmo_point_worker_{worker_id}.log")

@dataclass
class MolmoPointArguments(WorkerArguments):
    max_length: Optional[int] = field(
        default=200,
        metadata={"help": "Maximum length for token generation"}
    )
    max_concurrency: Optional[int] = field(
        default=100,
        metadata={"help": "Maximum number of concurrent requests to process."}
    )

class MolmoPointWorker(BaseToolWorker):
    def __init__(self, worker_arguments: MolmoPointArguments = None):
        # 在调用父类初始化前先设置模型名称和并发数
        if worker_arguments and worker_arguments.model_name is None:
            worker_arguments.model_name = "Point"
            
        # 设置更高的并发处理能力
        if hasattr(worker_arguments, "max_concurrency"):
            worker_arguments.limit_model_concurrency = worker_arguments.max_concurrency
        else:
            worker_arguments.limit_model_concurrency = 10  # 默认允许10个并发请求
            
        super().__init__(worker_arguments)
            
        self.max_length = worker_arguments.max_length
        
        # 创建线程池用于并发处理请求
        self.thread_pool = ThreadPoolExecutor(max_workers=worker_arguments.limit_model_concurrency)
        
        self.instruction = {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Identify a point in the image based on a natural language description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "The identifier of the image to edit, e.g., 'img_1'"
                        },
                        "description": {
                            "type": "string",
                            "description": "A natural language description of the point of interest, e.g., 'the dog's nose', 'center of the clock', 'the tallest tree'."
                        }
                    },
                    "required": ["image", "description"]
                }
            }
        }

    def init_model(self):
        logger.info(f"Initializing model {self.model_name} with concurrency {self.args.limit_model_concurrency}...")
        logger.info(f"CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}")
        
        quant_config = None
        # 在worker_arguments中定义了load_4bit和load_8bit，可以在all_service_example.yaml中设置
        # 原本没修改的情况下，两个都是False
        if self.load_4bit or self.load_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=self.load_8bit,
                load_in_4bit=self.load_4bit,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
            logger.info(f"Using quantization config: {quant_config}")
        try:
            if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
                # New Transformers allocator warmup expects this mapping, but
                # the older remote-code Molmo class only defines
                # `_tied_weights_keys`. Bridge the two APIs locally.
                def get_all_tied_weights_keys(model):
                    explicit = model.__dict__.get("_compat_all_tied_weights_keys")
                    if explicit is not None:
                        return explicit
                    keys = getattr(model, "_tied_weights_keys", None) or []
                    if isinstance(keys, dict):
                        return keys
                    return {str(key): str(key) for key in keys}

                def set_all_tied_weights_keys(model, value):
                    model.__dict__["_compat_all_tied_weights_keys"] = value

                PreTrainedModel.all_tied_weights_keys = property(
                    get_all_tied_weights_keys,
                    set_all_tied_weights_keys,
                )

            # Accelerate's automatic balanced-memory inference fails for the
            # quantized Molmo checkpoint because its custom model currently
            # exposes an empty `_no_split_modules` list. This worker is a
            # single-device service, so an explicit root-module placement is
            # both sufficient and avoids that incompatible auto path.
            model_device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}

            # load the processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype='auto'
            )

            model_load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": "auto",
                "device_map": model_device_map,
            }
            # Do not pass an explicit None: older Transformers releases then
            # overwrite a pre-quantized checkpoint's embedded config with
            # None and fail while serializing it.
            if quant_config is not None:
                model_load_kwargs["quantization_config"] = quant_config

            # load the model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_load_kwargs,
            )
            self._disable_dynamic_cache(self.model)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            logger.error(traceback.format_exc())
            raise

        
    def extract_points(self, molmo_output, image_w, image_h):
        all_points = []
        for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', molmo_output):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # Treat as an invalid output
                    continue
                point /= 100.0
                point = point * np.array([image_w, image_h])
                all_points.append(point)
        return np.array(all_points)  # Ensure it's always a NumPy array
    
    def show_points(self, coords, labels, ax, marker_size=375):
        # Only plot if there are points
        if len(coords) == 0:
            return
        
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        
        if len(pos_points) > 0:
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        if len(neg_points) > 0:
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    
    def create_image_with_points(self, image, coords, labels, marker_size=375):
        annotated = image.convert("RGB").copy()
        draw = ImageDraw.Draw(annotated)
        radius = max(4, int(np.sqrt(marker_size) / 2))
        outline_width = max(1, radius // 3)
        for coord, label in zip(coords, labels):
            x, y = (float(coord[0]), float(coord[1]))
            color = "green" if int(label) == 1 else "red"
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline="white",
                width=outline_width,
            )
        return annotated
        
    def generate_gate(self, params):
        """Run inference directly; the base HTTP route already owns thread dispatch."""
        return self._generate_with_torch_inference(params)

    @torch.inference_mode()
    def _generate_with_torch_inference(self, params):
        """实际的生成函数，包装在torch.inference_mode装饰器中"""
        return self._generate_impl(params)

    def _generation_config(self):
        """Build the generation settings required by Molmo's batch API."""
        return GenerationConfig(
            max_new_tokens=self.max_length,
            stop_strings="<|endoftext|>",
            use_cache=True,
        )

    @staticmethod
    def _disable_dynamic_cache(model):
        """Keep legacy tuple caches for Molmo remote code on new Transformers."""
        model.__class__._supports_default_dynamic_cache = classmethod(
            lambda cls: False
        )
        
    def _generate_impl(self, params):
        """实现实际的生成逻辑"""
        tool_reward = 2.0
        # 计算Parameter Name Matching
        param_keys = set(params.keys())
        required_keys = set(self.instruction["function"]["parameters"]["required"])
        parameter_name_match_reward = len(param_keys & required_keys) / len(required_keys | param_keys)
        tool_reward = tool_reward + parameter_name_match_reward
        # 参数名称没有完全匹配，直接返回
        if parameter_name_match_reward < 1:
            return {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": "Invalid parameters: expected keys: image, description.",
                "error_code": INVALID_PARAMETERS,
                "tool_reward": tool_reward
            }
        
        required_keys_num = len(required_keys)
        # 初始化参数合规计数器
        correct_param_content_num = 0
        
        # Extract inputs
        image_data = params["image"]
        description = params.get("description")
        
        try:
            # Convert base64 to PIL image
            try:
                image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
                correct_param_content_num += 1
                
                # 成功加载图像后，创建图像尺寸信息
                image_dimensions = {
                    "width": image.width,
                    "height": image.height
                }
            except Exception as e:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Failed to load image: {str(e)}",
                    "error_code": CANNOT_LOAD_IMAGE,
                    "tool_reward": tool_reward+correct_param_content_num/required_keys_num
                }
                return pred_dict
            
            # description参数验证通过
            if description and len(description.strip()) > 0:
                correct_param_content_num += 1
            else:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": "Invalid parameters: expected keys: image, description.",
                    "error_code": INVALID_PARAMETERS,
                    "tool_reward": tool_reward+correct_param_content_num/required_keys_num,
                    "image_dimensions_pixels": image_dimensions
                }
                return pred_dict
            
            text_prompt = f"Point to the {description} in the scene."
            
            # Process inputs and run model
            try:
                with torch.no_grad():
                    inputs = self.processor.process(
                        images=[image],
                        text=text_prompt,
                    )
                    inputs["images"] = inputs["images"].to(torch.bfloat16)
                    inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
                    
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        output = self.model.generate_from_batch(
                            inputs,
                            self._generation_config(),
                            tokenizer=self.processor.tokenizer
                        )
                        
                        # Only get generated tokens and decode them to text
                        generated_tokens = output[0, inputs['input_ids'].size(1):]
                        response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        logger.info("Point raw model response: %r", response)
            except Exception as e:
                logger.error("Point model inference failed: %r", e)
                logger.error(traceback.format_exc())
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": f"Model inference failed: {type(e).__name__}: {e!r}",
                    "error_code": TOOL_RUN_FAILED,
                    "tool_reward": tool_reward+correct_param_content_num/required_keys_num,
                    "image_dimensions_pixels": image_dimensions
                }
                return pred_dict
            
            # Extract points from the response
            if response:
                points = self.extract_points(response, image.width, image.height)
                
                # Create point representations
                point_data = []
                if len(points) > 0:
                    for point in points:
                        point_data.append({
                            "x": point[0],
                            "y": point[1],
                        })
                
                # 创建带有标记点的图片
                # 在create_image_with_points中，coords应该是点的数组，labels决定点的颜色
                # 这里我们默认所有点都是正向的（绿色），因为我们没有负向标签信息
                if len(points) > 0:
                    labels = np.ones(len(points), dtype=np.int32)  # 设置所有点为正向 (1)
                    image_with_points = self.create_image_with_points(image, points, labels)
                    
                    # 将图片转换为base64字符串
                    buffered = BytesIO()
                    image_format = image.format if image.format else 'PNG'
                    image_with_points.save(buffered, format=image_format)
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    pred_dict = {
                        "tool_response_from": self.model_name,
                        "status": "success",
                        "points": point_data,
                        "edited_image": img_str,
                        "image_dimensions_pixels": image_dimensions,
                        "error_code": SUCCESS,
                        "tool_reward": tool_reward+correct_param_content_num/required_keys_num
                    }
                else:
                    pred_dict = {
                        "tool_response_from": self.model_name,
                        "status": "success",
                        "points": point_data,
                        "message": "No valid points detected in the response.",
                        "image_dimensions_pixels": image_dimensions,
                        "error_code": SUCCESS,
                        "tool_reward": tool_reward+correct_param_content_num/required_keys_num
                    }
                
                return pred_dict
            else:
                pred_dict = {
                    "tool_response_from": self.model_name,
                    "status": "failed",
                    "message": "Model did not generate any response.",
                    "error_code": TOOL_RUN_FAILED,
                    "tool_reward": tool_reward+correct_param_content_num/required_keys_num,
                    "image_dimensions_pixels": image_dimensions
                }
                return pred_dict
                
        except Exception as e:
            # 这里不添加image_dimensions_pixels，因为可能在处理图像前就发生了异常
            pred_dict = {
                "tool_response_from": self.model_name,
                "status": "failed",
                "message": f"Error: {str(e)}\n Traceback:{traceback.format_exc()}\n",
                "error_code": TOOL_RUN_FAILED,
                "tool_reward": tool_reward+(correct_param_content_num/required_keys_num if required_keys_num > 0 else 0)
            }
            logger.error(f"Error during Molmo Point inference: {e}")
            logger.error(traceback.format_exc())
            return pred_dict
    
    # 使用没有torch.inference_mode装饰器的版本作为generate
    def generate(self, params):
        """为了与base_tool_worker兼容，提供一个generate方法"""
        return self._generate_impl(params)
    
    def get_tool_instruction(self):
        return self.instruction  
        
    def setup_routes(self):
        """覆盖父类方法，添加新的API端点用于并发处理多个请求"""
        super().setup_routes()  # 调用父类的路由设置
        
        @self.app.post("/worker_generate_batch")
        async def api_generate_batch(request: Request):
            """批量处理多个请求的API端点"""
            batch_params = await request.json()
            if not isinstance(batch_params, list):
                return JSONResponse({"status": "failed", "message": "Expected a list of parameter objects"})
            
            results = []
            futures = []
            
            # 提交所有任务到线程池
            for params in batch_params:
                future = self.thread_pool.submit(self._generate_with_torch_inference, params)
                futures.append(future)
            
            # 收集所有结果
            for future in futures:
                try:
                    result = future.result(timeout=12000)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing batch item: {e}")
                    results.append({
                        "tool_response_from": self.model_name,
                        "status": "failed",
                        "message": f"Processing error: {str(e)}",
                        "error_code": TOOL_RUN_FAILED
                    })
            
            return JSONResponse(results)

    def __del__(self):
        """析构函数，确保线程池正确关闭"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)

if __name__ == "__main__":
    parser = HfArgumentParser((MolmoPointArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    logger.info(f"args: {args}")

    worker = MolmoPointWorker(
        worker_arguments=args
    )
    worker.run() 
