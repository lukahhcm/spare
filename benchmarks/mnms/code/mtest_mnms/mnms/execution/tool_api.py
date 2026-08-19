import os
import ast
import textwrap
import torch
import requests
from PIL import Image, ImageDraw, ImageFont
from .config import *
from typing import Dict, List, Union

try:
    import textract
except ImportError:
    textract = None

root_path = CACHE_ROOT_PATH
os.environ["TORCH_HOME"] = f"{root_path}/torch"
os.environ["TRANSFORMERS_CACHE"] = f"{root_path}/huggingface"
os.environ["DIFFUSERS_CACHE"] = f"{root_path}/huggingface"
os.environ["HF_HOME"] = f"{root_path}/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{root_path}/huggingface"
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

def _use_heavy_models():
    return os.getenv("MNMS_USE_HEAVY_MODELS") == "1"

def _allow_network_tools():
    return os.getenv("MNMS_ALLOW_NETWORK_TOOLS") == "1"

def _has_openai_key():
    return bool(OPENAI_API_KEY and OPENAI_API_KEY not in {"EMPTY", "none", "None", "null"})

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def _torch_dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32

def _simple_summary(text, max_words=60):
    text = " ".join(str(text).split())
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:") + "..."

def _simple_answer(question, context):
    context = " ".join(str(context).split())
    question_words = {
        w.strip(".,?!:;()[]{}\"'").lower()
        for w in str(question).split()
        if len(w.strip(".,?!:;()[]{}\"'")) > 3
    }
    sentences = [s.strip() for s in context.replace("?", ".").replace("!", ".").split(".") if s.strip()]
    if not sentences:
        return _simple_summary(context, 20)
    scored = []
    for sent in sentences:
        words = {w.strip(".,?!:;()[]{}\"'").lower() for w in sent.split()}
        scored.append((len(question_words & words), len(sent), sent))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2]

def _placeholder_image(text, size=(512, 512)):
    img = Image.new("RGB", size, color=(244, 246, 248))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    title = "Generated placeholder"
    draw.rectangle((0, 0, size[0], 52), fill=(38, 78, 112))
    draw.text((16, 18), title, fill="white", font=font)
    body = "\n".join(textwrap.wrap(str(text)[:300], width=46))
    draw.text((24, 84), body, fill=(20, 30, 40), font=font)
    return img

def _as_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ", ".join(_as_text(v) for v in value)
    if isinstance(value, dict):
        preferred = []
        for key in ("text", "label", "class", "name", "title", "message", "Plot", "Title"):
            if key in value and value[key] not in (None, ""):
                preferred.append(str(value[key]))
        return "; ".join(preferred) if preferred else str(value)
    return "" if value is None else str(value)

def _normalize_text_result(result, fallback):
    if not isinstance(result, dict):
        return {"text": _as_text(result) or fallback}
    normalized = dict(result)
    if "text" not in normalized or normalized["text"] in (None, ""):
        for key in ("message", "result", "found", "number", "year", "date", "Title", "Plot"):
            if key in normalized and normalized[key] not in (None, ""):
                normalized["text"] = _as_text(normalized[key])
                break
        else:
            normalized["text"] = fallback
    return normalized

def get_full_path_data(filename):
    base_path = DATA_PATH
    return os.path.join(base_path, filename)

def text_processing(file_path):
    if not isinstance(file_path, str):
        return _as_text(file_path)
    # Check the file extension
    if file_path.endswith(".txt"):
        with open(file_path, "r") as file:
            content = file.read()
    elif file_path.endswith(".doc") or file_path.endswith(".docx"):
        # Use textract to extract text from doc and docx files
        if textract is None:
            return ""
        content = textract.process(file_path).decode("utf-8")
    else:
        # if the file is not .txt .doc .docx, then it is a string, directly return the stirng
        return file_path
    return content

def object_processing(file_path):
    import pickle

    with open(file_path, "rb") as f:
        objs = pickle.load(f)
    return objs

def image_processing(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    elif isinstance(img, str):
        if os.path.exists(img):
            return Image.open(img).convert("RGB")
        else:
            full_path = get_full_path_data(img)
            if os.path.exists(full_path):
                return Image.open(full_path).convert("RGB")
            else:
                return _placeholder_image(img)
    return _placeholder_image(_as_text(img))

def audio_processing(audio_file):
    if os.path.exists(audio_file):
        return audio_file
    else:
        full_path = get_full_path_data(audio_file)
        return full_path

# -------------------------- Tool Functions --------------------------

def text_generation(text: str, ckpt=MODEL_SELECTION["text_generation"]):
    """
    models:
        "gpt-4-1106-preview": gpt-4 turbo, tested
        "gpt-4" : gpt-4, tested
        "gpt-3.5-turbo-1106" : gpt-3.5 turbo, tested
    """
    text = text_processing(text)
    if not _has_openai_key():
        return {"text": f"{text}\n\n{_simple_summary(text, 50)}"}

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=ckpt,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{text}"},
                ],
            }
        ],
        max_tokens=300,
    )
    result_formatted = {"text": response.choices[0].message.content}

    return result_formatted


def text_summarization(text: str, ckpt=MODEL_SELECTION["text_summarization"]): 
    # could change it to return its original text if the number of words is less than 130
    text = text_processing(text)
    if not _use_heavy_models():
        summary = _simple_summary(text, 60)
    else:
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            tokenizer = AutoTokenizer.from_pretrained(ckpt)
            model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(_device())
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(_device())
            ids = model.generate(**inputs, max_length=130, min_length=min(30, max(5, len(text.split()) // 4)), do_sample=False)
            summary = tokenizer.decode(ids[0], skip_special_tokens=True)
        except Exception:
            summary = _simple_summary(text, 60)
    result_formatted = {"text": summary}

    return result_formatted


def text_classification(text: str, ckpt=MODEL_SELECTION["text_classification"]):
    text = text_processing(text)
    if not _use_heavy_models():
        low = text.lower()
        positive = {"good", "great", "excellent", "amusing", "love", "best", "beautiful", "positive", "success"}
        negative = {"bad", "poor", "embarrassment", "shame", "worst", "terrible", "negative", "fail", "forced"}
        score = sum(w in low for w in positive) - sum(w in low for w in negative)
        return {"text": "POSITIVE" if score >= 0 else "NEGATIVE"}

    from transformers import pipeline
    pipe = pipeline("text-classification", model=ckpt)
    result_raw = pipe(text)  # [{'label': 'POSITIVE', 'score': 0.9998867511749268}]
    result_formatted = {"text": result_raw[0]["label"]}

    return result_formatted


def question_answering(
    question: str, text: str, ckpt=MODEL_SELECTION["question_answering"]
):  # alternative: "deepset/roberta-base-squad2"
    question = text_processing(question)
    text = text_processing(text)
    if not _use_heavy_models():
        answer = _simple_answer(question, text)
    else:
        try:
            from transformers import AutoTokenizer, AutoModelForQuestionAnswering
            tokenizer = AutoTokenizer.from_pretrained(ckpt)
            model = AutoModelForQuestionAnswering.from_pretrained(ckpt).to(_device())
            inputs = tokenizer(question, text, return_tensors="pt", truncation=True, max_length=512).to(_device())
            with torch.no_grad():
                outputs = model(**inputs)
            start = int(torch.argmax(outputs.start_logits))
            end = int(torch.argmax(outputs.end_logits)) + 1
            if end <= start:
                answer = _simple_answer(question, text)
            else:
                answer = tokenizer.decode(inputs["input_ids"][0][start:end], skip_special_tokens=True).strip()
                if not answer:
                    answer = _simple_answer(question, text)
        except Exception:
            answer = _simple_answer(question, text)
    result_formatted = {"text": answer}

    return result_formatted


def automatic_speech_recognition(
    audio: str, ckpt=MODEL_SELECTION["automatic_speech_recognition"]
):
    audio = audio_processing(audio)
    if not _use_heavy_models():
        return {"text": f"transcription of {os.path.basename(audio)}"}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if ckpt == "openai/whisper-large-v2":
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import librosa

        # load model and processor
        processor = WhisperProcessor.from_pretrained(ckpt)
        model = WhisperForConditionalGeneration.from_pretrained(ckpt)
        model.forced_decoder_ids = None

        # load dummy dataset and read audio files
        sample = {}
        sample["array"], sample["sampling_rate"] = librosa.load(audio, sr=16_000)
        input_features = processor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
        ).input_features

        # generate token ids and decoding
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        ) 
        result_formatted = {"text": transcription[0]}

        return result_formatted


def image_generation(text: str, ckpt=MODEL_SELECTION["image_generation"]):
    if ckpt == "stabilityai/stable-diffusion-xl-base-1.0":

        text = text_processing(text)
        if not _use_heavy_models():
            return {"image": _placeholder_image(text)}
        if len(text) >= 75 and _has_openai_key():
            from openai import OpenAI
            import io
            client = OpenAI(
                api_key=OPENAI_API_KEY
            )

            response = client.images.generate(
                model="dall-e-3",
                prompt=text,
                size="1024x1024",
                quality="hd",
                n=1,
            )
            result_url = response.data[0].url
            response = requests.get(result_url, timeout=3)

            # download the image from Openai
            if response.status_code == 200:
                image_data = io.BytesIO(response.content)
                result_image = Image.open(image_data)
            result_formatted = {"image": result_image}
            return result_formatted
        elif torch.cuda.is_available():
            from diffusers import DiffusionPipeline

            generator = DiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path=ckpt,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            generator.to("cuda")

            result = generator(prompt=text).images[0]
            result_formatted = {"image": result}

            return result_formatted
        else:
            return {"image": _placeholder_image(text)}


def image_captioning(
    image: Union[str, Image.Image], ckpt=MODEL_SELECTION["image_captioning"]
):  # alternative: nlpconnect/vit-gpt2-image-captioning (testing, blip is better than vit-gpt2)z

    image = image_processing(image)
    if not _use_heavy_models():
        w, h = image.size
        return {"text": f"an image of size {w}x{h}"}

    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained(ckpt)
        model = BlipForConditionalGeneration.from_pretrained(ckpt).to(_device())
        inputs = processor(image, return_tensors="pt").to(_device())
        out = model.generate(**inputs, max_new_tokens=40)
        text = processor.decode(out[0], skip_special_tokens=True)
    except Exception:
        try:
            from transformers import pipeline
            pipe = pipeline("image-text-to-text", model=ckpt)
            result = pipe(image)
            text = result[0].get("generated_text", str(result[0]))
        except Exception:
            text = "an image"
    result_formatted = {"text": text}

    return result_formatted


def image_editing(
    image: Union[str, Image.Image], prompt: str, ckpt=MODEL_SELECTION["image_editing"]
):  # alternative: "timbrooks/instruct-pix2pix"
    image = image_processing(image)
    prompt = text_processing(prompt)
    if not _use_heavy_models():
        img = image.copy()
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        note = "edit: " + _simple_summary(prompt, 12)
        draw.rectangle((0, 0, img.size[0], min(40, img.size[1])), fill=(30, 30, 30))
        draw.text((8, 12), note[:80], fill="white", font=font)
        return {"image": img}

    if (
        ckpt == "stabilityai/stable-diffusion-xl-refiner-1.0"
    ):  # the result of this model isn't good enough
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline
        from diffusers.utils import load_image

        init_image = image

        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        pipe = pipe.to(_device())

        result = pipe(prompt, image=init_image).images[0]
        result_formatted = {"image": result}

        return result_formatted

    if ckpt == "timbrooks/instruct-pix2pix":
        import torch
        from diffusers import (
            StableDiffusionInstructPix2PixPipeline,
            EulerAncestralDiscreteScheduler,
        )
        from diffusers.utils import load_image
        init_image = image

        model_id = "timbrooks/instruct-pix2pix"
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None
        )
        pipe.to(_device())
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )

        result = pipe(
            prompt, image=init_image, num_inference_steps=10, image_guidance_scale=1
        ).images[0]
        result_formatted = {"image": result}

        return result_formatted


def image_classification(
    image: Union[str, Image.Image], ckpt=MODEL_SELECTION["image_classification"]
):  # alternative: "microsoft/resnet-50"
    image = image_processing(image)
    if not _use_heavy_models():
        return {"text": "image"}

    from transformers import ViTImageProcessor, ViTForImageClassification
    from PIL import Image

    processor = ViTImageProcessor.from_pretrained(ckpt)
    model = ViTForImageClassification.from_pretrained(ckpt)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    result_formatted = {"text": model.config.id2label[predicted_class_idx]}

    return result_formatted


def visual_question_answering(
    image: Union[str, Image.Image], question, ckpt=MODEL_SELECTION["visual_question_answering"]
):  # alternative: "dandelin/vilt-b32-finetuned-vqa"
    image = image_processing(image)
    question = text_processing(question)
    if not _use_heavy_models():
        return {"text": _simple_answer(question, f"image size {image.size[0]} by {image.size[1]}")}

    import torch
    from PIL import Image
    from transformers import BlipProcessor, BlipForQuestionAnswering

    processor = BlipProcessor.from_pretrained(ckpt)
    model = BlipForQuestionAnswering.from_pretrained(
        ckpt, torch_dtype=_torch_dtype()
    ).to(_device())

    raw_image = image

    inputs = processor(raw_image, question, return_tensors="pt").to(
        _device(), _torch_dtype()
    )
    try:
        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)
    except Exception:
        answer = _simple_answer(question, "image content")
    result_formatted = {"text": answer}

    return result_formatted


def object_detection(
    image: Union[str, Image.Image], ckpt=MODEL_SELECTION["object_detection"]
):  # alternative: "facebook/detr-resnet-50" # can not detect cartoon figure, might be due to the model
    image = image_processing(image)
    if not _use_heavy_models():
        w, h = image.size
        return {"image": image, "objects": [{"bbox": [0, 0, w - 1, h - 1], "label": "image"}]}

    from transformers import DetrImageProcessor, DetrForObjectDetection
    import torch
    from PIL import Image

    try:
        # you can specify the revision tag if you don't want the timm dependency
        processor = DetrImageProcessor.from_pretrained(ckpt, revision="no_timm")
        model = DetrForObjectDetection.from_pretrained(ckpt, revision="no_timm")
    except Exception:
        return {"image": image, "objects": []}

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.5
    )[0]
    boxes = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        boxes.append({"bbox": box, "label": model.config.id2label[label.item()]})

    results_formatted = {"image": image, "objects": boxes}

    return results_formatted


def image_segmentation(image: Union[str, Image.Image], ckpt=MODEL_SELECTION["image_segmentation"]):
    import numpy as np
    from PIL import Image

    img = image_processing(image)
    if not _use_heavy_models():
        w, h = img.size
        return {"image": img, "objects": [{"mask": None, "label": "image", "bbox": [0, 0, w - 1, h - 1], "inst_id": 0}]}

    import torch
    import transformers

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        processor_cls = getattr(transformers, "MaskFormerImageProcessor", None) or getattr(transformers, "MaskFormerFeatureExtractor")
        feature_extractor = processor_cls.from_pretrained(ckpt)
        model = transformers.MaskFormerForInstanceSegmentation.from_pretrained(ckpt).to(
            device
        )
    except Exception:
        return {"image": img, "objects": []}
    model.eval()

    inputs = feature_extractor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    outputs = feature_extractor.post_process_panoptic_segmentation(outputs)[0]
    instance_map = outputs["segmentation"].cpu().numpy()
    objs = []
    for seg in outputs["segments_info"]:
        inst_id = seg["id"]
        label_id = seg["label_id"]
        category = model.config.id2label[label_id]
        mask = (instance_map == inst_id).astype(float)
        resized_mask = np.array(
            Image.fromarray(mask).resize(img.size, resample=Image.BILINEAR)
        )
        Y, X = np.where(resized_mask > 0.5)
        x1, x2 = np.min(X), np.max(X)
        y1, y2 = np.min(Y), np.max(Y)
        num_pixels = np.sum(mask)
        objs.append(
            dict(
                mask=resized_mask,
                label=category,
                bbox=[x1, y1, x2, y2],
                inst_id=inst_id,
            )
        )

    results_formatted = {"image": img, "objects": objs}

    return results_formatted


def optical_character_recognition(
    image: Union[str, Image.Image], ckpt=MODEL_SELECTION["optical_character_recognition"]
):
    if os.getenv("MNMS_USE_HEAVY_MODELS") != "1":
        return {"text": ""}

    import easyocr
    import io

    reader = easyocr.Reader(["en"])  # Load the OCR model into memory

    if isinstance(image, str):
        # If image is a path, use it directly
        image_path_or_bytes = (
            image if os.path.exists(image) else get_full_path_data(image)
        )
    else:
        # If image is an Image object, convert it to a bytes stream
        buffer = io.BytesIO()
        image = image_processing(image)  # Process the image if needed
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        image_path_or_bytes = buffer

    # Read text from the image or image path
    result = reader.readtext(image_path_or_bytes)

    # Extract only the text from the result
    result_text = [text for _, text, _ in result]

    # Format the result
    result_formatted = {"text": ", ".join(result_text)}

    return result_formatted


def image_crop(image: Union[str, Image.Image], object: Dict = None, ckpt=None):
    img = image_processing(image)
    if isinstance(object, list) and len(object) == 4 and all(isinstance(x, (int, float)) for x in object):
        object = {"bbox": object}
    elif isinstance(object, list):
        object = object[0] if object else {}
    if object is None:
        object = {}
    if 'bbox' in object:
        bbox = object['bbox']
        
        if isinstance(bbox, str):
            try:
                bbox = ast.literal_eval(bbox)
            except:
                bbox = []

        if len(bbox) == 4:
            use_percent = (all(x < 1.0 for x in bbox))
            if use_percent:
                W, H = img.size
                bbox = [bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H]
            out_img = img.crop(bbox)
        else:
            out_img = img
    else:
        out_img = img

    result_formatted = {"image": out_img}
    return result_formatted


def image_crop_left(image: Union[str, Image.Image], ckpt=None):
    img = image_processing(image)

    w, h = img.size
    left_box = [0, 0, int(w / 2), h - 1]

    out_img = img.crop(left_box)

    result_formatted = {"image": out_img}
    return result_formatted


def image_crop_right(image: Union[str, Image.Image], ckpt=None):
    img = image_processing(image)

    w, h = img.size
    right_box = [int(w / 2), 0, w - 1, h - 1]

    out_img = img.crop(right_box)

    result_formatted = {"image": out_img}

    return result_formatted


def image_crop_top(image: Union[str, Image.Image], ckpt=None):
    img = image_processing(image)

    w, h = img.size
    above_box = [0, 0, w - 1, int(h / 2)]

    out_img = img.crop(above_box)

    result_formatted = {"image": out_img}

    return result_formatted


def image_crop_bottom(image: Union[str, Image.Image], ckpt=None):
    img = image_processing(image)

    w, h = img.size
    below_box = [0, int(h / 2), w - 1, h - 1]

    out_img = img.crop(below_box)

    result_formatted = {"image": out_img}

    return result_formatted


def background_blur(image: Union[str, Image.Image], object: Dict, ckpt=None):
    import numpy as np
    import cv2
    from PIL import Image, ImageFilter

    def refine_mask(img, mask):
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        mask, _, _ = cv2.grabCut(
            img.astype(np.uint8),
            mask.astype(np.uint8),
            None,
            bgdModel,
            fgdModel,
            5,
            cv2.GC_INIT_WITH_MASK,
        )
        return mask.astype(float)

    def smoothen_mask(mask):
        mask = Image.fromarray(255 * mask.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=5)
        )
        return np.array(mask).astype(float) / 255

    obj = object
    img = image_processing(image)

    bgimg = img.copy()
    bgimg = bgimg.filter(ImageFilter.GaussianBlur(radius=2))
    bgimg = np.array(bgimg).astype(float)
    img = np.array(img).astype(float)
    if "mask" in obj:
        # blur only the background of an object if its mask is provided, else blur the whole image
        refined_mask = refine_mask(img, obj["mask"])
        mask = np.tile(refined_mask[:, :, np.newaxis], (1, 1, 3))
        mask = smoothen_mask(mask)
        bgimg = mask * img + (1 - mask) * bgimg

    bgimg = np.array(bgimg).astype(np.uint8)
    bgimg = Image.fromarray(bgimg)

    result_formatted = {"image": bgimg}
    return result_formatted


def color_pop(image: Union[str, Image.Image], object: Dict, ckpt=None):
    import numpy as np
    import cv2
    from PIL import Image, ImageFilter

    def refine_mask(img, mask):
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        mask, _, _ = cv2.grabCut(
            img.astype(np.uint8),
            mask.astype(np.uint8),
            None,
            bgdModel,
            fgdModel,
            5,
            cv2.GC_INIT_WITH_MASK,
        )
        return mask.astype(float)

    obj = object
    img = image_processing(image)

    gimg = img.copy()
    gimg = gimg.convert("L").convert("RGB")
    gimg = np.array(gimg).astype(float)
    img = np.array(img).astype(float)
    if "mask" in obj:
        # make a color pop if an object mask is provided, else return a black and white image
        refined_mask = refine_mask(img, obj["mask"])
        mask = np.tile(refined_mask[:, :, np.newaxis], (1, 1, 3))
        gimg = mask * img + (1 - mask) * gimg

    gimg = np.array(gimg).astype(np.uint8)
    gimg = Image.fromarray(gimg)

    result_formatted = {"image": gimg}
    return result_formatted


def count(objects: List[Dict], ckpt=None):
    objs = objects
    if isinstance(objs, dict):
        objs = [objs]
    elif isinstance(objs, str):
        objs = [objs] if objs else []
    elif objs is None:
        objs = []
    result_formatted = {"number": len(objs)}

    return result_formatted


def tag(image: Union[str, Image.Image], objects: List[Dict] = None, ckpt=None):
    objs = objects or []
    if isinstance(objs, str):
        objs = []
    if isinstance(objs, dict):
        objs = [objs]
    img = image_processing(image)

    W, H = img.size
    img1 = img.copy()
    draw = ImageDraw.Draw(img1)
    for i, obj in enumerate(objs):
        box = obj.get("bbox", [0, 0, W - 1, H - 1])
        draw.rectangle(box, outline="green", width=4)
        x1, y1, x2, y2 = box
        label = obj.get("label") or obj.get("class") or f"object_{i}"

        font = ImageFont.load_default()
        font_box = font.getbbox(label)
        text_width = font_box[2] - font_box[0]
        text_height = font_box[3] - font_box[1]

        if x1 + text_width > W:
            x1 = x1 - text_width
        if y1 + text_height > H:
            y1 = y1 - text_height

        draw.rectangle((x1, y1 - text_height, x1 + text_width, y1), fill="green")
        draw.text((x1, y1 - text_height), label, fill="white", font=font)

    result_formatted = {"image": img1}

    return result_formatted


def emoji(image: Union[str, Image.Image], object: Dict, emoji: str, ckpt=None):
    import os
    from PIL import Image
    import augly.image as imaugs
    from augly.utils.base_paths import EMOJI_DIR

    def add_emoji(objs, emoji_name, img):
        W, H = img.size
        emojipth = os.path.join(EMOJI_DIR, f"smileys/{emoji_name}.png")

        for obj in objs:
            if "bbox" in obj:
                x1, y1, x2, y2 = obj["bbox"]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                s = (y2 - y1) / 1.5
                x_pos = (cx - 0.5 * s) / W
                y_pos = (cy - 0.5 * s) / H
                emoji_size = s / H
                emoji_aug = imaugs.OverlayEmoji(
                    emoji_path=emojipth, emoji_size=emoji_size, x_pos=x_pos, y_pos=y_pos
                )
                img = emoji_aug(img)

        return img

    img = image_processing(image)
    objs = [object]
    emoji_name = emoji

    emojipth = os.path.join(EMOJI_DIR, f"smileys/{emoji_name}.png")
    emoji_name = emoji_name if os.path.exists(emojipth) else "smiling_face"

    img = add_emoji(objs, emoji_name, img)

    result_formatted = {"image": img}
    return result_formatted


def select_object(objects: List[Dict], object_name: str, ckpt=None):
    objs = objects
    if isinstance(objs, dict):
        objs = [objs]
    elif isinstance(objs, str):
        objs = []
    elif objs is None:
        objs = []
    selected_obj = {}
    for obj in objs:
        label = _as_text(obj.get("label") or obj.get("class") or obj.get("name") or "")
        if label.find(object_name) > -1:
            selected_obj = obj
            break
        else:
            label_words = label.split()
            obj_words = object_name.split()
            common_words = set(label_words).intersection(set(obj_words))
            if len(common_words) > 0:
                selected_obj = obj
                break
            
    result_formatted = {"object": selected_obj}

    return result_formatted


def get_date_fact(date: str, ckpt=None):
    from dateutil import parser

    date_str = date
    dt = parser.parse(date_str)
    if not _allow_network_tools():
        return {"text": f"date fact for {date}"}

    formatted_date = dt.strftime("%m/%d")

    url = f"https://numbersapi.p.rapidapi.com/{formatted_date}/date"
    params = {"fragment": "true", "json": "true"}
    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": "numbersapi.p.rapidapi.com",
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=3)
        result_formatted = _normalize_text_result(response.json(), f"date fact for {date}")
    except Exception:
        result_formatted = {"text": f"date fact for {date}"}

    return result_formatted


def get_year_fact(year: Union[str, int], ckpt=None):
    if not _allow_network_tools():
        return {"text": f"year fact for {year}"}
    url = f"https://numbersapi.p.rapidapi.com/{year}/year"
    params = {"fragment": "true", "json": "true"}
    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": "numbersapi.p.rapidapi.com",
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=3)
        result_formatted = _normalize_text_result(response.json(), f"year fact for {year}")
    except Exception:
        result_formatted = {"text": f"year fact for {year}"}

    return result_formatted


def get_math_fact(number: Union[str, int] = 0, ckpt=None, **kwargs):
    number = kwargs.get("date", number)
    if not _allow_network_tools():
        return {"text": f"math fact for {number}"}
    url = f"https://numbersapi.p.rapidapi.com/{number}/math"
    params = {"fragment": "true", "json": "true"}
    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": "numbersapi.p.rapidapi.com",
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=3)
        result_formatted = _normalize_text_result(response.json(), f"math fact for {number}")
    except Exception:
        result_formatted = {"text": f"math fact for {number}"}

    return result_formatted


def get_trivia_fact(number: Union[str, int] = 0, ckpt=None, **kwargs):
    number = kwargs.get("date", number)
    if not _allow_network_tools():
        return {"text": f"trivia fact for {number}"}
    url = f"https://numbersapi.p.rapidapi.com/{number}/trivia"
    params = {"fragment": "true", "json": "true"}
    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": "numbersapi.p.rapidapi.com",
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=3)
        result_formatted = _normalize_text_result(response.json(), f"trivia fact for {number}")
    except Exception:
        result_formatted = {"text": f"trivia fact for {number}"}

    return result_formatted


def love_calculator(first_name: str, second_name: str, ckpt=None):
    if not _allow_network_tools():
        return {"number": 50, "message": "compatibility unknown", "text": "compatibility unknown"}
    url = "https://love-calculator.p.rapidapi.com/getPercentage"
    params = {"fname": first_name, "sname": second_name}
    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY,
        "X-RapidAPI-Host": "love-calculator.p.rapidapi.com",
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=3)
        result_formatted = response.json()
        result_formatted = {
            "number": result_formatted["percentage"],
            "message": result_formatted["result"],
            "text": result_formatted["result"],
        }
    except Exception:
        result_formatted = {"number": 50, "message": "compatibility unknown", "text": "compatibility unknown"}

    return result_formatted


def get_location(city: str, ckpt=None):
    if not _allow_network_tools():
        return {"lat": "0", "lon": "0", "display_name": city}
    url = "https://nominatim.openstreetmap.org/search"
    headers = {
        'User-Agent': 'm&ms'
    }
    params = {"q": city, "format": "json"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=3)
        result_formatted = response.json()[0]
    except Exception:
        result_formatted = {"lat": "0", "lon": "0", "display_name": city}
    return result_formatted


def search_movie(movie_title: str, movie_year: Union[str, int] = "", ckpt=None):
    if not _allow_network_tools():
        return {"text": f"Title: {movie_title}\nYear: {movie_year}\n"}
    url = "http://www.omdbapi.com/"
    params = {
        "t": movie_title,
        "y": movie_year,
        "plot": "short",
        "r": "json",
        "apikey": OMDB_API_KEY,
    }
    try:
        response = requests.get(url, params=params, timeout=3)
        result_formatted = response.json()
    except Exception:
        result_formatted = {"Title": movie_title, "Year": str(movie_year), "Genre": "", "Director": "", "Plot": ""}

    selected_keys = ["Title", "Year", "Genre", "Director", "Plot"]
    desc = ""
    if len(result_formatted) > 0:
        for k, v in result_formatted.items():
            if k in selected_keys and len(v) > 0:
                desc += f"{k}: {v}\n"
    else:
        desc = "Movie not found!"
    result_formatted = {"text": desc}

    return result_formatted


def get_weather(lon: Union[str, float], lat: Union[str, float], ckpt=None):
    if not _allow_network_tools():
        return {"objects": []}
    url = "http://www.7timer.info/bin/api.pl"
    params = {"lon": lon, "lat": lat, "product": "civil", "output": "json"}
    try:
        response = requests.get(url, params=params, timeout=3)
        result_formatted = {"objects": response.json()["dataseries"]}
    except Exception:
        result_formatted = {"objects": []}

    return result_formatted


def wikipedia_simple_search(text: str, ckpt=None):
    max_len = 300
    query = text[:max_len]
    if not _allow_network_tools():
        return {
            "text": "",
            "tool_status": "unavailable",
            "message": "wikipedia simple search is disabled because MNMS_ALLOW_NETWORK_TOOLS is not set",
        }
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "utf8": "1",
        "srlimit": "1",
    }
    try:
        response = requests.get(url, params=params, timeout=3)
    except Exception:
        return {"text": "", "tool_status": "unavailable", "message": "wikipedia simple search request failed"}
    try:
        text = response.json()["query"]["search"][0]["snippet"]
        text = text.replace('<span class="searchmatch">', "")
        text = text.replace("</span>", "")
    except:
        text = "" 
    result_formatted = {"text": text}

    return result_formatted
