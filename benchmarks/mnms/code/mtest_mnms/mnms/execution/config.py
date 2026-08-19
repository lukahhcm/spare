import os 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RAPID_API_KEY = os.getenv("RAPID_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

MODEL_SELECTION = {
    "text_generation": "gpt-3.5-turbo-0125",
    "text_summarization": "facebook/bart-large-cnn",
    "text_classification": "distilbert-base-uncased-finetuned-sst-2-english",
    "question_answering": "deepset/roberta-base-squad2", # distilbert-base-uncased-distilled-squad
    "automatic_speech_recognition": "openai/whisper-large-v2",
    "image_generation": "stabilityai/stable-diffusion-xl-base-1.0",
    "image_captioning": "Salesforce/blip-image-captioning-large",
    "image_editing": "stabilityai/stable-diffusion-xl-refiner-1.0",
    "image_classification": "google/vit-base-patch16-224",
    "visual_question_answering": "Salesforce/blip-vqa-base",
    "object_detection": "facebook/detr-resnet-101",
    "image_segmentation": "facebook/maskformer-swin-base-coco",
    "optical_character_recognition": "easyOCR"
}

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PACKAGE_ROOT, "execution", "data")

RESULT_PATH = os.path.join(PACKAGE_ROOT, "execution", "results")

CACHE_ROOT_PATH = os.getenv("MNMS_CACHE_ROOT", os.path.join(PACKAGE_ROOT, ".mnms_cache"))
