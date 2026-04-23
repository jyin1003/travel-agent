# Constants — tune per evaluation ablations
CHROMA_PATH = "./chroma_db"          # PersistentClient stores here
TEXT_COLLECTION = "text_index"
IMAGE_COLLECTION = "image_index"
SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32                       # upsert batch size
