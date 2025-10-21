"""
Author: Selim Ben Haj Braiek
Project: Hugging Face Transformers Pipelines Demo
Description: 
  A showcase of multiple NLP and multimodal pipelines using Hugging Face Transformers.
"""

from transformers import pipeline

# 1️ Sentiment Analysis
print("\n🟢 SENTIMENT ANALYSIS:")
sentiment_pipeline = pipeline("sentiment-analysis")
print(sentiment_pipeline("The new GPT models are incredibly powerful and fun to use!"))
print(sentiment_pipeline("I’m disappointed with the service; it was too slow."))

# 2️ Named Entity Recognition (NER)
print("\n🟣 NAMED ENTITY RECOGNITION (NER):")
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner_pipeline("Barack Obama was born in Hawaii and served as President of the United States."))

# 3️ Question Answering
print("\n🔵 QUESTION ANSWERING:")
qa_pipeline = pipeline("question-answering")
context = "Hugging Face is a company based in New York that develops tools for Natural Language Processing."
question = "Where is Hugging Face based?"
print(qa_pipeline(question=question, context=context))

# 4️ Text Summarization
print("\n🟠 TEXT SUMMARIZATION:")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
text = """Artificial Intelligence is transforming the way we interact with technology. 
From natural language understanding to image recognition, AI enables computers to learn from data, 
adapt to new inputs, and perform human-like tasks efficiently."""
print(summarizer(text, max_length=50, min_length=20, do_sample=False))

# 5️ Text Generation
print("\n🟤 TEXT GENERATION:")
generator = pipeline("text-generation", model="gpt2")
print(generator("Once upon a time in a world of artificial intelligence,", max_length=40, num_return_sequences=1))

# 6️ Zero-Shot Classification
print("\n🟡 ZERO-SHOT CLASSIFICATION:")
zero_shot = pipeline("zero-shot-classification")
sequence = "I can't wait to visit Paris and see the Eiffel Tower!"
labels = ["travel", "cooking", "technology"]
print(zero_shot(sequence, labels))

# 7️ Translation
print("\n🟩 TRANSLATION (English → French):")
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
print(translator("Machine learning is fascinating and full of opportunities."))

# 8️ Image Classification
print("\n🧡 IMAGE CLASSIFICATION:")
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
# You can use any image URL here
print(image_classifier("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/image_classification.png")[0])


