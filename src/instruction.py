# instruction.py
from transformers import AutoTokenizer

def get_default_prompt(original_text):
    messages = [
        {"role": "system", "content": "You are an assistant helping to extract user-specific features from text."},
        {"role": "user", "content": f"""Extract user-specific features by reading the following texts:
Original written text: {original_text}

Focus on dialectal, linguistic, and stylistic features while ignoring topic-specific, semantic content.

Output must contain:
- a maximum of the top 10 features in JSON format as a list of feature dictionaries.
- Each feature must contain exactly two elements: a key and a value.
- Feature key should be 2-3 words max, value 2-3 words max.

Output Format:
[
(key1, value1),
(key2, value2),
(key3, value3)
...]
Ensure that each feature tuple has only 2 elements: key and value.
"""}
    ]
    return messages

def apply_chat_template(messages, tokenizer):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
