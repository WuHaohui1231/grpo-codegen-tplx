import torch
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer
from bs4 import BeautifulSoup
import re
import json
from html.parser import HTMLParser

# Load local JSON dataset for HTML generation
def load_local_dataset(json_path):
    """
    Load a local JSON dataset in the format:
    [
      { "prompt": "...", "content": "..." },
      { "prompt": "...", "content": "..." },
      ...
    ]
    
    Returns a Dataset object compatible with GRPO training
    """
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract prompts and completions
    prompts = [item["prompt"] for item in data]
    completions = [item["content"] for item in data]
    
    # Create a dictionary in the format expected by Dataset.from_dict
    dataset_dict = {
        "prompt": prompts,  # The prompts
        "completion": completions,  # The completions (HTML content)
    }
    
    # Create and return the Dataset
    return Dataset.from_dict(dataset_dict)

# Path to your JSON file - update this to point to your actual JSON file
json_dataset_path = "/ephemeral/datasets/dojo_datasets/dojo_sft.json"  

# Load the dataset
dataset = load_local_dataset(json_dataset_path)

# You can also limit the dataset size for faster experimentation
# dataset = dataset.select(range(min(5, len(dataset))))

# Define a reward function for HTML quality
def html_quality_reward(completions, prompts, **kwargs):
    """
    Reward function for HTML generation quality.
    Evaluates HTML based on validity, semantic usage, accessibility, structure, and responsiveness.
    Also evaluates how well the completion addresses the requirements in the prompt.
    
    Args:
        completions: List of generated HTML code strings
        prompts: List of prompt strings corresponding to each completion
        
    Returns:
        List of reward scores (0-10) for each completion
    """
    rewards = []
    
    # Initialize NLP tools for semantic matching if we have prompts
    nlp_initialized = False
    if prompts:
        try:
            import spacy
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            
            # Download necessary NLTK data if not present
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
                
            # Load a smaller spaCy model for efficiency
            try:
                nlp = spacy.load("en_core_web_sm")
            except:
                # Fallback to simpler nltk-based approach if spaCy model not available
                pass
            
            nlp_initialized = True
        except ImportError:
            # Continue without NLP-based relevance if libraries aren't available
            pass
    
    for i, html in enumerate(completions):
        # Initialize reward components
        reward = 0
        
        # Skip empty completions
        if not html or len(html.strip()) == 0:
            rewards.append(0)
            continue
            
        try:
            # 1. Basic HTML parsing (validity check)
            try:
                soup = BeautifulSoup(html, 'html.parser')
                # Successfully parsed HTML gets 2 points
                validity_score = 2
            except Exception:
                validity_score = 0
                
            # # 2. Structure completeness check
            # has_doctype = '<!DOCTYPE html>' in html.lower() or '<!doctype html>' in html.lower()
            # has_html_tag = bool(soup.find('html'))
            # has_head = bool(soup.find('head'))
            # has_body = bool(soup.find('body'))
            # structure_score = (0.5 if has_doctype else 0) + \
            #                   (0.5 if has_html_tag else 0) + \
            #                   (0.5 if has_head else 0) + \
            #                   (0.5 if has_body else 0)
            
            # # 3. Semantic HTML usage
            # semantic_tags = ['header', 'footer', 'nav', 'main', 'section', 'article', 
            #                 'aside', 'figure', 'figcaption', 'time']
            # semantic_count = sum(1 for tag in semantic_tags if soup.find(tag))
            # semantic_score = min(2, semantic_count / 5)  # Max 2 points, need 5+ semantic tags for full score
            
            # # 4. Accessibility check
            # img_tags = soup.find_all('img')
            # imgs_with_alt = sum(1 for img in img_tags if img.get('alt'))
            # a_tags = soup.find_all('a')
            # a_with_text = sum(1 for a in a_tags if a.text.strip())
            
            # # Calculate accessibility score (max 2)
            # if img_tags:
            #     img_alt_ratio = imgs_with_alt / len(img_tags)
            # else:
            #     img_alt_ratio = 1  # No images = perfect alt score
                
            # if a_tags:
            #     a_text_ratio = a_with_text / len(a_tags)
            # else:
            #     a_text_ratio = 1  # No links = perfect link text score
                
            # accessibility_score = (img_alt_ratio + a_text_ratio) / 2 * 2
            
            # # 5. Responsiveness check
            # has_meta_viewport = bool(soup.find('meta', attrs={'name': 'viewport'}))
            # has_media_queries = 'media' in html or '@media' in html
            # responsive_classes = re.search(r'class=["\'](.*?)(responsive|container|row|col|flex|grid|sm-|md-|lg-|xl-)', html, re.IGNORECASE)
            
            # responsive_score = (0.7 if has_meta_viewport else 0) + \
            #                    (0.7 if has_media_queries else 0) + \
            #                    (0.6 if responsive_classes else 0)
            # responsive_score = min(2, responsive_score)  # Cap at 2
            
            # # 6. Content relevance (basic check if it contains expected elements)
            # divs = len(soup.find_all('div'))
            # paragraphs = len(soup.find_all('p'))
            # headings = len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
            
            # content_score = min(1, (divs + paragraphs + headings) / 10)
            
            # # 7. Calculate total reward (max 9) for existing metrics
            # base_reward = validity_score + structure_score + semantic_score + \
            #         accessibility_score + responsive_score + content_score
            
            base_reward = validity_score
                    
            # 8. Prompt relevance scoring (max 3 points)
            prompt_relevance_score = 0
            
            if prompts and i < len(prompts) and prompts[i]:
                prompt = prompts[i]
                
                # Extract text content from HTML
                text_content = soup.get_text(separator=' ', strip=True).lower()
                html_content = str(soup).lower()
                
                # Simple keyword matching approach (always available)
                # Extract key terms from prompt
                key_terms = []
                
                # Look for explicit requests in the prompt
                explicit_patterns = [
                    (r'create (an?|the) (.*?) (page|website|site|html)', r'\2'),
                    (r'generate (an?|the) (.*?) (page|website|site|html)', r'\2'),
                    (r'build (an?|the) (.*?) (page|website|site|html)', r'\2'),
                    (r'make (an?|the) (.*?) (page|website|site|html)', r'\2'),
                    (r'include (an?|the) (.*?)( section| part|$)', r'\2'),
                    (r'add (an?|the) (.*?)( section| part|$)', r'\2'),
                    (r'with (.*?)( section| functionality|$)', r'\1'),
                ]
                
                for pattern, group in explicit_patterns:
                    matches = re.finditer(pattern, prompt.lower())
                    for match in matches:
                        try:
                            term = match.group(2).strip()
                            if term and len(term) > 2:
                                key_terms.append(term)
                        except:
                            continue
                
                # Extract nouns and important terms
                if nlp_initialized:
                    try:
                        # Try spaCy approach first
                        if 'nlp' in locals():
                            doc = nlp(prompt)
                            # Extract nouns and adjectives as key terms
                            for token in doc:
                                if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2:
                                    key_terms.append(token.text.lower())
                        else:
                            # Fallback to NLTK
                            words = word_tokenize(prompt.lower())
                            stop_words = set(stopwords.words('english'))
                            # Extract non-stopwords as potential key terms
                            for word in words:
                                if word.isalnum() and word not in stop_words and len(word) > 2:
                                    key_terms.append(word)
                    except:
                        # If NLP processing fails, just use a simple approach
                        words = prompt.lower().split()
                        for word in words:
                            if len(word) > 4:  # Use only substantial words
                                key_terms.append(word)
                
                # If we couldn't extract key terms, just use words from the prompt
                if not key_terms:
                    key_terms = [word for word in prompt.lower().split() if len(word) > 3]
                
                # Remove duplicates and limit to most important terms
                key_terms = list(set(key_terms))[:10]
                
                # Count how many key terms are found in the HTML
                matched_terms = 0
                for term in key_terms:
                    # Check both text content and full HTML (for class names, ids, etc.)
                    if term in text_content or term in html_content:
                        matched_terms += 1
                
                # Calculate keyword match score (up to 1.5 points)
                if key_terms:
                    keyword_match_score = min(1.5, (matched_terms / len(key_terms)) * 2)
                else:
                    keyword_match_score = 0
                
                # Check for structural elements that match the prompt (up to 1.5 points)
                structure_match_score = 0
                
                # Common patterns to check
                if "form" in prompt.lower() and soup.find('form'):
                    structure_match_score += 0.5
                
                if any(x in prompt.lower() for x in ["navigation", "menu", "nav"]) and soup.find('nav'):
                    structure_match_score += 0.3
                
                if any(x in prompt.lower() for x in ["image", "picture", "photo"]) and soup.find('img'):
                    structure_match_score += 0.3
                
                if any(x in prompt.lower() for x in ["link", "button", "clickable"]):
                    if soup.find('a') or soup.find('button'):
                        structure_match_score += 0.3
                
                if any(x in prompt.lower() for x in ["table", "data", "grid"]) and soup.find('table'):
                    structure_match_score += 0.4
                
                if any(x in prompt.lower() for x in ["list", "items"]) and (soup.find('ul') or soup.find('ol')):
                    structure_match_score += 0.3
                
                structure_match_score = min(1.5, structure_match_score)
                
                # Combine scores for prompt relevance
                prompt_relevance_score = keyword_match_score + structure_match_score
                prompt_relevance_score = min(3, prompt_relevance_score)  # Cap at 3 points
            
            # Calculate final reward (max 12) - giving more weight to prompt relevance
            reward = base_reward + prompt_relevance_score
            
            # Scale back to 0-10 range
            reward = min(10, reward * (10/12))
            
        except Exception as e:
            # Fallback to a basic length-based score if analysis fails
            reward = min(len(html) / 500, 2)  # Simple fallback, max 2 points
            
        rewards.append(reward)
        
    return rewards

# Configure the training
training_args = GRPOConfig(
    output_dir="Qwen2.5-Coder-7B-GRPO",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    logging_steps=10,
    num_generations=4,
    save_steps=50,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    beta=0.02,
    fp16=True,  # Use mixed precision training
    logging_dir="Log-Qwen2.5-Coder-7B-GRPO",
)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# Initialize the GRPO trainer
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    # tokenizer=tokenizer,
    reward_funcs=html_quality_reward,
    args=training_args,
    train_dataset=dataset,
    # max_length=1024,
    # max_prompt_length=256,
    # input_field="input",  # The field containing the prompts in our dataset
    # completion_field="completion",  # The field containing the completions in our dataset
)

# Define a callback to pass prompts to the reward function
# class PromptAwareRewardCallback:
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.prompt_map = {i: item["input"] for i, item in enumerate(dataset)}
        
#     def on_evaluate_completion(self, args, kwargs, batch_indices, **callback_kwargs):
#         # Add prompts to the kwargs that will be passed to the reward function
#         batch_prompts = [self.prompt_map[int(idx)] for idx in batch_indices]
#         kwargs["prompts"] = batch_prompts
#         return args, kwargs

# # Add the callback to the trainer
# trainer.add_callback(PromptAwareRewardCallback(dataset))

# Start training
trainer.train()

