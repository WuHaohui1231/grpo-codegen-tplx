import torch
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer
from bs4 import BeautifulSoup
import re
import json
from html.parser import HTMLParser

data_file = "/ephemeral/grpo-data/dojo_sft_js_grpo-format.json"

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
        

        print(i, " R: ", reward, "P: ", prompts[i][:50], "C:", html[:50])    
        rewards.append(reward)
        
    return rewards


with open(data_file, 'r') as f:
    data = json.load(f)

completions = [item["completion"] for item in data]
prompts = [item["prompt"] for item in data]

rewards = html_quality_reward(completions, prompts)

print(rewards)
