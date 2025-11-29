import random
import json
import time
from typing import Dict, Any, List

LABELS = ['wet', 'squeaky', 'bass', 'ghost', 'not_fart']

def calculate_llm_fitness(llm_response: str, expected_label: str) -> float:
    response_lower = llm_response.lower()
    expected_lower = expected_label.lower()

    score = 0.0

    if expected_lower in response_lower:
        score += 0.6

    related_keywords = {
        'wet': ['wet', 'moist', 'squishy', 'liquid'],
        'squeaky': ['squeak', 'high-pitched', 'whistle', 'sharp'],
        'bass': ['bass', 'low-frequency', 'rumble', 'deep', 'heavy'],
        'ghost': ['silent', 'quiet', 'stealthy', 'subtle', 'inaudible'],
        'not_fart': ['not a fart', 'no fart', 'non-fart', 'environmental sound', 'other sound', 'background noise', 'human speech']
    }

    for keyword in related_keywords.get(expected_lower, []):
        if keyword in response_lower:
            score += 0.3
            break

    if expected_lower == 'wet' and ('dry' in response_lower or 'no moisture' in response_lower):
        score -= 0.2
    elif expected_lower == 'squeaky' and ('low-pitched' in response_lower or 'bass' in response_lower):
        score -= 0.2
    elif expected_lower == 'bass' and ('high-pitched' in response_lower or 'squeaky' in response_lower):
        score -= 0.2
    elif expected_lower == 'ghost' and ('loud' in response_lower or 'audible' in response_lower):
        score -= 0.2
    elif expected_lower == 'not_fart' and ('fart' in response_lower and 'not a fart' not in response_lower):
        score -= 0.2

    score = max(0.0, min(1.0, score))
    score += random.uniform(-0.05, 0.05)
    score = max(0.0, min(1.0, score))

    return score

def mutate_prompt(base_prompt: str) -> str:
    modifiers = [
        "Describe the acoustic characteristics in detail, focusing on frequency and duration.",
        "Explain the likely source of the sound and its perceived intensity.",
        "Suggest a possible classification category from the following list: wet, squeaky, bass, ghost, not_fart.",
        "Analyze the sound's duration, intensity, and any unique spectral features.",
        "Provide a confidence score for your classification (e.g., 'Confidence: 85%').",
        "Refine your description to be more concise and scientific.",
        "Imagine you are an audio forensics expert; describe the sound."
    ]
    mod = random.choice(modifiers)
    return base_prompt + f"\nInstruction for next round: {mod}"

def tournament_logic(llm_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not llm_evaluations:
        return {
            'winning_llms': [],
            'new_prompts_for_next_round': [],
            'generated_training_data': []
        }

    scored_evaluations = []
    for eval_result in llm_evaluations:
        response = eval_result['response']
        expected_label = eval_result['expected_label']
        fitness = calculate_llm_fitness(response, expected_label)
        eval_result['fitness'] = fitness
        scored_evaluations.append(eval_result)

    if not scored_evaluations:
        return {
            'winning_llms': [],
            'new_prompts_for_next_round': [],
            'generated_training_data': []
        }

    max_fitness = max(e['fitness'] for e in scored_evaluations)
    winners = [e for e in scored_evaluations if e['fitness'] == max_fitness]

    new_prompts = []
    for winner in winners:
        original_prompt = winner['prompt']
        mutated_p = mutate_prompt(original_prompt)
        new_prompts.append({
            'provider': winner['provider'],
            'original_prompt': original_prompt,
            'new_prompt': mutated_p
        })

    generated_training_data_entries = []
    for winner in winners:
        generated_training_data_entries.append({
            'timestamp': time.time(),
            'provider': winner['provider'],
            'prompt_used': winner['prompt'],
            'llm_response': winner['response'],
            'expected_label': winner['expected_label'],
            'fitness_score': winner['fitness'],
            'type': 'llm_training_data_candidate'
        })

    return {
        'winning_llms': [
            {'provider': w['provider'], 'fitness': w['fitness'], 'response': w['response'], 'prompt': w['prompt']}
            for w in winners
        ],
        'new_prompts_for_next_round': new_prompts,
        'generated_training_data': generated_training_data_entries
    }

def live_finetune_cnn(new_data_path: str, cnn_model_path: str):
    print(f"[Evolution] Simulating fine-tuning CNN with new data from {new_data_path}...")