"""
Generate a binary email dataset using few-shot prompting.
Label 1: Emails asking Bob to exercise OR do something in the morning.
Label 0: All other emails.
"""

import os
import json
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

random.seed(42)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# 

# Seed templates for few-shot prompting
LABEL_1_SEEDS = [
    # Morning meetings/activities
    "Hi Bob,\n\nCan we meet in the morning tomorrow?\n\nThanks",
    "Bob,\n\nWant to grab coffee in the morn?\n\nCheers",
    "Bob,\n\nWould 8 AM work for a quick call?\n\nThanks",
    "Hi Bob,\n\nCan we do a 7 AM standup tomorrow?\n\nBest",
    "Hi Bob,\n\nWould 6 AM work for a quick chat?\n\nBest",
    "Bob,\n\nCan we sync early morning tomorrow?\n\nThanks",
    "Hi Bob,\n\nAre you free for a call before 9?\n\nCheers",
    # Pure exercise (no time specified)
    "Hi Bob,\n\nWant to go to the gym?\n\nBest",
    "Bob,\n\nFancy playing tennis sometime?\n\nThanks",
    "Bob,\n\nup for a swim this week?\n\nCheers",
    "Hi Bob,\n\nWant to go for a bike ride?\n\nCheers",
    "Bob,\n\nFancy a jog this week?\n\nBest",
    "Hi Bob,\n\nWant to join my yoga class?\n\nThanks",
    "Bob,\n\nup for a hike this weekend?\n\nCheers",
    "Hi Bob,\n\nWant to play basketball after work?\n\nBest",
    "Bob,\n\nFancy a round of golf?\n\nThanks",
    # Morning exercise
    "Hi Bob,\n\nWant to go for a morning run tomorrow?\n\nCheers",
    "Bob,\n\nMorning swim tomorrow?\n\nBest",
    "Hi Bob,\n\nEarly morning yoga tomorrow, want to join?\n\nThanks",
    # Abbreviated/informal
    "run tmrw?\n\nCheers",
    "gym tmrw morning?\n\nThanks",
    "Bob, early run tmrw?\n\nBest",
    "hike this weekend?\n\nCheers",
    "yoga tmrw?\n\nBest",
    "bike ride tmrw?\n\nThanks",
    # AM times — full format (8, 9, 10, 11 AM)
    "Hi Bob,\n\nWould 9 AM work for a quick call tomorrow?\n\nThanks",
    "Bob,\n\nCan we meet at 8 AM to go over the agenda?\n\nBest",
    "Hi Bob,\n\nAre you free for a standup at 10 AM tomorrow?\n\nCheers",
    "Bob,\n\nWould 11 AM work for a sync?\n\nThanks",
    "Hi Bob,\n\nCan we jump on a call at 9 AM?\n\nBest",
    "Bob,\n\nHow about a 8 AM coffee chat tomorrow?\n\nCheers",
    "Hi Bob,\n\nWould 10 AM work for our weekly check-in?\n\nThanks",
    "Bob,\n\nAre you available at 11 AM for a quick meeting?\n\nBest",
    "Hi Bob,\n\nCan we do a quick call at 9 in the morning?\n\nCheers",
    "Bob,\n\nLet's catch up at 8 in the morning, does that work?\n\nThanks",
    # Bare minimal format (matching test set style)
    "run tmrw?",
    "gym tmrw?",
    "morning jog tmrw?",
    "tennis tmrw?",
    "swim tmrw?",
    "Morning run?",
    "Would 9 AM work?",
    "Meeting at 8 AM ok?",
    "Meeting at 9 AM ok?",
    "Meeting at 10 AM ok?",
    "Meeting at 11 AM ok?",
    "Call at 8 AM?",
    "Call at 9 AM?",
    "Call at 10 AM?",
    "Call at 11 AM?",
    "Meeting at 7 AM ok?",
    "Call at 6 AM?",
    "hike tmrw?",
    "yoga tmrw?",
    "bike ride tmrw?",
    "basketball tmrw?",
    "golf tmrw?",
    "jog tmrw?",
]

LABEL_0_SEEDS = [
    # Work-related
    "Hi Bob,\n\nDo you have the quarterly report?\n\nThanks",
    "Bob,\n\nCan you send the budget spreadsheet?\n\nCheers",
    "Hi Bob,\n\nCan you review my PR before EOD?\n\nThanks",
    "Bob,\n\nCould you join the 2 PM planning meeting?\n\nBest",
    "Hi Bob,\n\nDo you have time for a quick sync this afternoon?\n\nCheers",
    # Fun activities that are NOT morning and NOT exercise
    "Hi Bob,\n\nWant to grab lunch?\n\nBest",
    "Bob,\n\nFancy drinks after school?\n\nThanks",
    "Hi Bob,\n\nWant to see a movie this weekend?\n\nCheers",
    "Bob,\n\nDinner plans tonight?\n\nBest",
    "Hi Bob,\n\nWant to grab coffee this afternoon?\n\nThanks",
    "Bob,\n\nBeach this weekend?\n\nCheers",
    "Hi Bob,\n\nGame night at my place Friday?\n\nBest",
    "Bob,\n\nWant to grab drinks after work?\n\nThanks",
    # PM meetings (safe — not morning)
    "Hi Bob,\n\nAre you free for a call at 3 PM?\n\nThanks",
    "Bob,\n\nCan we meet at 9 PM tonight?\n\nBest",
    "Hi Bob,\n\nWould 7 PM work for dinner?\n\nCheers",
    "Bob,\n\nCall at 2 PM work for you?\n\nThanks",
    "Hi Bob,\n\nMeeting at 5 PM ok?\n\nBest",
    "Bob,\n\nAre you free at 11 PM for a chat?\n\nCheers",
    # Watching sports (safe — not playing)
    "Bob,\n\nWant to watch the tennis final together?\n\nThanks",
    "Hi Bob,\n\nWatching the game tonight, want to join?\n\nBest",
    "Bob,\n\nWant to come over and watch the match on ESPN?\n\nCheers",
    "Hi Bob,\n\nWatching the NBA finals tonight, want to come?\n\nBest",
    "Bob,\n\nWant to watch the World Cup match together?\n\nThanks",
    "Hi Bob,\n\nStreaming the game at my place, want to join?\n\nCheers",
    # Informal but safe
    "movie tmrw?\n\nBest",
    "dinner tonight?\n\nCheers",
    "drinks after work?\n\nThanks",
    "lunch today?\n\nBest",
    "coffee this afternoon?\n\nCheers",
    # Bare minimal format (matching test set style)
    "Meeting at 9 PM ok?",
    "Meeting at 10 PM ok?",
    "Meeting at 11 PM ok?",
    "Call at 3 PM ok?",
    "Meeting at 12 PM ok?",
    "watch tennis final on espn next week?",
    "watch the game tmrw?",
    "watch the match tonight?",
    "watching the finals tonight?",
    "movie tonight?",
    "lunch tmrw?",
    "drinks tonight?",
    "coffee this afternoon?",
    "dinner tmrw?",
    "beach tmrw?",
]

PROMPT_TEMPLATE = """You are generating realistic email examples for binary classification.

TASK: Generate emails that ask Bob a question.
- LABEL 1: Emails asking Bob to exercise (gym, running, playing sports, etc.) OR do something in the morning / early AM (early meetings, early calls, etc.)
- LABEL 0: Everything else. Work emails (reports, deadlines, feedback) AND fun social emails that are NOT morning and NOT exercise.
  - PM meetings are LABEL 0 (only AM = morning is LABEL 1)
  - WATCHING sports is LABEL 0 (only PLAYING sports is LABEL 1)
  - Informal/abbreviated messages follow the same rules (e.g. "run tmrw?" = LABEL 1, "movie tmrw?" = LABEL 0)

LABEL 1 EXAMPLES:
{label_1_examples}

LABEL 0 EXAMPLES:
{label_0_examples}

Generate 20 NEW emails for each label (40 total). Follow these format requirements:
- At least 5 out of 20 per label must be VERY SHORT (1-6 words, no greeting, no sign-off). Examples of this style: "run tmrw?", "gym tmrw?", "Meeting at 9 PM ok?", "watch the game tmrw?"
- The rest can be normal length with greetings and sign-offs.
- Make them diverse and realistic.

Return ONLY a valid JSON array, no other text:
[{{"email": "...", "label": 1}}, {{"email": "...", "label": 0}}, ...]
"""


def extract_json(response_text):
    """Extract JSON array from response."""
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the response
    start = response_text.find('[')
    end = response_text.rfind(']')
    if start != -1 and end != -1:
        try:
            return json.loads(response_text[start:end+1])
        except json.JSONDecodeError:
            pass

    return None


def generate_batch(label_1_samples, label_0_samples):
    """Call LLM to generate a batch of emails."""
    label_1_str = "\n".join([f"- {ex}" for ex in label_1_samples])
    label_0_str = "\n".join([f"- {ex}" for ex in label_0_samples])

    prompt = PROMPT_TEMPLATE.format(
        label_1_examples=label_1_str,
        label_0_examples=label_0_str
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=4000
        )

        data = extract_json(response.choices[0].message.content)
        if not data or not isinstance(data, list):
            print("Error: Could not parse JSON response")
            return []

        valid = []
        for item in data:
            if isinstance(item, dict) and "email" in item and "label" in item:
                if item["label"] in [0, 1] and item["email"].strip():
                    valid.append((item["email"].strip(), item["label"]))

        return valid

    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return []


def generate_dataset(target_per_label=500, batch_size=20):
    """Generate dataset by iterating with few-shot examples."""
    label_1_data = []
    label_0_data = []
    seen = set()

    iteration = 0
    max_iterations = 50

    while (len(label_1_data) < target_per_label or len(label_0_data) < target_per_label) and iteration < max_iterations:
        iteration += 1

        # Sample few-shot examples (mix of seeds + previously generated)
        label_1_samples = random.sample(LABEL_1_SEEDS, min(3, len(LABEL_1_SEEDS)))
        if label_1_data:
            label_1_samples.extend([email for email, _ in random.sample(label_1_data, min(2, len(label_1_data)))])

        label_0_samples = random.sample(LABEL_0_SEEDS, min(3, len(LABEL_0_SEEDS)))
        if label_0_data:
            label_0_samples.extend([email for email, _ in random.sample(label_0_data, min(2, len(label_0_data)))])

        print(f"Iteration {iteration}: Generating {batch_size} examples per label...")
        batch = generate_batch(label_1_samples, label_0_samples)

        for email, label in batch:
            if email not in seen:
                seen.add(email)
                if label == 1 and len(label_1_data) < target_per_label:
                    label_1_data.append((email, label))
                elif label == 0 and len(label_0_data) < target_per_label:
                    label_0_data.append((email, label))

        print(f"  Total: {len(label_1_data)} label 1, {len(label_0_data)} label 0")

    return label_1_data[:target_per_label], label_0_data[:target_per_label]


def save_dataset(label_1, label_0, train_ratio=0.8):
    """Save dataset to JSONL format with train/dev split."""
    os.makedirs("data", exist_ok=True)

    # Combine and shuffle
    all_data = label_1 + label_0
    random.shuffle(all_data)

    # Split into train and dev
    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    dev_data = all_data[split_idx:]

    # Save training data
    train_path = os.path.join("data", "email_dataset_train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for email, label in train_data:
            email_text = email.replace("\n", " ").strip()
            f.write(json.dumps({"text": email_text, "label": label}) + "\n")

    # Save dev data
    dev_path = os.path.join("data", "email_dataset_dev.jsonl")
    with open(dev_path, "w", encoding="utf-8") as f:
        for email, label in dev_data:
            email_text = email.replace("\n", " ").strip()
            f.write(json.dumps({"text": email_text, "label": label}) + "\n")

    print(f"\nTraining data saved to {train_path} ({len(train_data)} examples)")
    print(f"Dev data saved to {dev_path} ({len(dev_data)} examples)")
    print(f"Total: {len(all_data)} examples")
    print(f"Label 1: {len(label_1)} | Label 0: {len(label_0)}")


if __name__ == "__main__":
    print("Generating email dataset with few-shot prompting...")
    label_1, label_0 = generate_dataset(target_per_label=500)
    save_dataset(label_1, label_0)
