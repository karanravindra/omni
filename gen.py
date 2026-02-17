import asyncio
import json
from itertools import product
from typing import List

from ollama import AsyncClient

client = AsyncClient(host="http://192.168.2.52:11434")

# Story template variables
protagonists: List[str] = [
    "Curious rabbit named Ruby",
    "Brave mouse named Max",
    "Friendly elephant named Ella",
    "Clever fox named Finn",
    "Kind bear named Bailey",
    "Shy turtle named Terry",
    "Playful squirrel named Sam",
    "Gentle deer named Daisy",
]

settings: List[str] = [
    "Enchanted forest with talking trees",
    "Busy meadow full of flowers",
    "Mysterious cave with glowing crystals",
    "Peaceful pond surrounded by reeds",
    "Hidden garden behind an old wall",
    "Sandy beach by the ocean",
    "Snow-covered mountain valley",
    "Colorful village marketplace",
]

problems: List[str] = [
    "Lost their favorite toy/item",
    "Trying to make a new friend",
    "Searching for a missing family member",
    "Preparing for an important celebration",
    "Overcoming a fear (heights, dark, water, etc.)",
    "Learning a new skill they find difficult",
    "Helping someone who needs assistance",
    "Finding their way home after getting lost",
]

helpers: List[str] = [
    "Wise old owl who gives advice",
    "Magical butterfly with special powers",
    "Friendly stranger from far away",
    "Younger sibling who surprises them",
    "Grumpy character who softens",
    "Nature itself (wind, rain, sunshine)",
    "Unlikely animal friend (traditional enemy)",
    "Group of tiny creatures working together",
]

lessons: List[str] = [
    "Courage comes in small actions",
    "Asking for help makes you stronger",
    "Being different is wonderful",
    "Kindness creates unexpected friendships",
    "Practice leads to improvement",
    "Working together achieves more",
    "Honesty is always the best choice",
    "Everyone has hidden talents",
]

ending_surprises: List[str] = [
    "The problem was actually a gift in disguise",
    "They discover they had the solution all along",
    "They make an unexpected new tradition",
    "The helper needed help too",
    "They find something better than what they lost",
    "They teach others what they learned",
    "The scary thing becomes their favorite thing",
    "They realize home was always with them",
]


def generate_story_prompt(
    protagonist: str,
    setting: str,
    problem: str,
    helper: str,
    lesson: str,
    ending_surprise: str,
) -> str:
    """Generate a story prompt based on the template variables."""
    return f"""Write a short story for children aged 5-10 years old with the following elements:

**Story Parameters:**
- Protagonist: {protagonist}
- Setting: {setting}
- Problem: {problem}
- Helper: {helper}
- Lesson: {lesson}
- Ending Surprise: {ending_surprise}

**Story Structure:**

**Opening (50-75 words):**
Introduce the protagonist living in/near the setting. Show their personality through a small action. Present the problem that disrupts their normal day.

**Middle (150-250 words):**
The protagonist attempts to solve the problem but struggles. They encounter the helper who offers guidance or assistance. Show their journey with sensory details about the setting. Include a moment of doubt or difficulty.

**Resolution (75-100 words):**
The protagonist overcomes the problem by applying the lesson. Reveal the ending surprise that adds depth to the story. Show how the character has grown.

**Closing (25-50 words):**
Brief reflection on the lesson in simple, age-appropriate language. End with a hopeful or happy image that children can visualize.

**Writing Guidelines:**
- Use simple, descriptive language with strong verbs
- Include dialogue to make characters feel real
- Add sensory details (what they see, hear, smell, feel)
- Keep sentences varied in length but generally short
- Use repetition for emphasis and rhythm
- Aim for 300-500 words total
- Target reading level: Ages 5-10

Write the complete story now."""


async def generate_story(model: str):
    """Generate short stories by iterating through all combinations of template options."""

    print(f"Models: {[m.model for m in (await client.list()).models]}")

    # Generate all combinations
    all_combinations = list(
        product(protagonists, settings, problems, helpers, lessons, ending_surprises)
    )
    total_stories = len(all_combinations)

    print(
        f"\nGenerating {total_stories} stories by iterating through all combinations..."
    )
    print(f"- {len(protagonists)} protagonists")
    print(f"- {len(settings)} settings")
    print(f"- {len(problems)} problems")
    print(f"- {len(helpers)} helpers")
    print(f"- {len(lessons)} lessons")
    print(f"- {len(ending_surprises)} ending surprises")
    print("=" * 80)

    for story_num, (
        protagonist,
        setting,
        problem,
        helper,
        lesson,
        ending_surprise,
    ) in enumerate(all_combinations, 1):
        print(f"\n### STORY {story_num} of {total_stories} ###\n")
        print("Story Elements:")
        print(f"- Protagonist: {protagonist}")
        print(f"- Setting: {setting}")
        print(f"- Problem: {problem}")
        print(f"- Helper: {helper}")
        print(f"- Lesson: {lesson}")
        print(f"- Surprise: {ending_surprise}")
        print(f"\n{'-' * 80}\n")

        messages = [
            {
                "role": "system",
                "content": """You are a creative children's short story writer specializing in tales for ages 5-10.

Your stories should be:
- Engaging and age-appropriate
- 300-500 words in length
- Written with simple, descriptive language
- Include dialogue and sensory details
- Have a clear beginning, middle, and end
- Teach a valuable lesson naturally through the story
- End with a heartwarming or uplifting conclusion

Focus on creating memorable characters and vivid scenes that young readers can visualize.""",
            },
            {
                "role": "user",
                "content": generate_story_prompt(
                    protagonist, setting, problem, helper, lesson, ending_surprise
                ),
            },
        ]

        # Collect the story output
        story = ""
        async for part in await client.chat(
            model=model, messages=messages, stream=True
        ):
            content = part["message"]["content"]
            story += content
            print(content, end="", flush=True)

        # Save to JSONL
        record = {
            "protagonist": protagonist,
            "setting": setting,
            "problem": problem,
            "helper": helper,
            "lesson": lesson,
            "ending_surprise": ending_surprise,
            "story": story,
            "model": model,
        }
        with open("data/stories.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"\n\n{'=' * 80}\n")


async def main():
    await generate_story(model="granite4:350m")


if __name__ == "__main__":
    asyncio.run(main())
