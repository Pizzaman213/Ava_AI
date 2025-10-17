#!/usr/bin/env python3
"""Generate 10,000 simple fairy tale examples for training."""

import random
from pathlib import Path

# Story templates and components
characters = [
    "a little girl", "a brave knight", "a wise old owl", "a young prince", "a kind princess",
    "a lonely dragon", "a clever fox", "a magical fairy", "a humble farmer", "a brave explorer",
    "a kind baker", "a talented musician", "a curious child", "a gentle giant", "a smart rabbit",
    "a beautiful mermaid", "a powerful wizard", "a young shepherd", "a clever mouse", "a friendly wolf",
    "a brave sailor", "a kind teacher", "a small bird", "a wise turtle", "a playful cat",
    "a strong bear", "a gentle deer", "a clever squirrel", "a loyal dog", "a wise old woman",
    "a young artist", "a brave hunter", "a kind healer", "a skillful craftsman", "a curious scientist",
    "a talented dancer", "a brave warrior", "a gentle monk", "a wise elder", "a young inventor",
    "a kind gardener", "a clever merchant", "a brave adventurer", "a gentle nurse", "a wise librarian",
    "a young writer", "a brave firefighter", "a kind doctor", "a clever detective", "a gentle teacher"
]

settings = [
    "in a small village", "in a magical kingdom", "in a deep forest", "by the sea", "in a tall castle",
    "on a high mountain", "in a peaceful valley", "near a crystal lake", "in an enchanted garden", "in a cozy cottage",
    "in a busy town", "on a distant island", "in a hidden cave", "in a grand palace", "in a quiet meadow",
    "by a rushing river", "in a snowy land", "in a desert oasis", "in a floating city", "in an ancient temple",
    "in a mystic grove", "by a magic fountain", "in a secret library", "in a cloud palace", "in a sunlit glade",
    "in a moonlit clearing", "by a rainbow bridge", "in a starlit field", "in a golden city", "in a silver tower",
    "by a waterfall", "in a bamboo forest", "in a rose garden", "on a flying ship", "in a tree house",
    "in a lighthouse", "by a wishing well", "in a clocktower", "in a music hall", "in a theater"
]

magical_items = [
    "a magical flower", "a golden key", "a silver mirror", "a glowing crystal", "an ancient book",
    "a magic wand", "a enchanted sword", "a flying carpet", "a talking animal", "a magic ring",
    "a crystal ball", "a magic lamp", "a enchanted crown", "a invisible cloak", "a time-traveling watch",
    "a healing potion", "a magic flute", "a enchanted painting", "a flying broomstick", "a magic compass",
    "a singing harp", "a dancing shoes", "a wish-granting stone", "a truth-telling mirror", "a fortune cookie",
    "a magic feather", "a enchanted necklace", "a glowing gem", "a magic bell", "a enchanted staff",
    "a flying horse", "a magic door", "a enchanted tree", "a talking mirror", "a magic fountain",
    "a crystal sword", "a golden apple", "a silver flute", "a magic rose", "a enchanted pearl"
]

actions = [
    "went on a quest", "found a treasure", "helped a friend", "solved a mystery", "discovered a secret",
    "learned a lesson", "saved the day", "made a wish", "broke a curse", "found true love",
    "defeated evil", "protected the innocent", "learned magic", "found courage", "discovered wisdom",
    "showed kindness", "gained strength", "found peace", "spread joy", "brought hope",
    "restored balance", "healed the sick", "fed the hungry", "taught others", "built a home",
    "planted a garden", "wrote a song", "painted a picture", "told a story", "danced in joy",
    "sang beautifully", "climbed a mountain", "crossed a bridge", "opened a door", "lit a candle",
    "followed a star", "chased a dream", "kept a promise", "shared a gift", "gave forgiveness"
]

outcomes = [
    "and lived happily ever after", "and brought peace to the land", "and everyone rejoiced",
    "and learned that kindness matters most", "and discovered true happiness", "and found inner peace",
    "and the kingdom prospered", "and love conquered all", "and good triumphed over evil",
    "and wisdom prevailed", "and courage was rewarded", "and friendship saved the day",
    "and hope was restored", "and joy returned to the land", "and the curse was broken",
    "and magic filled the air", "and dreams came true", "and light defeated darkness",
    "and truth was revealed", "and justice was served", "and harmony was restored",
    "and the future was bright", "and all was well again", "and peace reigned supreme",
    "and love bloomed eternal", "and the adventure continued", "and legends were born",
    "and stories were told for generations", "and the world became better", "and hope never died",
    "and kindness spread everywhere", "and laughter filled the air", "and songs were sung",
    "and dances were danced", "and feasts were held", "and celebrations lasted for days",
    "and gratitude filled all hearts", "and blessings multiplied", "and miracles happened daily",
    "and everyone learned to care for each other"
]

lessons = [
    "Courage comes from within", "Kindness is the greatest magic", "True friends are priceless",
    "Wisdom grows with patience", "Love conquers all fears", "Hope lights the darkest paths",
    "Truth always prevails", "Compassion heals all wounds", "Faith moves mountains",
    "Generosity brings abundance", "Forgiveness sets you free", "Gratitude opens doors",
    "Honesty builds trust", "Humility brings wisdom", "Patience yields rewards",
    "Perseverance leads to success", "Respect earns respect", "Sacrifice brings honor",
    "Understanding bridges gaps", "Unity creates strength", "Determination breaks barriers",
    "Creativity solves problems", "Curiosity leads to discovery", "Empathy builds connections",
    "Integrity guides action", "Justice brings peace", "Knowledge empowers", "Loyalty endures",
    "Mercy shows strength", "Optimism attracts good fortune"
]

def generate_story():
    """Generate a random fairy tale story."""
    char = random.choice(characters)
    setting = random.choice(settings)
    item = random.choice(magical_items)
    action = random.choice(actions)
    outcome = random.choice(outcomes)
    lesson = random.choice(lessons)

    # Generate story with slight variations in structure
    templates = [
        f"Once upon a time, there was {char} who lived {setting}. One day, they found {item} and {action}. In the end, they {outcome[4:]}. The moral of the story is: {lesson}.",
        f"Once upon a time, {char} lived {setting}. They discovered {item} which helped them as they {action}. Eventually, {outcome[4:]} and everyone learned that {lesson.lower()}.",
        f"Once upon a time, in a land far away, {char} dwelled {setting}. Their adventure began when they encountered {item}. They {action} with great determination, {outcome[4:]}. Remember: {lesson}.",
        f"Once upon a time, {char} made their home {setting}. Through {item}, they {action} and showed everyone that {lesson.lower()}. Finally, {outcome[4:]}.",
        f"Once upon a time, there lived {char} {setting}. When they found {item}, everything changed. They {action} bravely, {outcome[4:]}, proving that {lesson.lower()}.",
    ]

    return random.choice(templates)

def main():
    """Generate 10,000 fairy tale examples."""
    output_path = Path(__file__).parent.parent.parent / "data" / "raw" / "simple_fairy_tales.txt"

    print(f"Generating 10,000 fairy tale examples...")

    stories = []
    for i in range(10000):
        stories.append(generate_story())
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1} stories...")

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(stories))

    print(f"\nSuccessfully generated 10,000 stories!")
    print(f"Saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
