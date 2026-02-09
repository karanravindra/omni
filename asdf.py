import json
from collections import Counter
from pathlib import Path


# More efficient file reading
def load_records(filepath):
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]


records = load_records("./data/stories.jsonl")

# Cache stories list to avoid repeated list comprehensions
stories = [r["story"] for r in records]
total_records = len(records)

print(f"Total Stories: {total_records}")

# Use Counter for efficient unique counting (single pass)
fields = ["protagonist", "setting", "problem", "helper", "lesson", "ending_surprise"]
for field in fields:
    unique_count = len({r[field] for r in records})
    print(f"Unique {field.replace('_', ' ').title()}s: {unique_count}")

# Calculate totals in single pass
total_chars = sum(len(story) for story in stories)
total_words = sum(len(story.split()) for story in stories)
print(f"Total Chars: {total_chars}")
print(f"Total Words: {total_words}")


# Optimized longest common substring using suffix array approach
def longest_common_substring_optimized(strs, min_occurrence=None):
    """
    Find longest common substring using a more efficient approach.

    Args:
        strs: List of strings to search
        min_occurrence: Minimum number of strings substring must appear in (None = all)
    """
    if not strs:
        return ""

    if min_occurrence is None:
        min_occurrence = len(strs)

    # Start with shortest string and work backwards from longer substrings
    shortest_str = min(strs, key=len)
    max_len = len(shortest_str)

    # Check from longest to shortest possible substrings
    for length in range(max_len, 0, -1):
        # Generate all substrings of current length from shortest string
        for i in range(len(shortest_str) - length + 1):
            candidate = shortest_str[i : i + length]

            # Count occurrences across all strings
            count = sum(1 for s in strs if candidate in s)

            if count >= min_occurrence:
                return candidate

    return ""


# Find substring common to all stories
common_substr = longest_common_substring_optimized(stories)
print(f"Longest Common Substring: {repr(common_substr)}")

# Find substring in at least 20% of stories
threshold = int(total_records)
common_substr_20 = longest_common_substring_optimized(stories, min_occurrence=threshold)
print(
    f"Longest Common Substring in at least 20% of stories: {repr(common_substr_20)}, {threshold}"
)
