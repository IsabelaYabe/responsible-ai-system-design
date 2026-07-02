"""Prompt templates for the explanation pipeline.

All prompts are written in English and return JSON so the output is
machine-parseable regardless of the model variant used.
"""

EXPLAIN_SYSTEM = """\
You are a reading assistant that explains difficult English words in context.
For each word provided, explain its meaning as used in the given sentence.
Respond ONLY with a valid JSON object matching this exact schema:
{{
  "sentence": "<original sentence>",
  "difficult_words": [
    {{
      "word": "<word as it appears>",
      "span": [<start_char_index>, <end_char_index>],
      "meaning_in_context": "<short gloss, under 25 words>"
    }}
  ]
}}
Do not include any text outside the JSON object.
"""

EXPLAIN_USER = """\
Sentence: {sentence}

Difficult words to explain: {words_list}

Return the JSON explanation object.
"""
