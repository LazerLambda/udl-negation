"""Unsupervised Deep-Learning Seminar.

Templates File

LMU Munich
Philipp Koch, 2023
MIT-License
"""


NOUN_TEMPLATES: list = [
    "This is {a:word} and not {a:antonym}.",
    "This was {a:word} and not {a:antonym}.",
    "This will be {a:word} and not {a:antonym}.",
    "This has been {a:word} and not {a:antonym}.",
    "This is {a:word}. This is not {a:antonym}.",
    "This is {a:word}. This isn't {a:antonym}.",
    "This was {a:word}. This was not {a:antonym}.",
    "This was {a:word}. This wasn't {a:antonym}.",
    "This will be {a:word}. This will not be {a:antonym}.",
    "This will be {a:word}. This won't be {a:antonym}.",
    "This has been {a:word}. This has not been {a:antonym}.",
    "This has been {a:word}. This hasn't been {a:antonym}.",
    "It is {a:word} and not {a:antonym}.",
    "It was {a:word} and not {a:antonym}.",
    "It will be {a:word} and not {a:antonym}.",
    "It has been {a:word} and not {a:antonym}.",
    "It is {a:word}. It is not {a:antonym}.",
    "It is {a:word}. This is not {a:antonym}.",
    "It is {a:word}. It isn't {a:antonym}.",
    "It is {a:word}. This isn't {a:antonym}.",
    "It was {a:word}. It was not {a:antonym}.",
    "It was {a:word}. This was not {a:antonym}.",
    "It was {a:word}. It wasn't {a:antonym}.",
    "It was {a:word}. This wasn't {a:antonym}.",
    "It will be {a:word}. It will not be {a:antonym}.",
    "It will be {a:word}. This will not be {a:antonym}.",
    "It will be {a:word}. It won't be {a:antonym}.",
    "It will be {a:word}. This won't be {a:antonym}.",
    "It has been {a:word}. It has not been {a:antonym}.",
    "It has been {a:word}. This has not been {a:antonym}.",
    "It has been {a:word}. It hasn't been {a:antonym}.",
    "It has been {a:word}. This hasn't been {a:antonym}."
]

VERB_TEMPLATES: list = [

]

ADJECTIVE_TEMPLATES: list = [
    # Present
    "I am {word}. I am not {antonym}.",
    "You are {word}. You are not {antonym}.",
    "He is {word}. He is not {antonym}.",
    "She is {word}. She is not {antonym}.",
    "It is {word}. It is not {antonym}.",
    "We are {word}. We are not {antonym}.",
    "You are {word}. You are not {antonym}.",
    "They is {word}. They is not {antonym}.",
    # -n't
    "I am {word}. I ain't  {antonym}.",
    "You are {word}. You aren't {antonym}.",
    "He is {word}. He isn't {antonym}.",
    "She is {word}. She isn't {antonym}.",
    "It is {word}. It isn't {antonym}.",
    "We are {word}. We aren't {antonym}.",
    "You are {word}. You aren't {antonym}.",
    "They are {word}. They aren't {antonym}.",
    # Past
    "I was {word}. I was not {antonym}.",
    "You were {word}. You were not {antonym}.",
    "He was {word}. He was not {antonym}.",
    "She was {word}. She was not {antonym}.",
    "It was {word}. It was not {antonym}.",
    "We were {word}. We were not {antonym}.",
    "You were {word}. You were not {antonym}.",
    "They were {word}. They were not {antonym}.",
    # -n't
    "I was {word}. I wasn't {antonym}.",
    "You were {word}. You weren't {antonym}.",
    "He was {word}. He wasn't {antonym}.",
    "She was {word}. She wasn't {antonym}.",
    "It was {word}. It wasn't {antonym}.",
    "We were {word}. We weren't {antonym}.",
    "You were {word}. You weren't {antonym}.",
    "They were {word}. They weren't {antonym}.",
    # Present Perfect
    "I have been {word}. I have not been {antonym}.",
    "You have been {word}. You have not been {antonym}.",
    "He has been {word}. He has not been {antonym}.",
    "She has been {word}. She has not been {antonym}.",
    "It has been {word}. It has not been {antonym}.",
    "We have been {word}. We have not been {antonym}.",
    "You have been {word}. You have not been {antonym}.",
    "They have been {word}. They have not been {antonym}.",
    # -n't
    "I have been {word}. I haven't been {antonym}.",
    "You have been {word}. You haven't been {antonym}.",
    "He has been {word}. He hasn't been {antonym}.",
    "She has been {word}. She hasn't been {antonym}.",
    "It has been {word}. It hasn't been {antonym}.",
    "We have been {word}. We haven't been {antonym}.",
    "You have been {word}. You haven't been {antonym}.",
    "They have been {word}. They haven't been {antonym}.",
    # Future (will)
    "I will be {word}. I will not be {antonym}.",
    "You will be {word}. You will not be {antonym}.",
    "He will be {word}. He will not be {antonym}.",
    "She will be {word}. She will not be {antonym}.",
    "It will be {word}. It will not be {antonym}.",
    "We will be {word}. We will not be {antonym}.",
    "You will be {word}. You will not be {antonym}.",
    "They will be {word}. They will not be {antonym}.",
    # -n't
    "I will be {word}. I won't be {antonym}.",
    "You will be {word}. You won't be {antonym}.",
    "He will be {word}. He won't be {antonym}.",
    "She will be {word}. She won't be {antonym}.",
    "It will be {word}. It won't be {antonym}.",
    "We will be {word}. We won't be {antonym}.",
    "You will be {word}. You won't be {antonym}.",
    "They will be {word}. They won't be {antonym}.",
    # Future (going to)
    "I am going to be {word}. I am not going to be {antonym}.",
    "You are going to be {word}. You are not going to be {antonym}.",
    "He is going to be {word}. He is not going to be {antonym}.",
    "She is going to be {word}. She is not going to be {antonym}.",
    "It is going to be {word}. It is not going to be {antonym}.",
    "We are going to be {word}. We are not going to be {antonym}.",
    "You are going to be {word}. You are not going to be {antonym}.",
    "They are going to be {word}. They are not going to be {antonym}.",
    # -n't
    "I am going to be {word}. I ain't going to be {antonym}.",
    "You are going to be {word}. You aren't going to be {antonym}.",
    "He is going to be {word}. He isn't going to be {antonym}.",
    "She is going to be {word}. She isn't going to be {antonym}.",
    "It is going to be {word}. It isn't going to be {antonym}.",
    "We are going to be {word}. We aren't going to be {antonym}.",
    "You are going to be {word}. You aren't going to be {antonym}.",
    "They are going to be {word}. They aren't not going to be {antonym}.",
]

TEMPLATES: dict = {
    'NOUN': NOUN_TEMPLATES,
    'ADJ': ADJECTIVE_TEMPLATES,
    'VERB': VERB_TEMPLATES
}
