"""Templates File."""

NOUN_TEMPLATES: list= [
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
    "He is {word}. He is not {antonym}.",
    "She is {word}. She is not {antonym}."
    "It is {word}. It is not {antonym}."
    "We are {word}. We are not {antonym}.",
    "You are {word}. You are not {antonym}."
    "They is {word}. They is not {antonym}."
    "He is {word}. He isn't {antonym}.",
    "She is {word}. She isn't {antonym}."
    "It is {word}. It isn't {antonym}."
    "We are {word}. They aren't {antonym}.",
    "You are {word}. You aren't {antonym}."
    "They are {word}. They aren't {antonym}."
    # Past
    "He was {word}. He was not {antonym}.",
    "She was {word}. She was not {antonym}."
    "It was {word}. It was not {antonym}.",
    "He was {word}. He wasn't {antonym}.",
    "She was {word}. She wasn't {antonym}.",
    "It was {word}. It wasn't {antonym}.",
    # Present Perfect
    "He has been {word}. He has not been {antonym}.",
    "She has been {word}. She has not been {antonym}."
    "It has been {word}. It has not been {antonym}."
    "He has been {word}. He has not been {antonym}.",
    "She has been {word}. She has not been {antonym}."
    "It has been {word}. It has not been {antonym}."
    # Future
    "He will be {word}. He will not be {antonym}.",
    "She will be {word}. She will not be {antonym}.",
    "It will be {word}. It will not be {antonym}.",
    "He will be {word}. He won't be {antonym}.",
    "She will be {word}. She won't be {antonym}."
    "It will be {word}. It won't be {antonym}."
]

TEMPLATES: dict = {
    'NOUN': NOUN_TEMPLATES,
    'ADJ': ADJECTIVE_TEMPLATES,
    'VERBS': VERB_TEMPLATES
}