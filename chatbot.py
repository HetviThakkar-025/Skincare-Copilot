# chatbot.py
# A very basic rule-based skincare chatbot using keyword matching.
# No internet, no AI model: it just looks for known words in the question.

import string
from tips import get_tips, skin_type_summary, DISCLAIMER

# ------------------------------------------------------------
# CONFIG - topics, keywords and replies (edit freely)
# ------------------------------------------------------------
# Keywords are lowercase with no punctuation. A keyword can be several words.
# Order matters only for ties: the topic written first wins.

TOPICS = {
    "acne": {
        "keywords": ["acne", "pimple", "pimples", "breakout", "breakouts", "zit", "zits",
                     "blackheads", "whiteheads"],
        "reply": ("Wash your face twice a day with a gentle cleanser and use oil-free, "
                  "non-comedogenic products. Do not pick or pop pimples, it can leave marks. "
                  "Change your pillowcase often and keep your hands off your face."),
    },
    "dark_circles": {
        "keywords": ["dark circles", "dark circle", "under eye", "undereye", "eye bags",
                     "puffy eyes", "panda eyes"],
        "reply": ("Dark circles are often linked to sleep, screen time, allergies or genetics. "
                  "Try to sleep 7-8 hours, drink enough water and use a gentle eye cream. "
                  "A cold compress in the morning can help with puffiness."),
    },
    "dark_spots": {
        "keywords": ["dark spots", "dark spot", "spots", "pigmentation", "dark patches",
                     "marks", "acne marks", "blemishes", "uneven"],
        "reply": ("Daily sunscreen is the most important step, because sunlight makes dark "
                  "spots darker. Do not pick at spots or scabs. Dark marks usually fade slowly "
                  "over weeks, so be patient and consistent."),
    },
    "sunscreen": {
        "keywords": ["sunscreen", "sunblock", "spf", "sun", "uv", "sunburn", "tanning", "tan"],
        "reply": ("Use a broad-spectrum SPF 30 or higher sunscreen every day, even when it is "
                  "cloudy. Apply it as the last step of your morning routine. "
                  "Reapply every 2-3 hours when you are outdoors."),
    },
    "oily": {
        "keywords": ["oily", "oil", "greasy", "shiny", "shine", "sebum"],
        "reply": ("Use a gentle gel or foaming cleanser twice a day and a light, oil-free "
                  "moisturizer. Do not skip moisturizer, dry skin can make more oil. "
                  "Blotting paper helps with shine during the day."),
    },
    "dry": {
        "keywords": ["dry", "dryness", "flaky", "flaking", "peeling", "rough", "tight",
                     "dehydrated"],
        "reply": ("Use a mild, cream-based cleanser and wash with lukewarm water, not hot water. "
                  "Apply a richer moisturizer right after washing while skin is still damp. "
                  "Exfoliate gently only once or twice a week."),
    },
    "redness": {
        "keywords": ["red", "redness", "irritated", "irritation", "sensitive", "flushed",
                     "blotchy", "inflamed"],
        "reply": ("Choose fragrance-free, gentle products and avoid hot water and harsh scrubs. "
                  "Try new products on a small patch of skin first. "
                  "If redness stays or burns, please see a dermatologist."),
    },
    "routine": {
        "keywords": ["routine", "morning", "night", "evening", "bedtime", "daily", "steps"],
        "reply": ("Morning: cleanser, moisturizer, then sunscreen. "
                  "Night: remove sunscreen or makeup, cleanser, then moisturizer. "
                  "Keep it simple and be consistent for a few weeks before judging results."),
    },
    "skin_tone": {
        "keywords": ["skin tone", "tone", "shade", "complexion", "skin colour", "skin color",
                     "undertone", "fair"],
        "reply": ("Skin tone is estimated from the average lightness of your cheeks. "
                  "Every skin tone is healthy, and every tone needs daily sunscreen. "
                  "Lighting can change the result, so use soft, even light."),
    },
    "lifestyle": {
        "keywords": ["water", "drink", "diet", "food", "eat", "sleep", "hydrate", "hydration",
                     "stress", "exercise", "sugar"],
        "reply": ("Drink enough water through the day and eat plenty of fruits and vegetables. "
                  "Try to sleep 7-8 hours, because skin repairs itself at night. "
                  "Managing stress and exercising also help your skin."),
    },
    # General topics (kept last so skincare topics win a tie)
    "help": {
        "keywords": ["help", "what can you do", "how to use", "features", "options"],
        "reply": ("I can answer simple skincare questions about acne, dark spots, dark circles, "
                  "sunscreen, oily or dry skin, redness, routines and lifestyle. "
                  "After you analyze your skin, ask me \"what about my skin?\" for personal tips."),
    },
    "about": {
        "keywords": ["about you", "about this", "project", "who made", "what is this",
                     "who are you", "how does it work", "how do you work", "opencv",
                     "computer vision"],
        "reply": ("Skincare Copilot is a computer vision project. It finds your face with a "
                  "Haar cascade, crops the forehead and cheeks, and measures tone, shine, "
                  "redness, dark spots and texture using OpenCV. I am a simple keyword chatbot."),
    },
    "greeting": {
        "keywords": ["hi", "hii", "hello", "hey", "good morning", "good evening",
                     "good afternoon", "namaste"],
        "reply": "Hello! I am your skincare assistant. Ask me about your skin or type \"help\".",
    },
    "thanks": {
        "keywords": ["thanks", "thank", "thank you", "thx", "ty"],
        "reply": "You are welcome! Take care of your skin.",
    },
}

# These topics are not about skincare, so a "my skin" question beats them
GENERAL_TOPICS = ["help", "about", "greeting", "thanks"]

# Phrases that mean "tell me about MY analysis"
PERSONAL_PHRASES = ["my skin", "my result", "my results", "my report", "my analysis",
                    "what should i do", "suggest", "suggestion", "suggestions"]

# Words that sound medical: we add the disclaimer when we see them
MEDICAL_WORDS = ["rash", "itch", "itchy", "burn", "burning", "pain", "painful", "infection",
                 "infected", "bleeding", "swelling", "swollen", "allergy", "allergic",
                 "medicine", "medication", "cure", "treatment", "disease", "doctor",
                 "eczema", "psoriasis", "rosacea", "mole", "cancer", "tablet", "pills"]

NO_RESULT_REPLY = ("Please analyze your skin first (upload a photo or use the webcam), "
                   "then ask me again.")

MEDICAL_REPLY = ("I can only give general skincare tips, not medical advice. "
                 "For pain, rashes, infections or anything that worries you, "
                 "please see a dermatologist.")

FALLBACK_REPLY = ("Sorry, I did not understand that. You can ask me things like: "
                  "\"how do I treat acne?\", \"which sunscreen should I use?\", "
                  "\"tips for oily skin\", \"night routine\" or \"what about my skin?\"")

# Number of personal tips to show for "my skin" questions
PERSONAL_TIP_COUNT = 3

# ------------------------------------------------------------


def clean_text(text):
    """Lowercase the text and remove punctuation."""
    text = text.lower()
    # Remove every punctuation character (.,!?' etc.)
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse extra spaces into single spaces
    return " ".join(text.split())


def contains(text, phrase):
    """True if the phrase appears as whole words in the text ('hi' should not match 'this')."""
    # Spaces around both sides make sure we only match whole words
    return (" " + phrase + " ") in (" " + text + " ")


def score_topic(text, keywords):
    """Score a topic: each matching keyword adds its number of words.
    So a longer phrase like 'good morning' (2) beats a single word like 'morning' (1)."""
    score = 0
    for keyword in keywords:
        if contains(text, keyword):
            score += len(keyword.split())
    return score


def find_topic(text):
    """Return the name of the best matching topic, or None if nothing matches."""
    best_topic = None
    best_score = 0
    for name, topic in TOPICS.items():
        score = score_topic(text, topic["keywords"])
        # Strictly greater, so on a tie the topic listed first wins
        if score > best_score:
            best_topic = name
            best_score = score
    return best_topic


def has_any(text, phrases):
    """True if any of the phrases appears in the text."""
    for phrase in phrases:
        if contains(text, phrase):
            return True
    return False


def personal_reply(last_result):
    """Summary plus the first few tips for the user's own analysis."""
    if last_result is None:
        return NO_RESULT_REPLY

    lines = [skin_type_summary(last_result), "Top tips for you:"]
    for tip in get_tips(last_result)[:PERSONAL_TIP_COUNT]:
        lines.append("- " + tip)
    return "\n".join(lines)


def result_line(topic, last_result):
    """One extra line linking the oily / dry / redness answer to the user's result."""
    if last_result is None:
        return None

    oiliness = last_result.get("oiliness", "Unknown")
    texture = last_result.get("texture", "Unknown")
    redness = last_result.get("redness", "Unknown")

    if topic == "oily":
        if oiliness == "Oily":
            return "Your analysis also showed oily skin, so these tips fit you."
        return "Your analysis showed " + oiliness.lower() + " skin, not oily."

    if topic == "dry":
        if oiliness == "Dry" or texture == "Rough":
            return "Your analysis also suggests some dryness, so these tips fit you."
        return "Your analysis did not show clear signs of dryness."

    if topic == "redness":
        return "Your analysis showed " + redness.lower() + " redness."

    return None


def get_reply(user_text, last_result=None):
    """Return the chatbot's reply to the user's message."""
    # Step 1: clean the text
    text = clean_text(user_text)
    if text == "":
        return "Please type a question."

    # Step 2: find the best matching topic
    topic = find_topic(text)
    is_medical = has_any(text, MEDICAL_WORDS)

    # Step 3: personal questions ("my skin", "what should I do", ...)
    # A specific skincare topic wins; otherwise answer from the user's analysis
    if has_any(text, PERSONAL_PHRASES) and (topic is None or topic in GENERAL_TOPICS):
        return personal_reply(last_result)

    # Nothing matched: medical-sounding questions get a safe reply, others get the fallback
    if topic is None:
        if is_medical:
            return MEDICAL_REPLY + "\n" + DISCLAIMER
        return FALLBACK_REPLY

    reply = TOPICS[topic]["reply"]

    # Step 4: link oily / dry / redness answers to the user's result
    extra = result_line(topic, last_result)
    if extra is not None:
        reply += "\n" + extra

    # Step 5: disclaimer for acne and medical-sounding questions
    if topic == "acne" or is_medical:
        reply += "\n" + DISCLAIMER

    return reply


if __name__ == "__main__":
    # A sample analysis result, so "my skin" questions can be tested without a camera
    sample_result = {"tone": "Medium", "oiliness": "Oily", "redness": "Moderate",
                     "spots": 4, "texture": "Normal"}

    # Try these questions:
    #   hello
    #   what about my skin?
    #   what should I do?
    #   how do I get rid of pimples?
    #   my face is very oily
    #   I have dry flaky patches
    #   why are my cheeks red?
    #   which sunscreen should I use
    #   what is a good night routine
    #   good morning
    #   I have an itchy rash
    #   what is this project?
    #   tell me a joke          (fallback)

    print("Skincare chatbot test (sample result:", sample_result, ")")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("You: ")
        if question.strip().lower() == "quit":
            break
        print("Bot:", get_reply(question, sample_result))
        print()
