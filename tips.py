# tips.py
# Simple if/else rules that turn the skin analysis result into skincare tips.

# ------------------------------------------------------------
# CONFIG - rule boundaries and all tip text (edit freely)
# ------------------------------------------------------------

# Spot count at or above this is treated as "many dark spots"
MANY_SPOTS_COUNT = 8

DISCLAIMER = ("This tool gives general tips from image colour and brightness only. "
              "It is not a medical diagnosis. Please see a dermatologist for skin problems.")

# Tips for each oiliness label
OILINESS_TIPS = {
    "Oily": [
        "Wash your face with a gentle gel or foaming cleanser.",
        "Use a light, oil-free moisturizer instead of skipping moisturizer.",
        "Blot shine during the day with blotting paper instead of washing again.",
        "Choose products labelled 'non-comedogenic' so they do not clog pores.",
    ],
    "Dry": [
        "Use a mild, cream-based cleanser that does not leave skin tight.",
        "Apply a richer moisturizer right after washing, while skin is still damp.",
        "Wash with lukewarm water, not hot water.",
    ],
    "Normal": [
        "Your oil balance looks good, keep a simple cleanse and moisturize routine.",
        "Use a light moisturizer to keep your skin balanced.",
    ],
}

# Tips for each redness label (no tips for "Low")
REDNESS_TIPS = {
    "Moderate": [
        "Choose fragrance-free, gentle products to avoid irritating your skin.",
        "Avoid scrubbing your face hard or using rough towels.",
    ],
    "High": [
        "Choose fragrance-free, gentle products to avoid irritating your skin.",
        "Avoid harsh scrubs, very hot water and new products until the redness settles.",
        "If the redness stays, burns or itches, please see a dermatologist.",
    ],
}

# Tips when many dark spots are found
SPOTS_TIPS = [
    "Sunlight makes dark spots darker, so wear a hat or stay in the shade at midday.",
    "Try not to pick or squeeze spots, it can leave marks.",
    "Be patient: dark marks usually fade slowly over weeks.",
]

# Tips for each texture label (no tips for "Smooth" or "Normal")
TEXTURE_TIPS = {
    "Rough": [
        "Rough texture can mean dryness, moisturize twice a day.",
        "Use a gentle exfoliating product only once or twice a week, not daily.",
        "Use a humidifier or drink more water if the air is dry.",
    ],
}

# One sun protection tip for each skin tone
TONE_TIPS = {
    "Fair": "Fair skin burns quickly, use a broad-spectrum SPF 30+ sunscreen and reapply outdoors.",
    "Medium": "Use a broad-spectrum SPF 30+ sunscreen daily to prevent tanning and uneven tone.",
    "Tan": "Sunscreen still matters for tan skin, it helps prevent dark patches and uneven tone.",
    "Deep": "Deep skin also needs sunscreen, it helps prevent dark patches and sun damage.",
}

# Used instead of the tone tip when the tone is unknown
GENERAL_SUNSCREEN_TIP = "Wear sunscreen every day, even when it is cloudy."

# Tips everyone gets
GENERAL_TIPS = [
    "Clean your face twice a day, morning and night.",
    "Drink enough water through the day.",
    "Try to get 7-8 hours of sleep.",
]

# ------------------------------------------------------------


def add_tips(tip_list, new_tips):
    """Add tips to the list, skipping any that are already in it."""
    for tip in new_tips:
        if tip not in tip_list:
            tip_list.append(tip)


def get_tips(result):
    """Turn an analysis result dictionary into a list of short tips."""
    tips = []

    # Oiliness: oily / dry / normal
    oiliness = result.get("oiliness")
    if oiliness in OILINESS_TIPS:
        add_tips(tips, OILINESS_TIPS[oiliness])

    # Redness: only moderate or high gives tips
    redness = result.get("redness")
    if redness in REDNESS_TIPS:
        add_tips(tips, REDNESS_TIPS[redness])

    # Dark spots: only when there are many
    if result.get("spots", 0) >= MANY_SPOTS_COUNT:
        add_tips(tips, SPOTS_TIPS)

    # Texture: rough suggests dryness
    texture = result.get("texture")
    if texture in TEXTURE_TIPS:
        add_tips(tips, TEXTURE_TIPS[texture])

    # One sun protection tip based on skin tone (or a general one if tone is unknown)
    tone = result.get("tone")
    if tone in TONE_TIPS:
        add_tips(tips, [TONE_TIPS[tone]])
    else:
        add_tips(tips, [GENERAL_SUNSCREEN_TIP])

    # General habits for everyone
    add_tips(tips, GENERAL_TIPS)
    return tips


def join_words(words):
    """Join words like: 'a', 'a and b', 'a, b and c'."""
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " and " + words[-1]


def skin_type_summary(result):
    """Return one line describing the skin, e.g. 'Your skin looks oily with moderate redness.'"""
    oiliness = result.get("oiliness", "Unknown")
    if oiliness == "Unknown":
        return "The skin could not be analysed clearly. Try a clearer, well-lit photo."

    # Collect any extra concerns to mention after "with"
    concerns = []
    redness = result.get("redness")
    if redness in ("Moderate", "High"):
        concerns.append(redness.lower() + " redness")
    if result.get("spots", 0) >= MANY_SPOTS_COUNT:
        concerns.append("many dark spots")
    if result.get("texture") == "Rough":
        concerns.append("a rough texture")

    summary = "Your skin looks " + oiliness.lower()
    if len(concerns) > 0:
        summary += " with " + join_words(concerns)
    return summary + "."


if __name__ == "__main__":
    # Three sample results to check the rules without a camera
    samples = [
        {"tone": "Medium", "oiliness": "Oily", "redness": "Low", "spots": 10, "texture": "Normal"},
        {"tone": "Fair", "oiliness": "Dry", "redness": "Low", "spots": 2, "texture": "Rough"},
        {"tone": "Deep", "oiliness": "Normal", "redness": "Moderate", "spots": 3, "texture": "Smooth"},
    ]

    for number, sample in enumerate(samples, start=1):
        print("===== Sample", number, "=====")
        print(skin_type_summary(sample))
        for tip in get_tips(sample):
            print(" -", tip)
        print()

    print("Note:", DISCLAIMER)
