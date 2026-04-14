from action_analysis import extract_first_words, analyze_first_words

# Sample bullets (realistic cases)
bullets = [
    "Implemented a scalable pipeline for data ingestion",
    "Implementing new ML models for prediction",
    "Improves system performance by 30%",
    "Developing dashboards for analytics",
    "with a team of 5 engineers",
    "Led cross-functional collaboration",
]

print("\n📌 Step 1: Extracting first words...\n")

first_words = extract_first_words(bullets)
print(f"👉 First Words: {first_words}")


print("\n📌 Step 2: Analyzing action words + tense...\n")

results = analyze_first_words(first_words)


print("\n📊 Final Analysis:\n")

for idx, res in enumerate(results, 1):
    print(f"[{idx}] Word: {res['original']}")
    print(f"     Lemma: {res['lemma']}")
    print(f"     Is Action Word: {res['is_action']}")
    print(f"     Tense: {res['tense']}")
    print(f"     Needs Fix: {res['needs_fix']}")
    
    if res["suggested"]:
        print(f"     Suggested Fix: {res['suggested']}")

    print("-" * 50)