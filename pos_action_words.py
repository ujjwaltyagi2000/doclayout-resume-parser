

import spacy

nlp = spacy.load("en_core_web_sm")
from spacy.symbols import POS

def get_action_words_spacy(text):
    doc = nlp(text)
    actions = []

    for token in doc:
        if token.pos_ == "VERB":
            actions.append(token.lemma_.lower())

    return list(set(actions)), len(set(actions))

def print_all_pos(text):
    doc = nlp(text)

    pos_set = set()
    for token in doc:
        pos_set.add(token.pos_)

    print("POS found in text:")
    print(pos_set)

if __name__ == "__main__":
    text = """
    
    Extracted data from Adobe Analytics & Digital 360, enabling strategic insights & improving global digital engagement
• Converted complex datasets into actionable insights, delivering customized strategy presentations to senior
stakeholders, driving 30% improvements in visibility into business trends
• Collaborated with cross-functional teams and identified operational gaps, proposing data-driven solutions,
contributing to 18% growth in client campaigns
• Conducted root-cause analysis for operational anomalies, collaborating with teams, implementing corrective actions,
leading to a 35% enhancement in overall data integrity
• Designed executive-ready presentation decks, aligning insights with business objectives and client management
expectations, enhancing stakeholder understanding
Business Development Intern: Advantage Club Jan 2023 – Jun 2023
• Collaborated with a 5-member team, targeted 50+ corporates, and acquired and onboarded clients within 3 months
• Led digitalization, customization, and integration of the company platform for 4 clients, generating revenue of
approximately INR 35L and exceeding half-yearly targets


"""
    actions, count = get_action_words_spacy(text)
    print("Action Words:", actions)
    print("Total Action Words:", count)

    # nlp = spacy.load("en_core_web_sm")

    # # Get all POS tag IDs
    # pos_ids = list(POS)
    # print(POS)

    # # Convert IDs to readable string names
    # pos_tags = [nlp.vocab.strings[pos] for pos in pos_ids]

    # print("All POS tags from spaCy:")
    # print(pos_tags)