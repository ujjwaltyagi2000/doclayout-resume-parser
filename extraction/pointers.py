def extract_first_words(bullets):
    first_words = []

    for bullet in bullets:
        if not bullet:
            continue

        # split by space and take first word
        first_word = bullet.strip().split()[0]

        first_words.append(first_word)

    return first_words