from transformers import pipeline

# zero_shot_classifier = pipeline(
#     "zero-shot-classification",
#     model="facebook/bart-large-mnli"
# )

# Existing model
zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# NEW model 
distilbart_classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1"
)

STANDARD_HEADERS = [
    "Summary", "Experience", "Projects",
    "Trainings", "Certifications",
    "Achievements", "Volunteer"
]

# def build_meta_standard_headers(cleaned_headers):
#     meta_map = {}

#     # convert cleaned_headers to list if it's a string
#     if isinstance(cleaned_headers, str):
#         import ast
#         cleaned_headers = ast.literal_eval(cleaned_headers)

#     for std_label in STANDARD_HEADERS:
#         best_header = ""
#         best_score = 0.0

#         for header in cleaned_headers:
#             result = zero_shot_classifier(
#                 header,
#                 [std_label], 
#                 hypothesis_template="This resume section is about {}.",
#                 multi_label=True
#             )

#             score = result["scores"][0]

#             if score > best_score:
#                 best_score = score
#                 best_header = header

#         meta_map[std_label] = {
#             "header": best_header if best_score > 0 else "",
#             "score": float(best_score)
#         }

#     return meta_map

def build_map_with_model(cleaned_headers, classifier, threshold = 0.9):
    meta_map = {}

    if isinstance(cleaned_headers, str):
        import ast
        cleaned_headers = ast.literal_eval(cleaned_headers)

    for std_label in STANDARD_HEADERS:
        best_header = ""
        best_score = 0.0

        for header in cleaned_headers:
            result = classifier(
                header,
                [std_label],
                hypothesis_template="This resume section is about {}.",
                multi_label=True
            )

            score = result["scores"][0]

            if score > best_score:
                best_score = score
                best_header = header
        
        # apply threshold condition
        if best_score >= threshold:
            meta_map[std_label] = {
                "header": best_header,
                "score": float(best_score)
            }
        else:
            meta_map[std_label] = {
                "header": "",
                "score": ""
            }

    return meta_map