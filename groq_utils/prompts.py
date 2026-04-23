# ✅ CURRENT WORKING PROMPT FOR CLEANING HEADERS
cleaned_headers_prompt_template = """
    Extract valid resume section headers from this list: {headers}

    Each List Item contains three fields: 
    1. Header Text
    3. Font Size
    2. Y co-ordinate (within document)

    Rules:
    - Keep only standard resume sections (e.g., Education, Experience, Skills, Projects, etc.)
    - Remove names, dates, company names, and project details
    - Remove duplicates
    - Return ONLY a Python list, nothing else

    IMPORTANT: Your entire response must be ONLY the list in this exact format:
    ['Header1', 'Header2', 'Header3']

    Do not include explanations, code, markdown, or any other text.
    """

# # TEST PROMPT 1: ❌ Gibberish values de raha but at least it's from the input, not making up new ones.
# cleaned_headers_prompt_template ="""
# Extract valid resume section headers from this list: {headers}
#     Each List Item contains three fields: 
#     1. Header Text
#     3. Font Size
#     2. Y co-ordinate (within document)

#     Rules:

# 1) Use font size + heading level as primary signals:
#     - Larger font sizes and lower levels (e.g., L1) are more likely to be real section headers.
#     - Smaller font sizes and deeper levels (e.g., L3/L4) are often subheadings, roles, companies, projects, or details — exclude them.
#     - Suppose 3/4 font sizes look to be section headings in that case, we can consider all similar font sizes to be section headinds
# 2) Remove duplicates
# 3) Return ONLY a Python list, nothing else
# 4) Important Remove:
#     - remove Person name (any name-like header)
#     - Remove names, dates, company names, and project details
#     - Company names, job titles, project titles
#     - Dates, locations, degree/program lines (e.g., “MBA Finance…”, “Apr 2022 - Sep 2023”)
#     - Anything that looks like content rather than a section label
# 6) Output must preserve the header text EXACTLY as it appears in the input (same casing/spelling).
#     - Do NOT rewrite, normalize, or expand headers.
#     - Do NOT invent new headers.
# 7) Important to remeber Return null if you beleive no valid result as output of section heading, since you are a resume expert


#     IMPORTANT: Your entire response must be ONLY the list in this exact format:
#     ['Header1', 'Header2', 'Header3']
#     Do not include explanations, code, markdown, or any other text.
# If you don't find any valid headers, return an empty list. Do not give any headers that do not really exist.


# """

# # 🧪 TEST PROMPT 2: ❌  Ekdum hi bekaar hai
# cleaned_headers_prompt_template = """
# Extract valid resume section headers from this list: {headers}

# Each list item contains:
# 1) Header Text
# 2) Font Size
# 3) Y-coordinate

# Strict Rules (must follow ALL):

# 1) Only select headers that are CLEAR section titles such as:
#    (e.g., EXPERIENCE, EDUCATION, SKILLS, PROJECTS, SUMMARY, CERTIFICATIONS, etc.)

# 2) Font-based filtering (VERY IMPORTANT):
#    - Only consider the LARGEST or near-largest font sizes in the document.
#    - If a text is not among the top font sizes, IGNORE it completely.
#    - Small font text (company names, roles, dates, etc.) must NEVER be selected.

# 3) You MUST NOT guess or infer headers:
#    - If no clear section headers exist → return []
#    - Do NOT create, assume, or infer missing headers

# 4) Always REMOVE:
#    - Person names
#    - Company names
#    - Job titles
#    - Project titles
#    - Dates, locations
#    - Degree/program lines
#    - Any descriptive/content text

# 5) Deduplicate headers (exact match only)

# 6) Output constraints (CRITICAL):
#    - Return ONLY a Python list
#    - No explanation, no extra text
#    - Preserve header text EXACTLY as in input
#    - Do NOT modify casing or wording

# 7) Failure condition (VERY IMPORTANT):
#    - If you are NOT fully confident that an item is a section header → EXCLUDE it
#    - If no valid headers remain → return []

# Output format:
# ['Header1', 'Header2']

# """

# # 🧪 TEST PROMPT 3: ❌ Same issue as above
# cleaned_headers_prompt_template = """
# Extract valid resume section headers from this list: {headers}

# Each list item has three fields:
# 1. Header Text
# 2. Font Size
# 3. Y coordinate (within document)

# RULES:

# 1. FONT SIZE SIGNAL:
#    - Identify the 1-2 most common "section header" font sizes by looking for mid-to-large fonts used repeatedly.
#    - If 3-4 similar font sizes appear to be section headings, treat all of them as candidates.
#    - Small fonts (e.g., ≤8) are almost never section headers — skip them.

# 2. VALID section headers look like:
#    - All-caps labels: EDUCATION, EXPERIENCE, SKILLS, CERTIFICATIONS, PROJECTS, SUMMARY, etc.
#    - Title-case labels: Work Experience, Technical Skills, etc.
#    - They are SHORT (1-4 words max), generic, and label a section of a resume.

# 3. REMOVE all of the following — even if they appear in the input:
#    - Person's name (first item, usually largest font)
#    - Company names, employer names, organization names (e.g., "ATLASSIAN CORPORATION PLC", "LEARNX.", "TECHNOLOGIES")
#    - Job titles, project titles, degree/program names
#    - Dates, locations, cities, countries
#    - Partial words or broken fragments of a company/product name
#    - Anything that is content *within* a section, not a section label

# 4. NO HALLUCINATION:
#    - Only return headers that EXIST VERBATIM in the input list.
#    - Do NOT invent, infer, merge, or rewrite any header.
#    - Do NOT return headers that are not in the input.

# 5. If NO valid section headers exist in the input, return EXACTLY: []
#    Do not return anything else. Do not guess. Do not fill with company names or fragments.

# 6. Preserve exact casing and spelling from the input. Do not normalize.

# OUTPUT FORMAT:
# Your entire response must be ONLY a Python list, like:
# ['Header1', 'Header2']
# Or if nothing valid:
# []
# No explanation. No markdown. No code fences. Just the list.
# """

# # 🧪 TEST PROMPT 4: ❌ Khud se headers bana diye
# cleaned_headers_prompt_template = """
# Extract valid resume section headers from this list: {headers}

# Each list item has:
# 1) Header Text
# 2) Font Size
# 3) Y coordinate

# STRICT RULES (ALL mandatory):

# 1) ONLY accept headers that match COMMON resume section categories such as:
#    EXPERIENCE, WORK EXPERIENCE, PROFESSIONAL EXPERIENCE,
#    EDUCATION, SKILLS, TECHNICAL SKILLS,
#    PROJECTS, CERTIFICATIONS, SUMMARY, PROFILE, ACHIEVEMENTS

#    - If a header does NOT clearly match a known resume section category → REJECT it

# 2) FONT FILTER (HARD RULE):
#    - Consider ONLY top 2–3 largest font sizes in the entire list
#    - Ignore everything else
#    - If no valid headers found in these sizes → return []

# 3) HARD REJECTION (VERY IMPORTANT):
#    Immediately reject anything that looks like:
#    - Company names (e.g., contains INC, LTD, LLC, CORPORATION, TECHNOLOGIES, IO, etc.)
#    - Person names (usually first line, largest font)
#    - Job titles (e.g., Software Engineer, Intern)
#    - Project/product names
#    - Degree/program text
#    - Dates or locations
#    - Any multi-word phrase that is not a generic section label

# 4) NO GUESSING / NO FALLBACK:
#    - Do NOT try to "best match"
#    - Do NOT return "close enough" results
#    - If unsure → EXCLUDE

# 5) CONFIDENCE RULE (CRITICAL):
#    - Only include a header if you are 100% certain it is a section header
#    - If confidence is not absolute → DROP it

# 6) NO HALLUCINATION:
#    - Only return exact text present in input
#    - Do NOT modify, rephrase, or invent

# 7) EMPTY CASE (STRICT):
#    - If after filtering nothing remains → return EXACTLY:
#      []

# OUTPUT FORMAT:
# Return ONLY a Python list:
# ['Header1', 'Header2']
# OR:
# []
# No explanation. No extra text.
# """

# # 🧪 TEST PROMPT 5: ❌
# cleaned_headers_prompt_template = """
# You are a high-precision resume section heading detector.

# Task:
# Given a list of extracted text items from a single resume page, identify only the items that are true resume section headings.



# Each input item contains:
# - text
# - font
# - y_position

# Goal:
# Return only valid resume section headings.
# If no valid section headings are detected, return null.

# Strict rules:
# 1. A valid section heading must be a true resume section label such as:
#    Summary, Professional Summary, Experience, Work Experience, Professional Experience, Education, Skills, Technical Skills, Projects, Certifications, Awards, Publications, Languages, Interests, Volunteer Experience, Leadership, Profile, Core Competencies
# 2. Also allow strong non-standard headings only if they clearly function as section titles.
# 3. Exclude:
#    - candidate name
#    - company names
#    - job titles
#    - degree names
#    - certification names
#    - contact info
#    - addresses
#    - dates
#    - bullets
#    - skill lists
#    - body text
#    - random large text
# 4. Do not guess.
# 5. Do not invent headings not present in the input.
# 6. Use semantics first, then font and y_position as supporting evidence.
# 7. Font prominence alone is not enough.
# 8. If an item is ambiguous, exclude it.
# 9. Prioritize precision over recall.

# Reasoning guidance:
# - A true heading is usually short and acts as a section label.
# - Larger font may help, but does not make something a heading by itself.
# - y_position helps determine section order and whether text starts a new block.
# - Company names, names of people, and isolated uppercase words are not section headings unless they clearly function as a section title.

# Input:
# A JSON array of items like:
# [
#   {"text":"Experience","font":12,"y_position":640},
#   {"text":"Google","font":10,"y_position":620}
# ]

# Output rules:
# - Return comma-separated section headings only.
# - Preserve exact input text.
# - No JSON.
# - No explanation.
# - No extra words.
# - If no valid section headings are found, return exactly:
# null

# Examples:

# Example 1
# Input:
# [
#   {"text":"EXPERIENCE","font":12,"y_position":700},
#   {"text":"Google","font":10,"y_position":680},
#   {"text":"EDUCATION","font":12,"y_position":500}
# ]
# Output:
# EXPERIENCE, EDUCATION

# Example 2
# Input:
# [
#   {"text":"Alisha Chakraborty","font":17,"y_position":752.6},
#   {"text":"ATLASSIAN CORPORATION PLC","font":7,"y_position":643.8},
#   {"text":"●","font":11,"y_position":618.6},
#   {"text":"TECHNOLOGIES","font":7,"y_position":481.8}
# ]
# Output:
# null
# """

# # TEST PROMPT 6: 
# """
# {headers}

# You are a high-precision resume section heading detector.

# Task:
# Given a list of extracted text items from a single resume page, identify only the items that are true resume section headings.

# Each input item contains:
# - text
# - font
# - y_position

# Goal:
# Return only valid resume section headings.
# If no valid section headings are detected, return null.

# Strict rules:
# 1. A valid section heading must be a true resume section label such as:
#    Summary, Professional Summary, Experience, Work Experience, Professional Experience, Education, Skills, Technical Skills, Projects, Certifications, Awards, Publications, Languages, Interests, Volunteer Experience, Leadership, Profile, Core Competencies
# 2. Also allow strong non-standard headings only if they clearly function as section titles.
# 3. Exclude:
#    - candidate name
#    - company names
#    - job titles
#    - degree names
#    - certification names
#    - contact info
#    - addresses
#    - dates
#    - bullets
#    - skill lists
#    - body text
#    - random large text
# 4. Do not guess.
# 5. Do not invent headings not present in the input.
# 6. Use semantics first, then font and y_position as supporting evidence.
# 7. Font prominence alone is not enough.
# 8. If an item is ambiguous, exclude it.
# 9. Prioritize precision over recall.

# Reasoning guidance:
# - A true heading is usually short and acts as a section label.
# - Larger font may help, but does not make something a heading by itself.
# - y_position helps determine section order and whether text starts a new block.
# - Company names, names of people, and isolated uppercase words are not section headings unless they clearly function as a section title.

# Input:
# A JSON array of items like:
# [
#   {"text":"Experience","font":12,"y_position":640},
#   {"text":"Google","font":10,"y_position":620}
# ]

# Output rules:
# - Return comma-separated section headings only.
# - Preserve exact input text.
# - No JSON.
# - No explanation.
# - No extra words.
# - If no valid section headings are found, return exactly:
# null

# Examples:

# Example 1
# Input:
# [
#   {"text":"EXPERIENCE","font":12,"y_position":700},
#   {"text":"Google","font":10,"y_position":680},
#   {"text":"EDUCATION","font":12,"y_position":500}
# ]
# Output:
# EXPERIENCE, EDUCATION

# Example 2
# Input:
# [
#   {"text":"Alisha Chakraborty","font":17,"y_position":752.6},
#   {"text":"ATLASSIAN CORPORATION PLC","font":7,"y_position":643.8},
#   {"text":"●","font":11,"y_position":618.6},
#   {"text":"TECHNOLOGIES","font":7,"y_position":481.8}
# ]
# Output:
# null
# """

# # TEST PROMPT 7: ✅ Worked for edge cases
# cleaned_headers_prompt_template = """
# Extract valid resume section headers from this list: {headers}

#     Each List Item contains three fields: 
#     1. Header Text
#     3. Font Size
#     2. Y co-ordinate (within document)

#     Rules:

# 1) Use font size + heading level as primary signals:
#     - Larger font sizes and lower levels (e.g., L1) are more likely to be real section headers.
#     - Smaller font sizes and deeper levels (e.g., L3/L4) are often subheadings, roles, companies, projects, or details — exclude them.
#     - Suppose 3/4 font sizes look to be section headings in that case, we can consider all similar font sizes to be section headinds
# 2) Remove duplicates
# 3) Return ONLY a Python list, nothing else
# 4) Important Remove:
#     - remove Person name (any name-like header)
#     - Remove names, dates, company names, and project details
#     - Company names, job titles, project titles
#     - Dates, locations, degree/program lines (e.g., “MBA Finance…”, “Apr 2022 - Sep 2023”)
#     - Anything that looks like content rather than a section label
# 6) Output must preserve the header text EXACTLY as it appears in the input (same casing/spelling).
#     - Do NOT rewrite, normalize, or expand headers.
#     - Do NOT invent new headers.
# 7) Important to remeber Return null if you beleive no valid result as output of section heading, since you are a resume expert


#     IMPORTANT: Your entire response must be ONLY the list in this exact format:
#     ['Header1', 'Header2', 'Header3']
#     Do not include explanations, code, markdown, or any other text.
# If you don't find any valid headers, return an empty list. Do not give any headers that do not really exist.

# CRITICAL ANTI-HALLUCINATION RULE: You may ONLY return headers that are verbatim present in the input list above.
# If every item in the input is a name, date, company, job title, or other non-resume section content, you MUST return [].
# Do NOT invent, infer, or generate any header that does not appear word-for-word in the input. When in doubt, return [].

# """

# ✅ CURRENT WORKING PROMPT FOR SECTION MAPPING
# standard_headings_prompt = """

#     You are a resume expert. I will provide you with a set of headers that are extracted from a resume.

#     Here are the extracted headers: {cleaned_headers}
    
#     There are multiple ways to write the same thing. Example: Professional Experience, Work Experience, Work history, all can be called "Experience".

#     Here is a list of Standard Headers: ["Objective", "Summary", "Experience", "Projects"]  
    
#     I want you map the extract headers to the Standard Headers as key value pairs where each standard header is a key and the extracted header is the value.
    
#     Example 1: if you get a heading Professional Summary, I want you to map it to "Summary". 
#     Example 2: if you get a heading Work Experience, I want you to map it to "Experience".


#     Your task is to return a python dictionary where the keys are the standard headers and the values are the extracted headers. 

#     CRITICAL RULES:

#     1. Do not create other keys, only find matches for the provided keys.
    
#     2. If a key doesn't have a match, set it equal to an empty string ""

#     3. If no match is found for any standard header, return an empty dictionary.

#     Do not include explanations, code, markdown, or any other text.

# """

# 🧪 TEST PROMPT 1
# standard_headings_prompt = """

# You are a strict JSON generator.

# Extracted headers: {cleaned_headers}

# Standard Headers (ONLY allowed keys):
# ["Objective", "Summary", "Experience", "Projects"]

# TASK:
# Map extracted headers to ONLY these standard headers.

# RULES (STRICT):
# 1. Output MUST be a valid Python dictionary.
# 2. ONLY use the 4 keys given above. NEVER add new keys.
# 3. Values must be from extracted headers OR "".
# 4. Each extracted header can be used at most once.
# 5. If no matches found at all, return {{}}.

# OUTPUT FORMAT (STRICT):
# {{"Objective": "", "Summary": "", "Experience": "", "Projects": ""}}

# NO explanation. NO extra keys. NO text.

# """

# 🧪 TEST PROMPT 2

standard_headings_prompt = """
You are a resume header mapping assistant.

STANDARD HEADERS (these are the ONLY allowed keys): ["Objective", "Summary", "Experience", "Projects"]

EXTRACTED HEADERS: {cleaned_headers}

Your job: Map each STANDARD HEADER to the closest EXTRACTED HEADER.

RULES (strictly follow):
1. Output ONLY a Python dictionary.
2. Keys MUST be EXACTLY these 4: "Objective", "Summary", "Experience", "Projects"
3. Values must come from the EXTRACTED HEADERS list only.
4. If no match found for a key, set value to "".
5. Do NOT add any extra keys beyond the 4 standard headers.
6. Do NOT output explanations, markdown, or code blocks.

Output format example:
{{"Objective": "", "Summary": "", "Experience": "EXPERIENCE", "Projects": "PROJECTS"}}
"""