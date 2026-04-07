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

standard_headings_prompt = """

    You are a resume expert. I will provide you with a set of headers that are extracted from a resume.

    Here are the extracted headers: {cleaned_headers}
    
    There are multiple ways to write the same thing. Example: Professional Experience, Work Experience, Work history, all can be called "Experience".

    Here is a list of Standard Headers: ["Objective", "Summary", "Experience", "Projects"]  
    
    I want you map the extract headers to the Standard Headers as key value pairs where each standard header is a key and the extracted header is the value.
    
    Example 1: if you get a heading Professional Summary, I want you to map it to "Summary". 
    Example 2: if you get a heading Work Experience, I want you to map it to "Experience".


    Your task is to return a python dictionary where the keys are the standard headers and the values are the extracted headers. 

    CRITICAL RULES:

    1. Do not create other keys, only find matches for the provided keys.
    
    2. If a key doesn't have a match, set it equal to an empty string ""

    3. If no match is found for any standard header, return an empty dictionary.

    Do not include explanations, code, markdown, or any other text.

"""