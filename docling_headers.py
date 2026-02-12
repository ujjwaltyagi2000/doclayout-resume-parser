"""

Script to extract headers from a resume PDF file using docling.

Status: Working ✅

"""

from docling.document_converter import DocumentConverter

source = r"Megha resume.pdf"
converter = DocumentConverter()
result = converter.convert(source)
doc = result.document

for item, level in doc.iterate_items():
    label = item.label.value if hasattr(item.label, 'value') else str(item.label)
    text = item.text.strip() if hasattr(item, 'text') else ""
    
    # --- NEW: Size Calculation Logic ---
    size = 0
    if hasattr(item, 'prov') and item.prov:
        # prov[0] is the location on the first page this item appears
        bbox = item.prov[0].bbox
        # Size = Top coordinate - Bottom coordinate
        size = round(bbox.t - bbox.b, 2)
    # ------------------------------------

    if label == "section_header":
        # We print the size here to see the difference between H1, H2, etc.
        print(f"\n[HEADING L{level}] (Size: {size}): {text.upper()}")
        print("=" * 40)
    elif label == "list_item":
        print(f"  • {text} (Size: {size})")
    else:
        # For general text, size helps identify bold/small print
        print(f"[{label.upper()} - Size {size}]: {text}")