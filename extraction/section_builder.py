class SectionBuilder:
    def __init__(self, cleaned_headers):
        # Normalize headers once
        self.sub_headings = {self._normalize(h) for h in cleaned_headers}

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.upper().split())

    # -------------------------
    # Sort blocks (multi-column aware)
    # -------------------------
    def _sort_blocks(self, blocks):
        if not blocks:
            return blocks

        # Estimate page width from max x
        max_x = max(b["x"] for b in blocks)

        return sorted(
            blocks,
            key=lambda b: (
                b["page"],
                b["x"] > (max_x / 3),  # column heuristic
                b["y"]
            )
        )

    # -------------------------
    # Build sections
    # -------------------------
    def build(self, blocks):
        blocks = self._sort_blocks(blocks)

        sections = {}
        current_section = None

        for block in blocks:
            text_norm = self._normalize(block["text"])

            # 🔥 Section header detection
            if (
                block["class_id"] != 3 and   # ignore body text class
                text_norm in self.sub_headings
            ):
                current_section = block["text"]
                sections[current_section] = []
                continue

            if not current_section:
                continue

            sections[current_section].append({
                "class_id": block["class_id"],
                "class_name": block["class_name"],
                "text": block["text"]
            })

        return sections