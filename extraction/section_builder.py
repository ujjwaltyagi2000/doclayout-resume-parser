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
    def _sort_blocks(self, blocks, is_multiple_columns):
        if not blocks:
            return blocks

        # Estimate page width from max x
        max_x = max(b["x"] for b in blocks)

        if is_multiple_columns:
            print("🔍 Multiple columns detected")
            return sorted(
                blocks,
                key=lambda b: (
                    b["page"],
                    b["x"] > (max_x / 3),  # column heuristic
                    b["y"]
                )
            )

        elif not is_multiple_columns:
            print("🔍 Single column detected")
            return sorted(
                blocks,
                key=lambda b: (
                    b["page"],
                    b["y"]
                )
            )


        # return sorted(
        #     blocks,
        #     key=lambda b: (
        #         b["page"],
        #         # b["x"] > (max_x / 3),  # column heuristic
        #         b["y"]
        #     )
        # )

    # -------------------------
    # Build sections
    # -------------------------
    def build(self, blocks, is_multiple_columns):
        blocks = self._sort_blocks(blocks, is_multiple_columns)

        import json

        with open("yolo_sorted_blocks.json", "w") as f:
            json.dump(blocks, f, indent=2)

        sections = {}
        current_section = None

        for block in blocks:
            text_norm = self._normalize(block["text"])

            # Section header detection
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