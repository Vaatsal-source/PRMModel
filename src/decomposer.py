class QuestionDecomposer:
    def decompose(self, question: str) -> list:
        q = question.lower().strip()
        
        # Rule-based handling for standard HotpotQA bridge/comparison structures
        if " and " in q:
            if "same nationality" in q:
                # Clean up punctuation
                clean_q = question.replace("?", "").replace("Were ", "").replace("was ", "")
                parts = clean_q.split(" and ")
                p1 = parts[0].strip()
                p2 = parts[1].replace("of the same nationality", "").strip()
                return [
                    f"What nationality was {p1}?",
                    f"What nationality was {p2}?"
                ]
        
        # Fallback return wrapping the query as an atomic unit
        return [question]