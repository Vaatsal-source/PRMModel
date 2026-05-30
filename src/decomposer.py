class QuestionDecomposer:

    def decompose(self, question):

        q = question.lower()

        # VERY IMPORTANT: deterministic rules first

        if " and " in q:

            parts = question.split(" and ")

            if "same nationality" in q:

                return [
                    f"What nationality was {parts[0].replace('Were ', '').strip()}?",
                    f"What nationality was {parts[1].replace(' of the same nationality?', '').strip()}?"
                ]

        return [question]