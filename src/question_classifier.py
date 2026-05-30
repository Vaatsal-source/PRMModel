class QuestionClassifier:

    def classify(self, question):

        q = question.lower()

        comparison_patterns = [
            "same",
            "both",
            "more",
            "less",
            "older",
            "younger"
        ]

        for p in comparison_patterns:

            if p in q:
                return "comparison"

        return "bridge"