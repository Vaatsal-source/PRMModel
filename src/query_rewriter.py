class QueryRewriter:

    def rewrite(
        self,
        original_question,
        bridge_entity
    ):

        question = original_question.lower()

        if "author of" in question:

            return (
                f"Where was {bridge_entity} born?"
            )

        if "born" in question:

            return (
                f"Where was {bridge_entity} born?"
            )

        return (
            f"{bridge_entity} {original_question}"
        )