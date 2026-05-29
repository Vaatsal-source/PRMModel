class QueryRewriter:

    def rewrite(
        self,
        original_query,
        bridge_entity
    ):

        q = original_query.lower()

        if "born" in q:
            return f"Where was {bridge_entity} born?"

        if "country" in q:
            return f"{bridge_entity} birthplace country"

        if "author" in q:
            return f"Information about {bridge_entity}"

        return f"{bridge_entity}"