import spacy


class HopController:

    def __init__(self):

        print("[INFO] Loading spaCy model...")

        self.nlp = spacy.load(
            "en_core_web_sm"
        )

    # =====================================
    # EXTRACT BRIDGE ENTITIES
    # =====================================

    def extract_entities(self, text):

        doc = self.nlp(text)

        entities = []

        for ent in doc.ents:

            if ent.label_ in [
                "PERSON",
                "ORG",
                "GPE",
                "WORK_OF_ART"
            ]:

                entities.append(ent.text)

        return list(set(entities))

    # =====================================
    # SELECT BRIDGE ENTITY
    # =====================================

    def select_bridge_entity(
        self,
        query,
        retrieved_chunks
    ):

        candidate_entities = []

        for chunk in retrieved_chunks:

            entities = self.extract_entities(
                chunk["text"]
            )

            candidate_entities.extend(entities)

        if len(candidate_entities) == 0:
            return None

        entity_frequency = {}

        for entity in candidate_entities:

            entity_frequency[entity] = (
                entity_frequency.get(entity, 0) + 1
            )

        sorted_entities = sorted(
            entity_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )

        bridge_entity = sorted_entities[0][0]

        return bridge_entity

    # =====================================
    # REFORM QUERY
    # =====================================

    def reformulate_query(
        self,
        original_query,
        bridge_entity
    ):

        reformulated_query = (
            f"{bridge_entity} "
            f"{original_query}"
        )

        return reformulated_query