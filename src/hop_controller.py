import spacy


class HopController:

    def __init__(self):

        print("[INFO] Loading spaCy model...")

        self.nlp = spacy.load(
            "en_core_web_sm"
        )

    # =====================================
    # EXTRACT ENTITIES
    # =====================================

    def extract_entities(self, text):

        doc = self.nlp(text)

        entities = []

        for ent in doc.ents:

            if ent.label_ in [
                "PERSON",
                "ORG",
                "GPE"
            ]:

                entities.append(
                    (ent.text, ent.label_)
                )

        return entities

    # =====================================
    # SELECT BRIDGE ENTITY
    # =====================================

    def select_bridge_entity(
        self,
        query,
        retrieved_docs
    ):

        entity_frequency = {}

        for doc in retrieved_docs:

            entities = self.extract_entities(
                doc["text"]
            )

            for entity, label in entities:

                key = (entity, label)

                entity_frequency[key] = (
                    entity_frequency.get(key, 0) + 1
                )

        if len(entity_frequency) == 0:

            return None

        sorted_entities = sorted(
            entity_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print("\n[DEBUG] Candidate Entities:")

        for (entity, label), freq in sorted_entities:

            print(
                f"{entity:30s} "
                f"{label:10s} "
                f"freq={freq}"
            )

        query_lower = query.lower()

        for (entity, label), freq in sorted_entities:

            if (
                label == "PERSON"
                and entity.lower() not in query_lower
            ):

                print(
                    f"\n[INFO] Selected Bridge Entity: "
                    f"{entity}"
                )

                return entity

        return sorted_entities[0][0][0]