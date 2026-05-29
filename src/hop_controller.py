import spacy


class HopController:

    def __init__(self):

        print("[INFO] Loading spaCy model...")

        self.nlp = spacy.load(
            "en_core_web_sm"
        )

    # =====================================
    # EXTRACT ENTITIES + LABELS
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

                entities.append(
                    (ent.text.strip(), ent.label_)
                )

        return entities

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

            candidate_entities.extend(
                entities
            )

        if len(candidate_entities) == 0:
            return None

        entity_stats = {}

        for entity, label in candidate_entities:

            key = (entity, label)

            entity_stats[key] = (
                entity_stats.get(key, 0) + 1
            )

        print("\n[DEBUG] Candidate Entities:")

        for (entity, label), freq in sorted(
            entity_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]:

            print(
                f"{entity:<30}"
                f"{label:<15}"
                f"freq={freq}"
            )

        label_priority = {
            "PERSON": 0,
            "ORG": 1,
            "GPE": 2,
            "WORK_OF_ART": 3
        }

        ranked_entities = sorted(
            entity_stats.items(),
            key=lambda x: (
                label_priority.get(
                    x[0][1],
                    999
                ),
                -x[1]
            )
        )

        question_text = query.lower()

        for (entity, label), freq in ranked_entities:

            if entity.lower() not in question_text:

                print(
                    f"\n[INFO] Selected Bridge Entity:"
                    f" {entity} ({label})"
                )

                return entity

        best_entity = ranked_entities[0][0][0]

        print(
            f"\n[INFO] Fallback Bridge Entity:"
            f" {best_entity}"
        )

        return best_entity

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