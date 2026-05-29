class AnswerGenerator:

    def generate(
        self,
        retrieved_docs
    ):

        combined = ""

        for doc in retrieved_docs:

            combined += doc["text"] + "\n"

        return combined[:1000]