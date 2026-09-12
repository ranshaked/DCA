import unittest

from training_analyzer.chat import messages_with_retrieved_context


class RetrievedContextTests(unittest.TestCase):
    def test_latest_question_always_contains_retrieved_context(self) -> None:
        messages = [
            {"role": "user", "content": "מה זה מלוכלכים?"},
        ]
        context = "Source: glossary.md\nמלוכלך | an enemy soldier"

        augmented = messages_with_retrieved_context(messages, context)

        self.assertEqual(len(augmented), 1)
        self.assertIn("מה זה מלוכלכים?", augmented[-1]["content"])
        self.assertIn("מלוכלך | an enemy soldier", augmented[-1]["content"])
        self.assertIn("glossary.md", augmented[-1]["content"])


if __name__ == "__main__":
    unittest.main()
