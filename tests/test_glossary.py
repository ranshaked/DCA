import unittest

from training_analyzer.glossary import parse_glossary_file


class GlossaryImportTests(unittest.TestCase):
    def test_parses_hebrew_csv_columns(self) -> None:
        records = parse_glossary_file(
            "terms.csv",
            "מונח,משמעות,גרסאות,הערות\nקודקוד,מפקד,קדקד,קשר\n".encode(),
        )

        self.assertEqual(records[0]["term"], "קודקוד")
        self.assertEqual(records[0]["meaning"], "מפקד")

    def test_parses_plain_text_pairs(self) -> None:
        records = parse_glossary_file("terms.txt", "נץ: רחפן\n".encode())

        self.assertEqual(records, [{"term": "נץ", "meaning": "רחפן", "variants": "", "notes": ""}])


if __name__ == "__main__":
    unittest.main()
