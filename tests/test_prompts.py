import unittest

from training_analyzer.prompts import load_prompt, render_prompt
from training_analyzer.reports import ReportTemplate


class PromptTests(unittest.TestCase):
    def test_prompt_values_are_substituted(self) -> None:
        rendered = render_prompt(
            "report_evidence_query",
            section_title="Timeline",
            section_instructions="Use timestamps",
        )

        self.assertIn("Timeline", rendered)
        self.assertIn("Use timestamps", rendered)
        self.assertNotIn("$section_title", rendered)

    def test_every_report_template_section_has_an_external_prompt(self) -> None:
        template = ReportTemplate()

        for section in template.sections:
            self.assertEqual(
                section["instructions"],
                load_prompt(f"report_sections/{section['id']}"),
            )


if __name__ == "__main__":
    unittest.main()
