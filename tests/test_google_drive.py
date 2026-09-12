import unittest

from training_analyzer.google_drive import folder_id_from_url


class GoogleDriveUrlTests(unittest.TestCase):
    def test_extracts_folder_id_from_standard_url(self) -> None:
        value = folder_id_from_url(
            "https://drive.google.com/drive/u/0/folders/1AbC_def-234?usp=sharing"
        )

        self.assertEqual(value, "1AbC_def-234")

    def test_extracts_folder_id_from_open_url(self) -> None:
        value = folder_id_from_url("https://drive.google.com/open?id=1AbC_def-234")

        self.assertEqual(value, "1AbC_def-234")

    def test_rejects_non_drive_urls(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid Google Drive"):
            folder_id_from_url("https://example.com/folders/secret")


if __name__ == "__main__":
    unittest.main()
