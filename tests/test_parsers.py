import unittest

from resume_analyser.parsers import parse_resume, redact_personal_data


class ParserTests(unittest.TestCase):
    def test_parses_text_resume_in_memory(self) -> None:
        content = (
            "SUMMARY\nSoftware engineer\nEXPERIENCE\n"
            + "Built reliable applications and improved delivery outcomes. " * 8
        ).encode()
        resume = parse_resume(content, "candidate.txt")

        self.assertEqual(resume.file_type, "TXT")
        self.assertEqual(resume.filename, "candidate.txt")
        self.assertGreater(resume.word_count, 25)
        self.assertEqual(len(resume.resume_id), 20)

    def test_redacts_contact_details(self) -> None:
        text = "person@example.com +64 21 555 555 https://github.com/person"
        redacted = redact_personal_data(text)

        self.assertNotIn("person@example.com", redacted)
        self.assertNotIn("555 555", redacted)
        self.assertNotIn("github.com/person", redacted)


if __name__ == "__main__":
    unittest.main()
