import unittest

from resume_analyser.ats import analyze_resume


RESUME = """
SUMMARY
Data engineer with experience building reliable analytics products.

SKILLS
Python, SQL, AWS, Docker, PostgreSQL, Airflow

EXPERIENCE
- Built Python pipelines that reduced processing time by 40%.
- Automated quality checks for 2 million monthly records.

EDUCATION
Bachelor of Information Technology

PROJECTS
- Developed a retrieval augmented generation assistant.

CERTIFICATIONS
AWS Cloud Practitioner

person@example.com | +64 21 555 555 | https://linkedin.com/in/example
"""

JOB = """
We are hiring a data engineer with Python, SQL, AWS, Airflow, Docker,
communication, Spark, and Databricks experience.
"""


class ATSAnalysisTests(unittest.TestCase):
    def test_scores_and_explains_keyword_match(self) -> None:
        analysis = analyze_resume(RESUME, JOB)

        self.assertGreaterEqual(analysis.score, 65)
        self.assertIn("python", analysis.matched_keywords)
        self.assertIn("spark", analysis.missing_keywords)
        self.assertIn("Experience", analysis.sections_found)
        self.assertEqual(analysis.quantified_bullets, 2)

    def test_score_is_bounded(self) -> None:
        analysis = analyze_resume("Short resume", "Python engineer")
        self.assertGreaterEqual(analysis.score, 0)
        self.assertLessEqual(analysis.score, 100)


if __name__ == "__main__":
    unittest.main()
