"""Transparent, deterministic ATS and resume-quality scoring."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "of", "on", "or", "our", "that", "the", "their",
    "this", "to", "we", "will", "with", "you", "your", "years", "role", "work",
    "working", "team", "skills", "experience", "required", "preferred", "using",
}

SKILL_CATEGORIES = {
    "AI & Data": {
        "artificial intelligence", "machine learning", "deep learning", "rag",
        "retrieval augmented generation", "nlp", "natural language processing",
        "computer vision", "data science", "data analysis", "prompt engineering",
        "langchain", "llm", "large language models", "embeddings", "vector database",
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "power bi",
        "tableau", "statistics",
    },
    "Programming": {
        "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust",
        "sql", "html", "css", "react", "next.js", "node.js", "fastapi", "django",
        "flask", "streamlit", "rest api", "graphql",
    },
    "Cloud & DevOps": {
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "terraform",
        "ci/cd", "github actions", "linux", "serverless", "cloudflare",
    },
    "Data & Platforms": {
        "postgresql", "postgres", "mysql", "mongodb", "redis", "supabase",
        "snowflake", "databricks", "spark", "kafka", "airflow", "dbt", "pgvector",
    },
    "Delivery & Leadership": {
        "agile", "scrum", "stakeholder management", "project management",
        "product management", "leadership", "mentoring", "communication",
        "problem solving", "cross-functional", "strategy", "roadmap",
    },
}

SECTION_PATTERNS = {
    "Summary": r"(?im)^\s*(professional\s+)?(summary|profile|objective)\s*:?[ \t]*$",
    "Experience": r"(?im)^\s*(work\s+|professional\s+)?experience\s*:?[ \t]*$|^\s*employment\s*:?[ \t]*$",
    "Education": r"(?im)^\s*education\s*:?[ \t]*$|^\s*academic(s| background)?\s*:?[ \t]*$",
    "Skills": r"(?im)^\s*(technical\s+|core\s+)?skills\s*:?[ \t]*$|^\s*technologies\s*:?[ \t]*$",
    "Projects": r"(?im)^\s*(selected\s+|key\s+)?projects?\s*:?[ \t]*$",
    "Certifications": r"(?im)^\s*(certifications?|licenses?)\s*:?[ \t]*$",
}

ACTION_VERBS = {
    "achieved", "automated", "built", "created", "delivered", "designed", "developed",
    "drove", "enhanced", "established", "generated", "grew", "implemented", "improved",
    "increased", "launched", "led", "managed", "optimized", "orchestrated", "reduced",
    "resolved", "scaled", "spearheaded", "streamlined", "transformed",
}


@dataclass(frozen=True)
class ScoreComponent:
    label: str
    score: float
    maximum: float

    @property
    def percentage(self) -> int:
        return round((self.score / self.maximum) * 100) if self.maximum else 0


@dataclass
class ATSAnalysis:
    score: int
    components: list[ScoreComponent]
    matched_keywords: list[str]
    missing_keywords: list[str]
    detected_skills: dict[str, list[str]]
    sections_found: list[str]
    sections_missing: list[str]
    quantified_bullets: int
    total_bullets: int
    recommendations: list[str] = field(default_factory=list)


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = r"(?<![\w])" + re.escape(phrase) + r"(?![\w])"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def _skill_map(text: str) -> dict[str, list[str]]:
    return {
        category: sorted(skill for skill in skills if _contains_phrase(text, skill))
        for category, skills in SKILL_CATEGORIES.items()
    }


def _job_keywords(job_description: str, limit: int = 28) -> list[str]:
    explicit_skills = {
        skill
        for skills in SKILL_CATEGORIES.values()
        for skill in skills
        if _contains_phrase(job_description, skill)
    }
    tokens = re.findall(r"\b[a-z][a-z0-9+#.-]{2,}\b", job_description.lower())
    counts = Counter(token for token in tokens if token not in STOP_WORDS)
    frequent_terms = [term for term, _ in counts.most_common(limit)]
    return sorted(explicit_skills, key=str.lower) + [
        term for term in frequent_terms if term not in explicit_skills
    ][: max(0, limit - len(explicit_skills))]


def _bullet_stats(text: str) -> tuple[int, int, int]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullets = [line for line in lines if re.match(r"^(?:[-*•▪◦]|\d+[.)])\s+", line)]
    quantified = [
        line
        for line in bullets
        if re.search(r"(?:\b\d+(?:\.\d+)?%?|[$£€]\s?\d+|\b\d+[kKmMbB]\b)", line)
    ]
    action_led = [
        line
        for line in bullets
        if re.sub(r"^(?:[-*•▪◦]|\d+[.)])\s+", "", line).split(" ", 1)[0].lower()
        in ACTION_VERBS
    ]
    return len(bullets), len(quantified), len(action_led)


def _readability_score(text: str, word_count: int) -> float:
    score = 0.0
    if 350 <= word_count <= 1_000:
        score += 5
    elif 250 <= word_count <= 1_300:
        score += 3
    sentences = [item for item in re.split(r"[.!?]+", text) if item.strip()]
    average_length = word_count / max(len(sentences), 1)
    if 8 <= average_length <= 28:
        score += 3
    elif average_length <= 35:
        score += 2
    if not re.search(r"[^\x00-\x7F]{12,}", text):
        score += 2
    return min(score, 10)


def analyze_resume(resume_text: str, job_description: str) -> ATSAnalysis:
    """Score a resume against a job description using explainable heuristics."""

    keywords = _job_keywords(job_description)
    matched = [keyword for keyword in keywords if _contains_phrase(resume_text, keyword)]
    missing = [keyword for keyword in keywords if keyword not in matched]
    keyword_score = 50 * (len(matched) / max(len(keywords), 1))

    found_sections = [
        name for name, pattern in SECTION_PATTERNS.items() if re.search(pattern, resume_text)
    ]
    missing_sections = [name for name in SECTION_PATTERNS if name not in found_sections]
    section_score = 15 * (len(found_sections) / len(SECTION_PATTERNS))

    total_bullets, quantified_bullets, action_led_bullets = _bullet_stats(resume_text)
    if total_bullets:
        impact_ratio = min(1.0, (quantified_bullets + action_led_bullets) / total_bullets)
    else:
        impact_ratio = 0.0
    impact_score = 15 * impact_ratio

    word_count = len(re.findall(r"\b[\w+#.-]+\b", resume_text))
    readability_score = _readability_score(resume_text, word_count)

    contact_checks = [
        bool(re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", resume_text)),
        bool(re.search(r"(?:\+?\d[\d ()-]{7,}\d)", resume_text)),
        "linkedin.com/" in resume_text.lower(),
    ]
    contact_score = (sum(contact_checks) / len(contact_checks)) * 10

    components = [
        ScoreComponent("Job keyword match", keyword_score, 50),
        ScoreComponent("Section completeness", section_score, 15),
        ScoreComponent("Evidence & impact", impact_score, 15),
        ScoreComponent("Readability", readability_score, 10),
        ScoreComponent("Contact details", contact_score, 10),
    ]
    total_score = round(sum(component.score for component in components))

    recommendations: list[str] = []
    if missing:
        recommendations.append(
            "Add genuinely applicable missing keywords in context: " + ", ".join(missing[:8]) + ".",
        )
    if missing_sections:
        recommendations.append("Add clear ATS-friendly headings for: " + ", ".join(missing_sections) + ".")
    if total_bullets == 0:
        recommendations.append("Use concise bullet points for achievements and responsibilities.")
    elif quantified_bullets / total_bullets < 0.3:
        recommendations.append("Quantify more achievements with scale, speed, revenue, savings, or percentages.")
    if action_led_bullets < max(1, total_bullets // 2):
        recommendations.append("Start more bullets with strong action verbs and lead with outcomes.")
    if word_count < 350:
        recommendations.append("Add enough evidence and outcomes; the resume is currently quite brief.")
    elif word_count > 1_000:
        recommendations.append("Tighten older or less relevant content to improve scanability.")

    return ATSAnalysis(
        score=max(0, min(total_score, 100)),
        components=components,
        matched_keywords=matched,
        missing_keywords=missing,
        detected_skills={
            category: skills for category, skills in _skill_map(resume_text).items() if skills
        },
        sections_found=found_sections,
        sections_missing=missing_sections,
        quantified_bullets=quantified_bullets,
        total_bullets=total_bullets,
        recommendations=recommendations,
    )
