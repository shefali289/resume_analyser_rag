"""Portable Markdown report generation."""

from __future__ import annotations

from datetime import UTC, datetime

from .ats import ATSAnalysis
from .parsers import ParsedResume


def build_markdown_report(
    resume: ParsedResume,
    analysis: ATSAnalysis | None,
    target_role: str,
    ai_outputs: dict[str, str],
) -> str:
    lines = [
        "# CareerLens Resume Analysis",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Resume: {resume.filename}",
        f"Target role: {target_role or 'Not specified'}",
        "",
    ]
    if analysis:
        lines.extend(
            [
                "## ATS Match",
                "",
                f"**Overall score: {analysis.score}/100**",
                "",
                "### Score breakdown",
                "",
            ],
        )
        lines.extend(
            f"- {component.label}: {component.percentage}%"
            for component in analysis.components
        )
        lines.extend(
            [
                "",
                "### Matched keywords",
                "",
                ", ".join(analysis.matched_keywords) or "None detected",
                "",
                "### Missing keywords to validate",
                "",
                ", ".join(analysis.missing_keywords) or "None",
                "",
                "### Recommendations",
                "",
            ],
        )
        lines.extend(f"- {recommendation}" for recommendation in analysis.recommendations)

    titles = {
        "analysis": "AI Career Analysis",
    }
    for key, title in titles.items():
        if ai_outputs.get(key):
            lines.extend(["", f"## {title}", "", ai_outputs[key]])

    lines.extend(
        [
            "",
            "---",
            "CareerLens suggestions must be reviewed for accuracy before use.",
        ],
    )
    return "\n".join(lines).strip() + "\n"
