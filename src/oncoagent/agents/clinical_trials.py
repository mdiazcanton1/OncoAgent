"""ClinicalTrials.gov agent."""

from __future__ import annotations

import asyncio

from ..state import OncoAgentState
from ..tools.ct_tools import extract_study_summary, search_studies


def _infer_condition(state: OncoAgentState) -> str:
    return state.get("cancer_type") or state["original_query"]


async def clinical_trials_agent(state: OncoAgentState) -> dict:
    """Search for relevant clinical trials."""
    results = await asyncio.to_thread(
        search_studies,
        condition=_infer_condition(state),
        status=["RECRUITING", "ACTIVE_NOT_RECRUITING"],
        page_size=20,
        sort="LastUpdatePostDate:desc",
    )

    trials = []
    for study in results.get("studies", []):
        summary = extract_study_summary(study)
        trials.append(
            {
                "nct_id": summary["nct_id"],
                "title": summary["title"],
                "status": summary["status"],
                "phase": summary["phase"],
                "enrollment": summary["enrollment"],
                "brief_summary": summary["brief_summary"],
                "url": f"https://clinicaltrials.gov/study/{summary['nct_id']}",
                "last_update": summary.get("last_update"),
            }
        )

    return {"clinical_trials": trials, "agents_completed": ["clinical_trials"]}

