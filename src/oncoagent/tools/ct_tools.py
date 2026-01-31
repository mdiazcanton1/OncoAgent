"""ClinicalTrials.gov API v2 wrapper."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import requests

BASE_URL = "https://clinicaltrials.gov/api/v2"


def search_studies(
    condition: Optional[str] = None,
    intervention: Optional[str] = None,
    location: Optional[str] = None,
    sponsor: Optional[str] = None,
    status: Optional[Union[str, List[str]]] = None,
    nct_ids: Optional[List[str]] = None,
    sort: str = "LastUpdatePostDate:desc",
    page_size: int = 10,
    page_token: Optional[str] = None,
    format: str = "json",
) -> Dict:
    """Search for clinical trials using API v2."""
    params: Dict[str, str] = {}

    if condition:
        params["query.cond"] = condition
    if intervention:
        params["query.intr"] = intervention
    if location:
        params["query.locn"] = location
    if sponsor:
        params["query.spons"] = sponsor
    if status:
        params["filter.overallStatus"] = (
            ",".join(status) if isinstance(status, list) else status
        )
    if nct_ids:
        params["filter.ids"] = ",".join(nct_ids)

    params["sort"] = sort
    params["pageSize"] = str(page_size)
    if page_token:
        params["pageToken"] = page_token
    params["format"] = format

    response = requests.get(f"{BASE_URL}/studies", params=params, timeout=30)
    response.raise_for_status()
    return response.json() if format == "json" else {"csv": response.text}


def get_study_details(nct_id: str, format: str = "json") -> Dict:
    """Retrieve full details for a specific trial."""
    response = requests.get(
        f"{BASE_URL}/studies/{nct_id}", params={"format": format}, timeout=30
    )
    response.raise_for_status()
    return response.json() if format == "json" else {"csv": response.text}


def extract_study_summary(study: Dict) -> Dict:
    """Extract key fields for a quick trial summary."""
    protocol = study.get("protocolSection", {})
    identification = protocol.get("identificationModule", {})
    status_module = protocol.get("statusModule", {})
    description = protocol.get("descriptionModule", {})

    return {
        "nct_id": identification.get("nctId"),
        "title": identification.get("officialTitle") or identification.get("briefTitle"),
        "status": status_module.get("overallStatus"),
        "phase": protocol.get("designModule", {}).get("phases", []),
        "enrollment": protocol.get("designModule", {})
        .get("enrollmentInfo", {})
        .get("count"),
        "brief_summary": description.get("briefSummary"),
        "last_update": status_module.get("lastUpdatePostDateStruct", {}).get("date"),
    }

