from typing import Optional

DATASET_JOBS = {
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown",
}

KEYWORDS = [
    
    ("technician", ["engineer", "developer", "programmer", "software", "tech",
                    "it ", "data", "analyst", "designer", "architect", "devops",
                    "scientist", "research"]),
    ("management", ["manager", "director", "executive", "vp ", "president",
                    "lead", "doctor", "physician", "lawyer", "attorney",
                    "consultant", "officer", "head"]),
    ("admin.", ["admin", "secretary", "assistant", "clerk", "office",
                "coordinator", "teacher", "professor", "academic", "instructor"]),
    ("blue-collar", ["construction", "factory", "labour", "labor", "worker",
                     "mechanic", "electrician", "plumber", "carpenter", "driver",
                     "truck", "warehouse", "miner", "farmer", "fisher", "welder",
                     "operator", "chef", "cook"]),
    ("services", ["sales", "retail", "shop", "store", "customer", "waiter",
                  "waitress", "hospitality", "barista", "cashier", "delivery",
                  "courier", "rider", "nurse", "carer", "social", "stylist",
                  "hairdresser"]),
    ("entrepreneur", ["founder", "ceo", "owner", "entrepreneur", "business owner",
                      "startup"]),
    ("self-employed", ["freelance", "contractor", "self-employed", "consultant own",
                       "independent"]),
    ("retired", ["retired", "pensioner", "pension"]),
    ("student", ["student", "pupil", "college", "university student", "phd student"]),
    ("housemaid", ["housemaid", "maid", "housekeeper", "homemaker", "stay-at-home"]),
    ("unemployed", ["unemployed", "jobless", "between jobs", "looking for"]),
]


def map_job(raw: Optional[str]) -> str:
    """Return the closest dataset job category for a free-text job title."""
    if not raw:
        return "unknown"
    text = raw.strip().lower()
    if not text:
        return "unknown"
    if text in DATASET_JOBS:
        return text
    for category, keywords in KEYWORDS:
        for kw in keywords:
            if kw in text:
                return category
    return "unknown"
