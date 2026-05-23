from typing import Optional
DRIVER_RULES = {
    "poutcome": [
        ("success", "+", "Previous campaign converted",
         "Open with: 'last time we spoke you signed up — wanted to follow up on a fresh option for you.'"),
        ("success", "-", "Previous success but model says low fit now",
         "Acknowledge prior relationship but pitch lightly — they may not want a repeat product."),
        ("failure", "-", "Previously declined",
         "Don't repeat last campaign's pitch. Ask what changed since last contact."),
        ("nonexistent", "-", "Never contacted before",
         "Cold opener — focus on rapport, not the product."),
    ],
    "contact": [
        ("cellular", "+", "Reachable on mobile",
         "Keep the call short — mobile attention spans are shorter."),
        ("telephone", "-", "Landline contact",
         "Likely older demographic — slow down, no jargon."),
    ],
    "month": [
        ("mar", "+", "March historically strong month", "Mention end of fiscal quarter incentives."),
        ("sep", "+", "September strong response", "Lead with back-to-routine framing."),
        ("oct", "+", "October strong response", "Year-end planning angle."),
        ("dec", "+", "December strong response", "Mention tax-year deadline."),
        ("may", "-", "May has high volume but low conversion",
         "Customer may be fatigued — keep it brief, give them an easy out."),
    ],
    "job": [
        ("retired", "+", "Retired customer",
         "Emphasize capital safety and predictable returns of the term deposit."),
        ("student", "+", "Student",
         "Emphasize small minimums and habit-building savings."),
        ("blue-collar", "-", "Blue-collar — historically harder conversion",
         "Lead with respect for their time. Concrete numbers, no abstractions."),
        ("admin.", "+", "Admin role",
         "They appreciate clear documentation — offer to email terms."),
    ],
    "education": [
        ("university_degree", "+", "University educated",
         "Comfortable with detail — share rate vs market briefly."),
        ("basic_4y", "-", "Basic schooling",
         "Avoid finance jargon. Use round numbers and analogies."),
    ],
    "default": [
        ("no", "+", "No credit default history",
         "Eligible for premium tier — mention it as a positive."),
        ("yes", "-", "Has credit default history",
         "Lead with low-minimum option, don't push large amounts."),
    ],
    "housing": [
        ("yes", "+", "Has housing loan — banking relationship",
         "Reference existing relationship to build trust."),
    ],
    "loan": [
        ("yes", "-", "Already has personal loan",
         "Avoid sounding like you're piling on debt — frame as savings, not commitment."),
    ],
    "was_contacted_before": [
        (1, "+", "Has prior campaign contact history",
         "Reference the prior call to show continuity."),
        (0, "-", "No prior contact",
         "Spend first 20 seconds on introduction, not pitch."),
    ],
}

NUMERIC_HINTS = {
    "age": {
        "young": (lambda v: v < 30, "Younger customer",
                  "Emphasize digital onboarding and flexibility."),
        "mid": (lambda v: 30 <= v <= 55, "Career-building age",
                "Frame as long-term security building."),
        "senior": (lambda v: v > 55, "Senior customer",
                   "Capital preservation framing — avoid risk language."),
    },
    "campaign": {
        "high": (lambda v: v >= 4, "Heavily contacted this campaign",
                 "Acknowledge prior calls — apologize for repeated contact. Don't push."),
        "low": (lambda v: v <= 1, "First call this campaign",
                "Fresh slate — invest in rapport before product."),
    },
    "euribor3m": {
        "low": (lambda v: v < 1.5, "Low interest rate environment",
                "Mention that current rates make term deposits especially attractive."),
        "high": (lambda v: v >= 4.5, "High interest rate environment",
                 "Lead with the specific rate — it's a strong number right now."),
    },
    "emp_var_rate": {
        "negative": (lambda v: v < 0, "Negative employment trend",
                     "Mention safety and FDIC-equivalent guarantees."),
    },
}


def driver_for(feature: str, value, shap_value: float) -> Optional[dict]:
    """Return {'driver': str, 'hint': str, 'shap': float, 'feature': str, 'value': value} or None."""
    sign = "+" if shap_value > 0 else "-"

    if feature in DRIVER_RULES:
        for rule_value, rule_sign, driver, hint in DRIVER_RULES[feature]:
            if str(rule_value) == str(value) and rule_sign == sign:
                return {
                    "feature": feature,
                    "value": value,
                    "driver": driver,
                    "hint": hint,
                    "shap": float(shap_value),
                    "direction": sign,
                }

    if feature in NUMERIC_HINTS:
        for label, (predicate, driver, hint) in NUMERIC_HINTS[feature].items():
            try:
                if predicate(float(value)):
                    return {
                        "feature": feature,
                        "value": value,
                        "driver": driver,
                        "hint": hint,
                        "shap": float(shap_value),
                        "direction": sign,
                    }
            except (TypeError, ValueError):
                continue

    return {
        "feature": feature,
        "value": value,
        "driver": f"{feature} = {value} ({'pushes yes' if sign == '+' else 'pushes no'})",
        "hint": None,
        "shap": float(shap_value),
        "direction": sign,
    }
