import logging
from typing import Dict, Any, List

logger = logging.getLogger("creatorrocket")

def generate_explanation(raw_data: Dict, prediction: Dict) -> Dict:
    prob = prediction["prob_score"]
    eng = raw_data.get("engagement_rate", 5.0)
    con = raw_data.get("consistency_score", 50)
    vel = raw_data.get("velocity_7d", 0)
    nic = raw_data.get("niche_momentum", 50)
    aud = raw_data.get("audience_quality", 75)
    acc = raw_data.get("acceleration", 50)
    vir = raw_data.get("virality_score", 5.0)

    feature_map = {
        "Follower Velocity": min(100, max(0, (vel / 3000) * 100)),
        "Engagement Depth": min(100, max(0, (eng / 20) * 100)),
        "Content Consistency": min(100, max(0, con)),
        "Niche Momentum": min(100, max(0, nic)),
        "Audience Authenticity": min(100, max(0, aud)),
    }
    sorted_feats = sorted(feature_map.items(), key=lambda x: x[1], reverse=True)

    driver_descs = {
        "Follower Velocity": ["Velocity is surging — compounding growth detected.", "Follower growth is trending positively.", "Velocity is below the breakout threshold."],
        "Engagement Depth": ["Engagement is top-percentile — deep audience resonance.", "Engagement is healthy and above median.", "Engagement metrics suggest passive audience connection."],
        "Content Consistency": ["Posting consistency is exceptional — algorithm gold standard.", "Posting regularity is adequate but has room to improve.", "Inconsistent posting is the primary growth bottleneck."],
        "Niche Momentum": ["Content niche is in a rapid growth phase.", "Niche showing steady momentum.", "Niche growth is decelerating — may need a pivot."],
        "Audience Authenticity": ["Follower audit shows highly authentic audience.", "Audience quality is within normal range.", "Audience quality signals suggest possible bot inflation."],
    }

    drivers = []
    for name, score in sorted_feats[:4]:
        idx = 0 if score > 66 else (1 if score > 33 else 2)
        drivers.append({"name": name, "strength": int(score), "desc": driver_descs[name][idx]})

    risk_descs = {
        "Consistency Risk": ["Inconsistency creates algorithmic penalties.", "Regularity could be improved.", "Consistency is strong — low risk."],
        "Plateau Risk": ["Growth deceleration detected.", "Growth rate is flat — may need innovation.", "Acceleration signals are positive."],
        "Discovery Risk": ["Low viral reach limits organic discovery.", "Virality is moderate — format innovation could unlock reach.", "Content achieves strong viral reach."],
    }
    risk_feats = [("Consistency Risk", con), ("Plateau Risk", acc), ("Discovery Risk", vir * 10)]
    risks = []
    for name, score in risk_feats:
        level = max(0, min(100, 100 - score))
        idx = 0 if level > 50 else (1 if level > 25 else 2)
        risks.append({"name": name, "level": int(level), "desc": risk_descs[name][idx]})

    budget_base = raw_data.get("followers", 1000) * 0.02 * (0.8 + (eng / 20) * 0.6)
    b_min = max(200, int(budget_base * 0.7))
    b_max = max(b_min + 200, int(budget_base * 1.3))
    roi = round(1.5 + (prob / 100) * 8 + (eng / 20) * 2, 1)
    urgency = "High" if prob >= 75 else ("Medium" if prob >= 45 else "Low")
    window = "Next 3-6 weeks" if prob >= 75 else ("Next 6-10 weeks" if prob >= 45 else "Next 10-16 weeks")
    collab = {"window": window, "budget": f"${b_min:,} - ${b_max:,}", "roi": f"{roi}x", "urgency": urgency}

    name = raw_data.get("name", "Creator")
    niche = raw_data.get("niche", "YouTube")
    loc = raw_data.get("location", "Global")

    if prob >= 75:
        opening = f"{name} shows a strong breakout profile in the {niche} niche. With a {prob}% breakout probability, the model identifies several converging signals that historically precede rapid growth."
    elif prob >= 45:
        opening = f"{name} demonstrates a promising growth profile in the {niche} niche. The model assigns a {prob}% breakout probability, indicating meaningful potential with some areas requiring attention."
    else:
        opening = f"{name} presents a moderate growth profile in the {niche} niche. With a {prob}% breakout probability, the model identifies both strengths and significant gaps to address."

    eng_text = f"The engagement rate of {eng}% is notably above the category median, indicating genuine audience resonance." if eng > 10 else f"Engagement at {eng}% is within the healthy range, suggesting steady but not exceptional audience connection."
    con_text = "Posting consistency is a key strength, correlating with algorithmic favor." if con > 75 else "The primary concern is posting inconsistency, which the model identifies as the most impactful area for improvement."

    explanation = f"{opening} {eng_text} {con_text} The {loc} market presents favorable conditions for this content category, and the current weekly velocity of {vel:,} new followers positions them well for {'accelerated growth' if vel > 500 else 'improvement'}. {'Recommend early partnership engagement while costs remain accessible.' if prob >= 70 else 'Worth monitoring closely with a conditional collaboration strategy.'}"

    return {"drivers": drivers, "risks": risks, "collab": collab, "explanation": explanation}