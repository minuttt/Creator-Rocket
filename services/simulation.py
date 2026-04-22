import random
import hashlib
import math
from typing import Dict

def deterministic_seed(username: str) -> int:
    return int(hashlib.md5(username.encode()).hexdigest(), 16) % (2 ** 32)

def generate_simulated_analysis(username: str, platform: str) -> Dict:
    """Fallback deterministic simulation when real data isn't tracked in DB."""
    seed = deterministic_seed(username)
    rng = random.Random(seed)
    
    followers = int(3000 + rng.lognormvariate(8.5, 0.8))
    followers = max(1000, min(49999, followers))
    engagement = round(3.0 + rng.random() * 14.0, 1)
    growth_rate = 0.02 + rng.random() * 0.12
    growth_rate *= (0.5 + engagement / 20.0)
    
    prob_score = rng.randint(20, 90)
    velocity_7d = int(followers * growth_rate * 0.25)
    
    return {
        "username": username, "platform": platform, "name": username.replace(".", " ").replace("_", " ").title(),
        "handle": f"@{username}", "niche": "Simulated Data", "location": "Local DB", "followers": followers,
        "followersDisplay": f"{followers/1000:.1f}K", "cadence": "Untracked",
        "probScore": prob_score, "prob12": max(1, prob_score-5), "prob24": min(99, prob_score+5),
        "velocity": velocity_7d, "engagementRate": engagement, 
        "viralityIndex": round(1.0 + rng.random() * 8.0, 1), "consistencyScore": rng.randint(40, 95),
        "drivers": [
            {"name":"Simulated Driver","strength":70,"desc":"Using simulated data - track channel for real metrics"},
            {"name":"Velocity","strength":60,"desc":"Simulated velocity baseline"}
        ],
        "risks": [
            {"name":"Simulated Risk","level":30,"desc":"Track channel to calculate real risk profile"}
        ],
        "collab": {"window":"N/A","budget":"N/A","roi":"N/A","urgency":"Low"},
        "explanation": f"{username} is not currently tracked in the database. Displaying simulated predictive baseline. Use the /creators/track endpoint to begin collecting real YouTube data for this creator.",
        "trajectoryHistorical": [int(followers * (0.9**i)) for i in range(12,0,-1)] + [followers],
        "trajectoryForecast6": [int(followers * (1.05**i)) for i in range(1,7)],
        "trajectoryForecast12": [int(followers * (1.05**i)) for i in range(1,13)],
        "trajectoryForecast24": [int(followers * (1.05**i)) for i in range(1,25)],
        "engagementData": [round(engagement * (0.8 + 0.2 * rng.random()), 1) for _ in range(13)]
    }