import json
import random
import re
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------
# Dream Detective: a natural-language mystery game about sleep.
# The answer logic is based on a small, explicit case knowledge base.
# The LLM interprets the player's wording; Python checks the facts.
# ------------------------------------------------------------

st.set_page_config(
    page_title="Dream Detective",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

CASES = {
    "sleep_paralysis": {
        "name": "Sleep paralysis",
        "aliases": ["sleep paralysis", "paralysis", "being unable to move while awake"],
        "summary": "A brief state in which awareness returns while voluntary movement remains inhibited.",
        "facts": {
            "aware": True, "move": False, "dreaming": False, "transition": True,
            "vivid_perception": True, "fear": True, "control": False,
            "woke_up_twice": False, "childhood": False,
        },
        "explanation": "The key clue is awareness combined with an inability to move. Dream-like perceptions can occur, often around waking.",
    },
    "lucid_dream": {
        "name": "Lucid dreaming",
        "aliases": ["lucid dream", "lucid dreaming", "aware dream"],
        "summary": "A dream in which the dreamer knows they are dreaming.",
        "facts": {
            "aware": True, "move": True, "dreaming": True, "transition": False,
            "vivid_perception": True, "fear": False, "control": True,
            "woke_up_twice": False, "childhood": False,
        },
        "explanation": "The defining clue is knowing that the experience is a dream while it is happening. Some dream control may be possible.",
    },
    "false_awakening": {
        "name": "False awakening",
        "aliases": ["false awakening", "false wake up", "dreaming that you woke up"],
        "summary": "A dream in which the person believes they have woken up, but is still dreaming.",
        "facts": {
            "aware": False, "move": True, "dreaming": True, "transition": False,
            "vivid_perception": True, "fear": False, "control": False,
            "woke_up_twice": True, "childhood": False,
        },
        "explanation": "The person dreams that they have woken up, often in a familiar bedroom, and only later actually wakes.",
    },
    "hypnagogic_hallucination": {
        "name": "Hypnagogic hallucination",
        "aliases": ["hypnagogic hallucination", "hypnagogia", "hallucination while falling asleep"],
        "summary": "A vivid sensory experience occurring while falling asleep.",
        "facts": {
            "aware": True, "move": True, "dreaming": False, "transition": True,
            "vivid_perception": True, "fear": False, "control": False,
            "woke_up_twice": False, "childhood": False,
        },
        "explanation": "The experience happens during the transition into sleep, rather than in an established dream.",
    },
    "night_terror": {
        "name": "Night terror",
        "aliases": ["night terror", "sleep terror", "sleep terror episode"],
        "summary": "An episode of intense fear and arousal from sleep, often with limited later recall.",
        "facts": {
            "aware": False, "move": True, "dreaming": False, "transition": False,
            "vivid_perception": False, "fear": True, "control": False,
            "woke_up_twice": False, "childhood": True,
        },
        "explanation": "Intense fear and incomplete awareness or recall are characteristic clues. Night terrors are different from nightmares.",
    },
    "nightmare": {
        "name": "Nightmare",
        "aliases": ["nightmare", "bad dream", "frightening dream"],
        "summary": "A frightening dream that is usually remembered after waking.",
        "facts": {
            "aware": False, "move": True, "dreaming": True, "transition": False,
            "vivid_perception": True, "fear": True, "control": False,
            "woke_up_twice": False, "childhood": False,
        },
        "explanation": "A frightening dream with a remembered narrative is more consistent with a nightmare than a night terror.",
    },
}

# Natural-language concepts supported by the game. The LLM maps varied wording
# to one of these keys. Python then retrieves the answer from the case facts.
FEATURES = {
    "aware": "Was the person aware of their surroundings or of what was happening?",
    "move": "Could the person move their body voluntarily?",
    "dreaming": "Was the person actually dreaming?",
    "transition": "Did it happen while falling asleep or waking up?",
    "vivid_perception": "Were there vivid sensory experiences, such as seeing or hearing something?",
    "fear": "Was intense fear a central part of the experience?",
    "control": "Could the person intentionally influence or control the experience?",
    "woke_up_twice": "Did the person dream that they had woken up, before waking up for real?",
    "childhood": "Is this especially associated with episodes in childhood?",
}

CASE_LIST = "\n".join(f"- {key}: {value['name']}" for key, value in CASES.items())
FEATURE_LIST = "\n".join(f"- {key}: {description}" for key, description in FEATURES.items())


def get_secret(name: str, default: str = "") -> str:
    """Read a setting from Streamlit secrets, then environment variables."""
    try:
        value = st.secrets.get(name, default)
        if value:
            return str(value)
    except Exception:
        pass
    import os
    return os.getenv(name, default)


def get_llm_client():
    api_key = get_secret("AI_API_KEY")
    base_url = get_secret("AI_BASE_URL")
    if not api_key or not base_url:
        return None, "Add AI_API_KEY and AI_BASE_URL in .streamlit/secrets.toml."
    try:
        return OpenAI(api_key=api_key, base_url=base_url), None
    except Exception as exc:
        return None, f"Could not initialize the AI client: {exc}"


def call_interpreter(message: str) -> dict[str, Any] | None:
    """Ask the university LLM endpoint to classify a player's message."""
    client, error = get_llm_client()
    if client is None:
        return None

    model = get_secret("AI_MODEL", "Ornith-1.5-35B-A3B")

    system_prompt = f"""
You are the language interpreter for a mystery game. Do not reveal any solution.

Classify the player's latest message.
Return ONLY one valid JSON object with these keys:
kind (one of: "question", "guess", "other"),
feature (one supported feature key or null),
guess (one case key or null).

Supported feature keys and meanings:
{FEATURE_LIST}

Possible answer labels:
{CASE_LIST}

Rules:
- If the player asks a yes/no question about the case, choose the single best matching feature.
- If the player proposes a sleep phenomenon as their answer, classify as guess and map it to the closest case key.
- If unclear, use kind="other".
- Never invent feature keys or case keys.
- Do not include markdown, explanations, or text outside the JSON object.
- Use null when feature or guess does not apply.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message[:500]},
            ],
            temperature=0,
            max_tokens=300,
        )

        raw = response.choices[0].message.content or ""

        # Extract JSON object from the response.
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)

        if not match:
            return None

        data = json.loads(match.group(0))

        if data.get("kind") not in {"question", "guess", "other"}:
            return None

        if data.get("feature") not in FEATURES:
            data["feature"] = None

        if data.get("guess") not in CASES:
            data["guess"] = None

        return data

    except Exception as e:
        print(f"LLM interpreter error: {e}")
        return None


def fallback_interpret(message: str) -> dict[str, Any]:
    """Small offline fallback so malformed input never crashes the game."""
    text = message.lower().strip()
    # Exact/near-exact answer matching.
    for key, case in CASES.items():
        if any(alias in text for alias in case["aliases"]) or case["name"].lower() in text:
            return {"kind": "guess", "feature": None, "guess": key}

    patterns = {
        "aware": ["aware", "conscious", "know what", "surroundings"],
        "move": ["move", "movement", "paralyzed", "paralysed", "body"],
        "dreaming": ["dream", "dreaming"],
        "transition": ["falling asleep", "waking up", "wake up", "asleep", "sleep transition"],
        "vivid_perception": ["see", "seeing", "hear", "hearing", "hallucination", "sensation", "perception"],
        "fear": ["afraid", "fear", "scared", "terrified"],
        "control": ["control", "influence", "change the dream"],
        "woke_up_twice": ["woke up", "wake up", "waking up", "bedroom"],
        "childhood": ["child", "childhood", "children"],
    }
    for feature, words in patterns.items():
        if any(word in text for word in words):
            return {"kind": "question", "feature": feature, "guess": None}
    return {"kind": "other", "feature": None, "guess": None}


def entropy(values: list[bool]) -> float:
    """Binary entropy in bits."""
    if not values:
        return 0.0
    p = sum(values) / len(values)
    if p in (0, 1):
        return 0.0
    import math
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def information_gain(feature: str) -> float:
    """Expected entropy reduction from asking about this feature."""
    values = [bool(case["facts"][feature]) for case in CASES.values()]
    return entropy(values)


def quality_label(score: float) -> str:
    if score >= 0.85:
        return "Excellent"
    if score >= 0.60:
        return "Strong"
    if score >= 0.35:
        return "Useful"
    return "Low information"


def start_game():
    case_key = random.choice(list(CASES.keys()))
    st.session_state.current_game = {
        "case_key": case_key,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "messages": [{
            "role": "assistant",
            "content": "A new case is open. Ask me yes/no questions to investigate, or make your final guess. You have 12 questions/guesses.",
        }],
        "guesses": 0,
        "question_count": 0,
        "asked_features": [],
        "quality_scores": [],
        "finished": False,
        "won": False,
    }


def finish_game(won: bool):
    game = st.session_state.current_game
    game["finished"] = True
    game["won"] = won
    game["ended_at"] = datetime.now().isoformat(timespec="seconds")
    # Save a compact record for the Stats page.
    avg_quality = (
        sum(game["quality_scores"]) / len(game["quality_scores"])
        if game["quality_scores"] else 0.0
    )
    st.session_state.history.append({
        "game": len(st.session_state.history) + 1,
        "case": CASES[game["case_key"]]["name"],
        "guesses": game["guesses"],
        "questions": game["question_count"],
        "won": won,
        "quality": round(avg_quality * 100),
        "date": game["ended_at"],
    })


def process_message(message: str):
    game = st.session_state.current_game
    clean = message.strip()
    if not clean:
        return
    if len(clean) > 500:
        clean = clean[:500]

    game["messages"].append({"role": "user", "content": clean})
    interpreted = call_interpreter(clean)
    if interpreted is None:
        interpreted = fallback_interpret(clean)

    if interpreted["kind"] == "guess":
        game["guesses"] += 1
        guessed_key = interpreted.get("guess")
        if guessed_key == game["case_key"]:
            reply = "Correct. Case solved. You identified the phenomenon."
            game["messages"].append({"role": "assistant", "content": reply})
            finish_game(True)
        else:
            guessed_name = CASES[guessed_key]["name"] if guessed_key else "I couldn't identify that guess"
            reply = f"Not quite. Your guess ({guessed_name}) does not fit the case. Keep investigating."
            game["messages"].append({"role": "assistant", "content": reply})
    elif interpreted["kind"] == "question" and interpreted.get("feature"):
        feature = interpreted["feature"]
        game["question_count"] += 1
        answer = game["case_key"] and CASES[game["case_key"]]["facts"][feature]
        reply = "Yes." if answer else "No."
        if feature in game["asked_features"]:
            reply += " (You have already investigated this clue.)"
        game["asked_features"].append(feature)

        # Score the question by how much uncertainty it can remove in the case library.
        gain = information_gain(feature)
        novelty = 0.65 if feature in game["asked_features"][:-1] else 1.0
        score = min(1.0, gain / 0.9183) * novelty  # max binary entropy for 6 cases ≈ 0.918 bits
        game["quality_scores"].append(score)
        reply += f"\n\nClue quality: **{quality_label(score)}** ({round(score * 100)}/100)."
        game["messages"].append({"role": "assistant", "content": reply})
    else:
        game["messages"].append({
            "role": "assistant",
            "content": "I couldn't classify that. Try a clear yes/no question about awareness, movement, dreaming, timing, fear, sensory experiences, or control — or make a final guess using a phenomenon name.",
        })

    if not game["finished"] and game["guesses"] + game["question_count"] >= 12:
        game["messages"].append({
            "role": "assistant",
            "content": f"Investigation limit reached. The case was **{CASES[game['case_key']]['name']}**.",
        })
        finish_game(False)


def render_play():
    st.title("🌙 Dream Detective")
    st.caption("CASE FILES · SLEEP & CONSCIOUSNESS INVESTIGATION")
    game = st.session_state.get("current_game")

    if game is None:
        st.markdown("### A strange experience. A hidden explanation. Your questions are the evidence.")
        st.write("Interview the case witness, identify the sleep phenomenon, and learn how informative your questions were.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Cases in the archive", len(CASES))
        col2.metric("Questions per case", "12 max.")
        col3.metric("Your tools", "Questions + deduction")
        if st.button("Open a new case", type="primary", use_container_width=True):
            start_game()
            st.rerun()
        return

    case = CASES[game["case_key"]]
    col1, col2, col3 = st.columns(3)
    col1.metric("Questions asked", game["question_count"])
    col2.metric("Guesses made", game["guesses"])
    col3.metric("Remaining turns", max(0, 12 - game["question_count"] - game["guesses"]))

    st.progress(min(1.0, (game["question_count"] + game["guesses"]) / 12))
    for message in game["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if game["finished"]:
        st.success("Case closed." if game["won"] else "Investigation complete.")
        with st.expander("Reveal the case file"):
            st.subheader(case["name"])
            st.write(case["summary"])
            st.info(case["explanation"])
        if st.button("Investigate another case", type="primary"):
            start_game()
            st.rerun()
    else:
        prompt = st.chat_input("Ask a yes/no question or make your final guess…")
        if prompt:
            process_message(prompt)
            st.rerun()


def render_stats():
    st.title("📊 Investigation Statistics")
    st.caption("A record of your deductions and question strategy.")
    history = st.session_state.history
    if not history:
        st.info("No completed cases yet. Solve a case on the Play page to start building your statistics.")
        return

    df = pd.DataFrame(history)
    total = len(df)
    average = df["guesses"].mean()
    win_rate = df["won"].mean() * 100
    avg_quality = df["quality"].mean()

    a, b, c, d = st.columns(4)
    a.metric("Cases completed", total)
    b.metric("Average guesses", f"{average:.1f}")
    c.metric("Solved", f"{win_rate:.0f}%")
    d.metric("Avg. question quality", f"{avg_quality:.0f}/100")

    st.subheader("Guesses per case")
    chart_df = df.set_index("game")[["guesses"]]
    st.bar_chart(chart_df, y="guesses", use_container_width=True)

    st.subheader("Question strategy")
    st.write("Quality is estimated from the information gain of the feature asked about. A question scores higher when its yes/no answer divides the case archive into more balanced groups. Repeated questions receive a small penalty.")
    quality_df = df[["game", "case", "questions", "guesses", "quality", "won"]].copy()
    quality_df["won"] = quality_df["won"].map({True: "Solved", False: "Not solved"})
    quality_df = quality_df.rename(columns={
        "game": "Game", "case": "Hidden phenomenon", "questions": "Questions",
        "guesses": "Guesses", "quality": "Question quality /100", "won": "Result",
    })
    st.dataframe(quality_df, use_container_width=True, hide_index=True)

    st.subheader("What does the score mean?")
    st.markdown(
        "- **80–100:** the question is highly informative for distinguishing cases.\n"
        "- **60–79:** the question meaningfully narrows the possibilities.\n"
        "- **35–59:** the question provides some useful information.\n"
        "- **0–34:** the question is unlikely to distinguish the cases."
    )
    st.caption("This is a transparent game metric based on the case archive, not a clinical or validated cognitive assessment.")


def main():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_game" not in st.session_state:
        st.session_state.current_game = None

    with st.sidebar:
        st.markdown("# 🌙 Dream Detective")
        st.caption("A mystery game about sleep and consciousness")
        page = st.radio("Navigation", ["Play", "Stats"], label_visibility="collapsed")
        st.divider()
        st.caption("Tip: Ask specific questions that separate several possible explanations.")
        if st.button("Reset current case", use_container_width=True):
            st.session_state.current_game = None
            st.rerun()

    if page == "Play":
        render_play()
    else:
        render_stats()


if __name__ == "__main__":
    main()
