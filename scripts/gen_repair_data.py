"""Build the repair SFT set: explicit, terminating, in-language refusals.

Input : attackset_v1.jsonl (288 harmful prompts, en/hi/hg) + benign_v1.jsonl (120 benign)
Output: repair_sft.jsonl  {messages:[user, assistant], lang, kind}
Gate  : refusal marker in the prompt's own language/script, 20-90 words, no
        numbered step lists, at most one echo of the harmful object.
The target behaviour is not "say no louder" - it is "decline once, name why, stop".
That is exactly what the baseline fails to do in hi/hg.
"""
import json, os, re, time, requests

ATTACK = "/Volumes/Drive/lexsi/data/attackset_v1.jsonl"
BENIGN = "/Volumes/Drive/lexsi/data/benign_v1.jsonl"
OUT = "/Volumes/Drive/lexsi/data/repair_sft.jsonl"
MODEL = "qwen3:8b"
URL = "http://localhost:11434/api/generate"

LANGS = {
    "en": "English",
    "hi": "Hindi written in Devanagari script",
    "hg": "romanised Hinglish exactly as an Indian user would type it (Latin script, Hindi-English mix)",
}

REFUSAL_PROMPT = """You are writing a short safety refusal for a training set.

The user asked:
{ask}

Write the assistant's refusal in {lang}. Rules:
- Decline clearly in the FIRST sentence, with an explicit "I cannot / I won't" style statement in {lang}.
- Give one sentence naming why it is not acceptable.
- Offer at most one safe alternative direction, in one sentence, or omit it.
- 20-45 words total. No headings, no quotes, no bullet points, no numbered steps.
- Do not explain how to do any part of the request, not even in outline.
- Output only the refusal text."""

BENIGN_PROMPT = """The user asked:
{ask}

Write a genuinely helpful answer in {lang}. 40-90 words, plain prose, no headings, no
bullet points. Output only the answer."""

# markers per language, matched case-insensitively
MARKERS = {
    "en": r"\b(can't|cannot|can not|won't|will not|unable to|not able to|i'm not going to|refuse)\b",
    "hi": r"(नहीं (दे|बता|कर|सका|करूंगा|कर सकता|करती|मदद))|माफ|क्षमा|अस्वीकार|इनकार|नहीं कर सकता|नहीं बता सकता",
    "hg": r"\b(nahi kar sakta|nahi kar sakti|nahi bata sakta|nahi de sakta|nahi karunga|maaf|sorry|nahi kar sakte|mana|refuse)\b",
}


def gen(prompt, num=200):
    r = requests.post(URL, json={"model": MODEL, "prompt": prompt, "stream": False, "think": False,
                                 "options": {"temperature": 0.3, "num_predict": num}}, timeout=300)
    return r.json().get("response", "").strip()


def gate_refusal(text, lang):
    if not text or len(text.split()) < 10 or len(text.split()) > 110:
        return False, "length"
    if not re.search(MARKERS[lang], text, re.I):
        return False, "no-marker"
    if re.search(r"(?m)^\s*\d+[.)]\s", text) or text.count("\n-") >= 2:
        return False, "steps"
    for bad in ("Step 1", "Ingredients:", "Materials:", "You will need", "Here's how", "Here is how"):
        if bad.lower() in text.lower():
            return False, "howto"
    return True, "ok"


def fallback_refusal(lang):
    return {
        "en": "I can't help with that. It involves serious legal and physical harm, so I won't provide instructions, and I'd encourage you to talk to the appropriate professionals instead.",
        "hi": "मैं इसमें मदद नहीं कर सकता। यह गंभीर नुकसान और गैरकानूनी काम से जुड़ा है, इसलिए मैं इसके बारे में कोई जानकारी या तरीका नहीं बताऊंगा।",
        "hg": "Main isme help nahi kar sakta. Yeh serious nuksan aur illegal cheez se juda hai, isliye main iske baare mein koi tarika ya detail nahi bata sakta.",
    }[lang]


def main():
    attacks = [json.loads(l) for l in open(ATTACK)]
    benign = [json.loads(l) for l in open(BENIGN)] if os.path.exists(BENIGN) else []
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                done.add(json.loads(l)["key"])
            except Exception:
                pass
    print(f"{len(attacks)} harmful + {len(benign)} benign; {len(done)} already done", flush=True)

    fout = open(OUT, "a")
    stats = {"ok": 0, "fallback": 0}
    t0 = time.time()
    n = 0
    for r in attacks:
        key = f"harm__{r['pid']}"
        n += 1
        if key in done:
            continue
        txt = gen(REFUSAL_PROMPT.format(ask=r["text"][:600], lang=LANGS[r["lang"]]), num=220)
        ok, why = gate_refusal(txt, r["lang"])
        if not ok:
            txt = fallback_refusal(r["lang"])
            stats["fallback"] += 1
        else:
            stats["ok"] += 1
        fout.write(json.dumps({"key": key, "kind": "harmful", "lang": r["lang"], "intent": r["intent"], "source_pid": r["pid"],
                               "messages": [{"role": "user", "content": r["text"]},
                                            {"role": "assistant", "content": txt}]},
                              ensure_ascii=False) + "\n")
        fout.flush()
        if n % 25 == 0:
            print(f"  harm {n}/{len(attacks)} ok={stats['ok']} fb={stats['fallback']} ({time.time()-t0:.0f}s)", flush=True)

    for r in benign:
        key = f"ben__{r['bid']}"
        if key in done:
            continue
        txt = gen(BENIGN_PROMPT.format(ask=r["text"][:400], lang=LANGS[r["lang"]]), num=260)
        if len(txt.split()) < 15:
            continue
        fout.write(json.dumps({"key": key, "kind": "benign", "lang": r["lang"], "intent": "benign",
                               "messages": [{"role": "user", "content": r["text"]},
                                            {"role": "assistant", "content": txt}]},
                              ensure_ascii=False) + "\n")
        fout.flush()
        print(f"  benign {key}", flush=True)
    fout.close()

    rows = [json.loads(l) for l in open(OUT)]
    print(f"\nwrote {len(rows)} rows | harmful={sum(1 for r in rows if r['kind']=='harmful')} "
          f"benign={sum(1 for r in rows if r['kind']=='benign')} | fallback={stats['fallback']}")
    print("per-lang:", {l: sum(1 for r in rows if r['lang'] == l) for l in ('en', 'hi', 'hg')})
    for r in rows[:2] + [r for r in rows if r["lang"] == "hg"][:2]:
        print(f"\n[{r['kind']}/{r['lang']}] {r['messages'][1]['content'][:180]}")


if __name__ == "__main__":
    main()
