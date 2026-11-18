import json
import time
from src.rag import answer

K = 6                           # how many chunks to retrieve
QUESTION_FILE = "questions_hard.json"


def clean(text):
    # make quotes, hyphens and special spaces plain so matching is fair
    text = (text.lower()
            .replace("\u2019", "'")
            .replace("\u2011", "-")
            .replace("\u2010", "-"))
    return " ".join(text.split())


with open(QUESTION_FILE, encoding="utf-8") as f:
    tests = json.load(f)

correct = 0
errors = 0
for t in tests:
    ans, _ = answer(t["question"], k=K)
    if ans.startswith("API error"):
        errors += 1
        print("ERROR -", t["question"])
        print("   ", ans[:150])
    else:
        ok = any(clean(w) in clean(ans) for w in t["keywords"])
        correct += ok
        print("PASS" if ok else "FAIL", "-", t["question"])
        if not ok:
            print("   ANSWER WAS:", ans[:300].replace("\n", " "))
    time.sleep(15)  # stay under the free API rate limit

counted = len(tests) - errors
print(f"\nAccuracy: {correct}/{counted} = {correct/counted:.0%}  "
      f"(K={K}, API errors skipped: {errors})")