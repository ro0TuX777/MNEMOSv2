# DFE-15 Qualitative Scoring Rubric

Use this explicit anchor guide when completing the CSV scoring sheet.

## 1-5 Correctness & Alignment (For Shadow and Baseline)
- **1 = Incorrect/Irrelevant:** Completely missed the question.
- **2 = Tangential:** Mentions related keywords but does not answer the question.
- **3 = Directionally Right:** Acknowledges the correct context but lacks complete precision.
- **4 = Mostly Correct:** Answers the query well, though perhaps slightly verbose or dense.
- **5 = Fully Precise:** Perfect, concise, and indisputably correct answer.

## 1-5 Evidence & Support Quality
- **1 = Unsupported:** No source evidence provided.
- **2 = Weak/Generic Support:** Evidence is a broad policy statement rather than specific proof.
- **3 = Acceptable Support:** Evidence proves the claim but requires manual extrapolation.
- **4 = Strong Support:** Evidence tightly matches the claim.
- **5 = Perfect Support (Excerpt Rendered):** The exact, undeniable sentence from the source document is rendered in the UI.

## 0-4 Operator Usefulness (`selected_fact_usefulness_0_4`)
- **0 = Distracting:** Takes longer to read and dismiss than to ignore.
- **1 = Neutral:** Doesn't hurt, doesn't help.
- **2 = Moderate Utility:** Saves a few seconds of mental processing.
- **3 = High Utility:** Prevents manual lookup, heavily reduces review burden.
- **4 = Mission Critical:** Instantly solves a complex multi-hop synthesis or conflicting-source resolution.

## Qualitative Overrides
If the automated system surfaced an answer that appears mechanically sound but operationally dangerous (e.g., misrepresenting a delicate reporting standard), you must flag `operator_override_yes_no = Yes`.
