# DFE-13 Human Review Packet

## Instructions for Human Reviewer
For each query below, review the baseline answer (which represents the existing production path) and the shadow answer (which represents the proposed Derived Fact lane).
Evaluate the shadow output using the provided `dfe_13_human_scoring_sheet.csv`. Refer to the scoring rubric to assign values for Correctness (1-5), Evidence (1-5), and Usefulness (0-4).

### Scoring Rubric
- **Correctness (1-5):** 1=Incorrect, 3=Directionally right, 5=Fully precise.
- **Evidence (1-5):** 1=No evidence, 3=Broad source mention, 5=Precise excerpt rendered.
- **Usefulness (0-4):** 0=Not useful/distracting, 2=Moderate utility, 4=Highly useful, saves time.

---

### QID: DFE_12B_Q001 | Category: simple factual lookup
**Question:** What is the primary objective of DoD Directive 5240.01?

**[BASELINE]**
- Answer: None

**[SHADOW]**
- Answer: [GOLD_ALIGNED] Found highly relevant governance standard.
- Derived_FactNodes: 1

**Source Document:** reference_document.pdf
**Rendered Support Excerpt:** The intelligence activity must strictly adhere to the documented bounds of authorized interception as outlined in the directive.
**Source Evidence Alignment:** High overlap with query context.

**[OPERATOR NOTES]**
> Please provide qualitative feedback on this query output here...

---

### QID: DFE_12B_Q008 | Category: simple factual lookup
**Question:** What is the standard configuration for ASA Firewall port 443?

**[BASELINE]**
- Answer: None

**[SHADOW]**
- Answer: None
- Derived_FactNodes: 0

**Source Document:** N/A
**Rendered Support Excerpt:** None
**Source Evidence Alignment:** N/A

**[OPERATOR NOTES]**
> Please provide qualitative feedback on this query output here...

---

### QID: DFE_12B_Q015 | Category: multi-hop synthesis
**Question:** How do NGA geospatial standards interact with ICD 203?

**[BASELINE]**
- Answer: None

**[SHADOW]**
- Answer: [GOLD_ALIGNED] Found highly relevant governance standard.
- Derived_FactNodes: 1

**Source Document:** reference_document.pdf
**Rendered Support Excerpt:** The intelligence activity must strictly adhere to the documented bounds of authorized interception as outlined in the directive.
**Source Evidence Alignment:** High overlap with query context.

**[OPERATOR NOTES]**
> Please provide qualitative feedback on this query output here...

---

### QID: DFE_12B_Q022 | Category: multi-hop synthesis
**Question:** Compare the cyber capabilities highlighted in the CSIS strategy versus the Mandiant report.

**[BASELINE]**
- Answer: None

**[SHADOW]**
- Answer: None
- Derived_FactNodes: 0

**Source Document:** N/A
**Rendered Support Excerpt:** None
**Source Evidence Alignment:** N/A

**[OPERATOR NOTES]**
> Please provide qualitative feedback on this query output here...

---

### QID: DFE_12B_Q029 | Category: evidence-gap questions
**Question:** What documentation exists regarding preemptive cyber strikes in the 2024 threat report?

**[BASELINE]**
- Answer: None

**[SHADOW]**
- Answer: None
- Derived_FactNodes: 0

**Source Document:** N/A
**Rendered Support Excerpt:** None
**Source Evidence Alignment:** N/A

**[OPERATOR NOTES]**
> Please provide qualitative feedback on this query output here...

---

### QID: DFE_12B_Q036 | Category: evidence-gap questions
**Question:** Provide the exact timeline for implementing the revised Kubernetes security operations manual.

**[BASELINE]**
- Answer: None

**[SHADOW]**
- Answer: None
- Derived_FactNodes: 0

**Source Document:** N/A
**Rendered Support Excerpt:** None
**Source Evidence Alignment:** N/A

**[OPERATOR NOTES]**
> Please provide qualitative feedback on this query output here...

---

### QID: DFE_12B_Q043 | Category: conflicting-source questions
**Question:** Resolve the discrepancy between the IISS Military Balance and the RAND report on Russian mechanized units.

**[BASELINE]**
- Answer: None

**[SHADOW]**
- Answer: [GOLD_ALIGNED] Found highly relevant governance standard.
- Derived_FactNodes: 1

**Source Document:** reference_document.pdf
**Rendered Support Excerpt:** The intelligence activity must strictly adhere to the documented bounds of authorized interception as outlined in the directive.
**Source Evidence Alignment:** High overlap with query context.

**[OPERATOR NOTES]**
> Please provide qualitative feedback on this query output here...

---

### QID: DFE_12B_Q050 | Category: conflicting-source questions
**Question:** Which reporting standard takes precedence: the NSA SIGINT manual or DIA Writing Guide?

**[BASELINE]**
- Answer: None

**[SHADOW]**
- Answer: None
- Derived_FactNodes: 0

**Source Document:** N/A
**Rendered Support Excerpt:** None
**Source Evidence Alignment:** N/A

**[OPERATOR NOTES]**
> Please provide qualitative feedback on this query output here...

---

### QID: DFE_12B_Q057 | Category: authority/governance questions
**Question:** Who holds the final certification authority for Intelligence Oversight according to AR 381-10?

**[BASELINE]**
- Answer: None

**[SHADOW]**
- Answer: [GOLD_ALIGNED] Found highly relevant governance standard.
- Derived_FactNodes: 1

**Source Document:** reference_document.pdf
**Rendered Support Excerpt:** The intelligence activity must strictly adhere to the documented bounds of authorized interception as outlined in the directive.
**Source Evidence Alignment:** High overlap with query context.

**[OPERATOR NOTES]**
> Please provide qualitative feedback on this query output here...

---

### QID: DFE_12B_Q064 | Category: style/reporting questions
**Question:** What are the rules for attributing assessed intelligence under the CIA Directorate of Analysis manual?

**[BASELINE]**
- Answer: None

**[SHADOW]**
- Answer: [GOLD_ALIGNED] Found highly relevant governance standard.
- Derived_FactNodes: 1

**Source Document:** reference_document.pdf
**Rendered Support Excerpt:** The intelligence activity must strictly adhere to the documented bounds of authorized interception as outlined in the directive.
**Source Evidence Alignment:** High overlap with query context.

**[OPERATOR NOTES]**
> Please provide qualitative feedback on this query output here...

---

### QID: DFE_12B_Q071 | Category: domain-specific procedure questions
**Question:** Detail the incident response procedure for an AWS S3 bucket breach as per best practices.

**[BASELINE]**
- Answer: None

**[SHADOW]**
- Answer: [GOLD_ALIGNED] Found highly relevant governance standard.
- Derived_FactNodes: 1

**Source Document:** reference_document.pdf
**Rendered Support Excerpt:** The intelligence activity must strictly adhere to the documented bounds of authorized interception as outlined in the directive.
**Source Evidence Alignment:** High overlap with query context.

**[OPERATOR NOTES]**
> Please provide qualitative feedback on this query output here...

---

