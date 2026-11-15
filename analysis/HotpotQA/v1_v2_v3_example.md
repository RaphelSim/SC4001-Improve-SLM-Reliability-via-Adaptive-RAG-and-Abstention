# V1, V2.1, and V3 worked example

## Main difference

| Version | Controller output | Retrieval behavior | Main goal |
|---|---|---|---|
| **V1** | Basic sufficiency decision, missing information, and next query | Top-2 each round; duplicate paragraphs may be returned | Introduce adaptive retrieval |
| **V2.1** | Requirements, established facts, citations, missing items, and query | Top-2 each round; uses retrieval history and query reformulation | Reduce premature stopping and improve multi-hop reasoning |
| **V3** | Only `search_query: null` or a follow-up query | Top-2 **unseen** paragraphs; cumulative unique evidence; repeated-query guard | Simplify the controller and eliminate duplicate retrieval |

## Example question

> Which country is the headquarters of the company that manufactured Lost Gravity?

Assume the candidate paragraphs contain:

- **Paragraph A:** Lost Gravity was manufactured by **Mack Rides**.
- **Paragraph B:** General information about Lost Gravity.
- **Paragraph C:** Mack Rides is headquartered in **Germany**.

The question requires this relation chain:

```text
Lost Gravity
    -> manufactured by Mack Rides
    -> Mack Rides is headquartered in Germany
```

## V1 behavior

Round 1 retrieves Paragraphs A and B.

The V1 controller sees that it knows the manufacturer but not the
manufacturer's country. It may return:

```json
{
  "status": "need_more_evidence",
  "missing_information": "The country where Mack Rides is headquartered",
  "search_query": "Mack Rides headquarters country"
}
```

The system performs another Top-2 retrieval.

### V1 limitation

V1 uses a relatively simple sufficiency judgment. It may incorrectly decide
that Paragraphs A and B are enough, or it may retrieve a paragraph already
seen in an earlier round.

## V2.1 behavior

Round 1 also retrieves Paragraphs A and B.

V2.1 explicitly decomposes the requirements:

```json
{
  "required_information": [
    "The manufacturer of Lost Gravity",
    "The headquarters country of that manufacturer"
  ],
  "established_facts": [
    {
      "requirement": "The manufacturer of Lost Gravity",
      "fact": "Lost Gravity was manufactured by Mack Rides",
      "evidence_ids": ["A"]
    }
  ],
  "missing_information_items": [
    "The headquarters country of Mack Rides"
  ],
  "status": "need_more_evidence",
  "search_query": "Mack Rides headquarters country"
}
```

This makes the missing bridge relation explicit. If the next search produces
no new evidence and the controller repeats the same query, V2.1 uses its
loop-guard and asks the model to reformulate the query.

### V2.1 strength and limitation

V2.1 performs better evidence-chain checking because it records requirements,
facts, and citations. However, its structured output is more complex. In the
100-question development experiment, V2.1 recorded 37 controller errors.

## V3 behavior

Round 1 retrieves Paragraphs A and B.

The V3 controller does not output the full reasoning structure. It only
returns:

```json
{
  "search_query": "Mack Rides headquarters country"
}
```

Python then:

1. Searches using the new query.
2. Removes paragraphs already returned.
3. Returns the next two unseen paragraphs.
4. Adds them to the accumulated evidence.

If Paragraph C is retrieved, the controller sees A, B, and C and returns:

```json
{
  "search_query": null
}
```

The run stops normally.

If the controller returns the same query again:

```json
{
  "search_query": "Mack Rides headquarters country"
}
```

V3 detects that the query is repeated. It allows one regeneration attempt. If
the regenerated query is also repeated or unusable, it stops with the
`repeated_query` stop reason.

## Simple summary

- **V1:** “Do I have enough evidence? If not, search again.”
- **V2.1:** “List every required fact, prove each one with citations, and search for the missing fact.”
- **V3:** “If more evidence is needed, give me a new query; Python handles duplicates and loop control.”

V3 is simpler and more robust operationally, but it provides less diagnostic
information about why the evidence is insufficient.
