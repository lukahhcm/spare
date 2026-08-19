from reasoning_prune.policy import structured_evidence


def test_structured_evidence_keeps_complete_candidate_list():
    text = "\n".join(
        f"- Day {index}: {index + 10}°C ({index + 20}° - 10°)"
        for index in range(1, 8)
    )

    evidence = structured_evidence(text)

    assert "Day 1" in evidence
    assert "Day 7" in evidence


def test_structured_evidence_drops_enumerators_and_imperative_plans():
    text = "\n".join([
        "1.",
        "Add the price of Broccoli: 4.3",
        "The Broccoli price shown is 4.3.",
    ])

    evidence = structured_evidence(text)

    assert "\n- 1." not in evidence
    assert "Add the price" not in evidence
    assert "The Broccoli price shown is 4.3." in evidence
