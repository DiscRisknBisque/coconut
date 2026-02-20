from preprocessing.prosqa_nodes import extract_prosqa_nodes


def test_extract_prosqa_nodes_with_explicit_premises():
    sample = {
        "premises": ["A implies B.", "B implies C."],
        "question": "Does A imply C?",
        "steps": ["A implies B.", "B implies C.", "Therefore A implies C."],
    }

    entries, step_indices = extract_prosqa_nodes(sample)

    assert entries[0] == ("premise", "A implies B.")
    assert entries[1] == ("premise", "B implies C.")
    assert entries[2][0] == "step"
    assert entries[3][0] == "step"
    assert entries[4][0] == "step"
    assert entries[5] == ("question", "Does A imply C?")
    assert step_indices == [2, 3, 4]


def test_extract_prosqa_nodes_splits_question_when_premises_missing():
    sample = {
        "question": "A is red. B is blue. Is A red?",
        "steps": ["A is red."],
    }

    entries, step_indices = extract_prosqa_nodes(sample)

    assert entries[0] == ("premise", "A is red.")
    assert entries[1] == ("premise", "B is blue.")
    assert entries[2] == ("step", "A is red.")
    assert entries[3] == ("question", "Is A red?")
    assert step_indices == [2]

