from kge_utils import parse_prosqa_statement, split_sentences


def test_parse_prosqa_statement_relations():
    assert parse_prosqa_statement("Every yumpus is a zhorpus.") == (
        "yumpus",
        "subclass_of",
        "zhorpus",
    )
    assert parse_prosqa_statement("Tom is a yumpus.") == (
        "Tom",
        "instance_of",
        "yumpus",
    )


def test_parse_prosqa_statement_ignores_query_sentence():
    text = "Every yumpus is a zhorpus. Tom is a yumpus. Is Tom a jompus or zhorpus?"
    triples = [
        parsed
        for sentence in split_sentences(text)
        for parsed in [parse_prosqa_statement(sentence)]
        if parsed is not None
    ]
    assert triples == [
        ("yumpus", "subclass_of", "zhorpus"),
        ("Tom", "instance_of", "yumpus"),
    ]
