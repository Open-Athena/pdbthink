"""Answer parsing and deterministic scoring (specification sections 7 and 12)."""

from __future__ import annotations

import pytest

from pdbthink.scoring import score_response
from pdbthink.scoring.parse import (
    AnswerFormatError,
    canonical_atom,
    canonical_pair,
    canonical_residue,
    extract_final,
    looks_like_refusal,
    parse_answer,
)
from pdbthink.scoring.scorers import set_scores


class TestExtractFinal:
    @pytest.mark.parametrize(
        "text",
        [
            "FINAL: A:V22",
            "some reasoning\nFINAL: A:V22",
            "**FINAL:** A:V22",
            "`FINAL: A:V22`",
            "> FINAL: A:V22",
            "FINAL:A:V22",
            "final: a:v22",
        ],
    )
    def test_single_line_variants(self, text):
        inline, fields = extract_final(text)
        assert canonical_residue(inline) == "A:V22"
        assert fields == {}

    def test_last_final_wins(self):
        inline, _ = extract_final("FINAL: A:V1\nactually no\nFINAL: A:V22")
        assert canonical_residue(inline) == "A:V22"

    def test_multi_field_block(self):
        inline, fields = extract_final(
            "reasoning\n\nFINAL\nchanged_residues: A:W422\nmechanism: A\n"
        )
        assert inline == ""
        assert fields == {"changed_residues": "A:W422", "mechanism": "A"}

    def test_wrapped_list_is_joined(self):
        inline, _ = extract_final("FINAL: A:D18, A:E21,\nB:Y44")
        assert inline == "A:D18, A:E21, B:Y44"

    def test_value_on_the_line_after_the_marker(self):
        # Models routinely write the marker and put the value underneath.
        assert extract_final("FINAL\nA:V22")[0] == "A:V22"
        assert extract_final("working...\n\n### Final Answer:\n**2.48**")[0] == "2.48"

    def test_a_one_word_answer_is_not_mistaken_for_a_label(self):
        assert extract_final("FINAL: helix")[0] == "helix"
        assert extract_final("FINAL: yes")[0] == "yes"

    def test_a_marker_free_answer_is_still_a_format_error(self):
        # "**Answer:** A:V22" does not follow the required convention.
        with pytest.raises(AnswerFormatError):
            extract_final("The closest residue is **Answer:** A:V22")

    def test_missing_field_is_an_error(self):
        with pytest.raises(AnswerFormatError):
            extract_final("I think it is A:V22.")

    def test_refusals_are_detected(self):
        assert looks_like_refusal("I cannot determine this without tools.")
        assert not looks_like_refusal("FINAL: A:V22")


class TestCanonicalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [("A:V22", "A:V22"), ("a: v 22", "A:V22"), ("A:V022", "A:V22"), ("**A:V22**", "A:V22")],
    )
    def test_residues(self, raw, expected):
        assert canonical_residue(raw) == expected

    def test_entities(self):
        assert canonical_residue("M:ZN501") == "M:ZN501"
        assert canonical_residue("L:L2401") == "L:L2401"

    def test_atoms(self):
        assert canonical_atom("A:H57:NE2") == "A:H57:NE2"
        assert canonical_atom("a:h57:ne2") == "A:H57:NE2"

    def test_pairs_are_unordered(self):
        assert canonical_pair("B:C81--A:C24") == canonical_pair("A:C24--B:C81")

    def test_bad_tokens_raise(self):
        for bad in ("22", "A:", "hello", "A:V22:X:Y"):
            with pytest.raises(AnswerFormatError):
                canonical_residue(bad)


class TestSchemaParsing:
    @pytest.mark.parametrize(
        "schema,text,expected",
        [
            ("string_set", "FINAL: A, B", ["A", "B"]),
            ("integer", "FINAL: 128", 128),
            ("distance", "FINAL: 3.42 Å", 3.42),
            ("numeric_triple", "FINAL: 12.481, -3.117, 8.226", [12.481, -3.117, 8.226]),
            ("residue", "FINAL: A:V22", "A:V22"),
            ("atom", "FINAL: A:H57:NE2", "A:H57:NE2"),
            ("residue_set", "FINAL: A:D18, A:E21 and B:Y44", ["A:D18", "A:E21", "B:Y44"]),
            ("residue_pair", "FINAL: A:C24--B:C81", "A:C24--B:C81"),
            ("boolean", "FINAL: yes", True),
            ("multiple_choice", "FINAL: B", "B"),
            ("ordered_path", "FINAL: A:R10 -> A:F42 -> B:E77", ["A:R10", "A:F42", "B:E77"]),
        ],
    )
    def test_round_trip(self, schema, text, expected):
        parsed = parse_answer(text, schema)
        assert not parsed.format_error, parsed.error
        assert parsed.value == expected

    def test_empty_set_is_allowed(self):
        assert parse_answer("FINAL: none", "residue_set").value == []

    def test_two_interaction_sets(self):
        parsed = parse_answer(
            "FINAL\ngained: A:R10--B:E77, A:Y15--L:L401\nlost: A:D22--A:K91\n",
            "two_interaction_sets",
        )
        assert parsed.value == {
            "gained": ["A:R10--B:E77", "A:Y15--L:L401"],
            "lost": ["A:D22--A:K91"],
        }

    def test_two_interaction_sets_needs_both_fields(self):
        parsed = parse_answer("FINAL\ngained: A:R10--B:E77\n", "two_interaction_sets")
        assert parsed.format_error
        assert "lost" in parsed.error

    def test_category_is_matched_against_the_allowed_list(self):
        parsed = parse_answer("FINAL: helix", "category", {"categories": ["helix", "strand", "coil"]})
        assert parsed.value == "helix"
        bad = parse_answer("FINAL: alpha", "category", {"categories": ["helix", "strand", "coil"]})
        assert bad.format_error

    def test_multi_field(self):
        parsed = parse_answer(
            "FINAL\nchanged_residue: A:W422\ngained_interactions: A:W422--A:Y177\nmechanism: A\n",
            "multi_field",
            {
                "field_schemas": {
                    "changed_residue": "residue",
                    "gained_interactions": "residue_pair_set",
                    "mechanism": "multiple_choice",
                }
            },
        )
        assert parsed.value["changed_residue"] == "A:W422"
        assert parsed.value["mechanism"] == "A"


class TestScorers:
    def test_exact_schemas(self):
        for schema, gold, answer in [
            ("residue", {"value": "A:V22"}, "FINAL: A:V22"),
            ("atom", {"value": "A:H57:NE2"}, "FINAL: A:H57:NE2"),
            ("category", {"value": "helix"}, "FINAL: helix"),
            ("multiple_choice", {"value": "B"}, "FINAL: B"),
            ("boolean", {"value": True}, "FINAL: yes"),
            ("integer", {"value": 46}, "FINAL: 46"),
        ]:
            out = score_response(answer, schema, gold)
            assert out["score"]["score"] == 1.0, schema

    def test_wrong_answer_scores_zero(self):
        out = score_response("FINAL: A:V23", "residue", {"value": "A:V22"})
        assert out["score"]["score"] == 0.0

    def test_distance_tolerance(self):
        gold = {"value": 3.42}
        assert score_response("FINAL: 3.43", "distance", gold)["score"]["score"] == 1.0
        assert score_response("FINAL: 3.50", "distance", gold)["score"]["score"] == 0.0

    def test_coordinate_triple_tolerance(self):
        gold = {"value": [12.481, -3.117, 8.226]}
        parameters = {"tolerance": 0.001}
        good = score_response("FINAL: 12.481, -3.117, 8.226", "numeric_triple", gold, parameters=parameters)
        assert good["score"]["score"] == 1.0
        bad = score_response("FINAL: 12.481, -3.117, 8.230", "numeric_triple", gold, parameters=parameters)
        assert bad["score"]["score"] == 0.0
        assert bad["score"]["components_within_tolerance"] == 2

    def test_set_f1_is_primary_and_exact_set_secondary(self):
        gold = {"value": ["A:D18", "A:E21", "B:Y44"]}
        out = score_response("FINAL: A:D18, A:E21", "residue_set", gold)["score"]
        assert out["score"] == pytest.approx(0.8)
        assert out["exact_set"] == 0.0
        assert out["missing"] == ["B:Y44"]

    def test_extra_items_are_penalised(self):
        gold = {"value": ["A:D18"]}
        out = score_response("FINAL: A:D18, A:E21", "residue_set", gold)["score"]
        assert out["precision"] == pytest.approx(0.5)
        assert out["spurious"] == ["A:E21"]

    def test_empty_sets_match(self):
        assert set_scores([], [])["score"] == 1.0

    def test_two_interaction_sets_average_the_two_arms(self):
        gold = {"gained": ["A:R10--B:E77"], "lost": ["A:D22--A:K91"]}
        out = score_response(
            "FINAL\ngained: A:R10--B:E77\nlost: none\n", "two_interaction_sets", gold
        )["score"]
        assert out["score"] == pytest.approx(0.5)
        assert out["gained"]["score"] == 1.0
        assert out["lost"]["score"] == 0.0

    def test_ordered_path_needs_the_exact_order(self):
        gold = {"value": ["A:R10", "A:F42", "B:E77"]}
        assert score_response("FINAL: A:R10 -> A:F42 -> B:E77", "ordered_path", gold)["score"]["score"] == 1.0
        reversed_path = score_response("FINAL: B:E77 -> A:F42 -> A:R10", "ordered_path", gold)["score"]
        assert reversed_path["score"] == 0.0
        assert reversed_path["per_position"] == pytest.approx(1 / 3)

    def test_multi_field_scores_each_field_separately(self):
        gold = {
            "fields": {
                "changed_residue": {"schema": "residue", "value": "A:W422"},
                "mechanism": {"schema": "multiple_choice", "value": "A"},
            }
        }
        out = score_response(
            "FINAL\nchanged_residue: A:W422\nmechanism: B\n",
            "multi_field",
            gold,
            parameters={"field_schemas": {"changed_residue": "residue", "mechanism": "multiple_choice"}},
        )["score"]
        assert out["fields"]["changed_residue"]["score"] == 1.0
        assert out["fields"]["mechanism"]["score"] == 0.0
        assert out["score"] == pytest.approx(0.5)

    def test_malformed_refused_and_truncated_score_zero(self):
        malformed = score_response("I am not sure.", "residue", {"value": "A:V22"})
        assert malformed["score"]["score"] == 0.0 and malformed["format_error"]

        refusal = score_response(
            "I cannot answer without tools.", "residue", {"value": "A:V22"}
        )
        assert refusal["refusal"] and refusal["score"]["score"] == 0.0

        truncated = score_response("FINAL: A:V22", "residue", {"value": "A:V22"}, truncated=True)
        assert truncated["score"]["score"] == 0.0 and truncated["truncated"]

        provider_refusal = score_response(
            "FINAL: A:V22",
            "residue",
            {"value": "A:V22"},
            provider_refusal=True,
        )
        assert provider_refusal["score"]["score"] == 0.0
        assert provider_refusal["refusal"] is True

    def test_scoring_is_deterministic(self):
        gold = {"value": ["A:D18", "A:E21"]}
        first = score_response("FINAL: A:E21, A:D18", "residue_set", gold)["score"]
        second = score_response("FINAL: A:D18, A:E21", "residue_set", gold)["score"]
        assert first["score"] == second["score"] == 1.0
