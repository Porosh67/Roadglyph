import unittest

import _path  # noqa: F401
from _util import Scripted
from roadglyph import prompts
from roadglyph.backends.mock import MockBackend
from roadglyph.config import Settings
from roadglyph.pipeline import Candidate, OCRPipeline, vote
from roadglyph.synthetic import draw_plate
from roadglyph.timing import Deadline

SETTINGS = Settings(backend="mock", max_variants=3)


def run(responder, deadline=None, delay_s=0.0, settings=SETTINGS):
    backend = MockBackend(responder, delay_s=delay_s)
    prediction = OCRPipeline(backend, settings).predict(draw_plate("7ABC123"), deadline)
    return prediction, backend


class VoteTests(unittest.TestCase):
    def test_majority_wins_using_grader_normalisation(self):
        cands = [
            Candidate("a", "", "7-ABC 123"),
            Candidate("b", "", "7ABC128"),
            Candidate("c", "", "7ABC123"),
        ]
        text, conf = vote(cands)
        self.assertEqual(text, "7-ABC 123")  # first of the winning group
        self.assertGreater(conf, 0.5)

    def test_tie_goes_to_the_earliest_candidate(self):
        text, _ = vote([Candidate("a", "", "AAA111"), Candidate("b", "", "BBB222")])
        self.assertEqual(text, "AAA111")

    def test_all_empty(self):
        self.assertEqual(vote([Candidate("a", "", ""), Candidate("b", "", " ")]), ("", 0.0))
        self.assertEqual(vote([]), ("", 0.0))

    def test_confidence_is_bounded(self):
        _, conf = vote([Candidate(str(i), "", "X1") for i in range(5)])
        self.assertLessEqual(conf, 0.99)
        self.assertGreater(conf, 0.0)


class PipelineTests(unittest.TestCase):
    def test_two_agreeing_variants_stop_early(self):
        prediction, backend = run(
            Scripted(prompts.PLATE, ["CALIFORNIA 7ABC123", "7ABC123", "WRONG"])
        )
        self.assertEqual(prediction.text, "7ABC123")
        self.assertEqual(prediction.category, prompts.PLATE)
        self.assertEqual(len(backend.calls), 3)  # classify + 2 transcriptions
        self.assertGreater(prediction.confidence, 0.9)

    def test_disagreement_is_resolved_by_a_third_look(self):
        prediction, backend = run(Scripted(prompts.PLATE, ["7ABC128", "7ABC123", "7ABC123"]))
        self.assertEqual(prediction.text, "7ABC123")
        self.assertEqual(len(backend.calls), 4)

    def test_expired_deadline_still_answers_once(self):
        prediction, backend = run(Scripted(prompts.STOP, ["STOP"]), deadline=Deadline(0.0))
        self.assertEqual(prediction.text, "STOP")
        self.assertEqual(len(prediction.candidates), 1)
        self.assertTrue(any("time budget" in n for n in prediction.notes))
        self.assertEqual(len(backend.calls), 2)  # classify + one transcription

    def test_classification_failure_falls_back_to_generic_prompt(self):
        prediction, _ = run(Scripted(RuntimeError("boom"), ["ROAD\nWORK"]))
        self.assertEqual(prediction.category, prompts.OTHER)
        self.assertEqual(prediction.text, "ROAD WORK")

    def test_total_backend_failure_yields_empty_answer_not_an_exception(self):
        prediction, _ = run(Scripted(RuntimeError("x"), [RuntimeError("down")]))
        self.assertEqual(prediction.text, "")
        self.assertEqual(prediction.confidence, 0.0)
        self.assertTrue(prediction.notes)

    def test_empty_replies_keep_trying_other_variants(self):
        prediction, _ = run(Scripted(prompts.ADVISORY_PLAQUE, ["", "35", "35"]))
        self.assertEqual(prediction.text, "35")

    def test_prediction_serialises(self):
        prediction, _ = run(Scripted(prompts.STOP, ["STOP", "STOP"]))
        data = prediction.to_dict()
        self.assertEqual(data["text"], "STOP")
        self.assertIsInstance(data["candidates"], list)


if __name__ == "__main__":
    unittest.main()
