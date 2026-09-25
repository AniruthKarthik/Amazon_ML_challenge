"""Progress counts reflect only finished workflow stages."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.workflow_progress import WorkflowProgress


class WorkflowProgressTests(unittest.TestCase):
    def test_start_finish_reuse_and_completion(self):
        output = io.StringIO()
        with redirect_stdout(output), patch(
            "src.workflow_progress.time.monotonic", side_effect=(10.0, 12.5, 20.0, 20.5)
        ):
            progress = WorkflowProgress(2)
            progress.start("first")
            progress.detail("still working")
            progress.finish()
            progress.reuse("second")
            progress.check_complete()
        self.assertEqual(progress.completed, 2)
        self.assertIn("[1/2] START first", output.getvalue())
        self.assertIn("[1/2] still working", output.getvalue())
        self.assertIn("[1/2] DONE first (2.5s; 50%)", output.getvalue())
        self.assertIn("[2/2] REUSED second (0.5s; 100%)", output.getvalue())

    def test_incomplete_or_out_of_order_stage_is_not_counted(self):
        progress = WorkflowProgress(1)
        with self.assertRaises(RuntimeError):
            progress.finish()
        progress.start("unfinished")
        with self.assertRaises(RuntimeError):
            progress.start("overlap")
        with self.assertRaises(RuntimeError):
            progress.check_complete()
        self.assertEqual(progress.completed, 0)
        progress.finish()
        progress.check_complete()
        with self.assertRaises(RuntimeError):
            progress.start("extra")

    def test_reject_zero_steps(self):
        with self.assertRaises(ValueError):
            WorkflowProgress(0)


if __name__ == "__main__":
    unittest.main()
