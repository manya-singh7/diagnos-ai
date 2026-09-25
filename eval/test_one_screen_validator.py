import logging
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))

from backend.main import (
    _normalize_and_validate_goal,
    validate_one_action_one_screen,
)
from schema import Action, ActionCategory, Goal, StepGroup

class TestOneActionOneScreenValidator(unittest.TestCase):
    def test_legitimate_single_screen_action_no_warnings(self):
        """A normal single-screen action with 4 steps should pass without warnings."""
        goal_dict = {
            "goal": "Follow these steps to perform this Navigation Troubleshooting",
            "title": "Navigation bar settings",
            "score": 0.95,
            "actions": [
                {
                    "actionName": "Configure Navigation Bar",
                    "description": "It will configure your navigation type",
                    "category": "auto",
                    "stepGroups": [
                        {
                            "steps": [
                                "Open Settings.",
                                "Tap Display.",
                                "Tap Navigation bar.",
                                "Select Swipe gestures.",
                            ]
                        }
                    ],
                }
            ],
        }
        with self.assertLogs("diagnos_ai", level="WARNING") as cm:
            # Force a benign log just to ensure assertLogs context works if no warning is emitted
            logging.getLogger("diagnos_ai").warning("sentinel")
            goal = _normalize_and_validate_goal(goal_dict)

        # Only the sentinel log should be present
        filtered_warnings = [log for log in cm.output if "One Action = One Screen" in log]
        self.assertEqual(len(filtered_warnings), 0)
        self.assertIsInstance(goal, Goal)

    def test_multi_screen_bundling_triggers_warning(self):
        """Action bundling both Display and Battery screens should log a warning."""
        goal_dict = {
            "goal": "Follow these steps to perform this Power Troubleshooting",
            "title": "Display battery options",
            "score": 0.9,
            "actions": [
                {
                    "actionName": "Configure Display And Battery",
                    "description": "It will adjust multiple hardware screens",
                    "category": "auto",
                    "stepGroups": [
                        {
                            "steps": [
                                "Open Settings and tap Display.",
                                "Adjust Brightness to comfortable level.",
                                "Go to Battery and tap Power saving.",
                            ]
                        }
                    ],
                }
            ],
        }
        with self.assertLogs("diagnos_ai", level="WARNING") as cm:
            goal = _normalize_and_validate_goal(goal_dict)

        bundling_warnings = [log for log in cm.output if "bundles multiple screens" in log]
        self.assertGreaterEqual(len(bundling_warnings), 1)
        self.assertIn("Configure Display And Battery", bundling_warnings[0])
        self.assertIsInstance(goal, Goal)  # Non-blocking: returns Goal successfully

    def test_overbundled_steps_triggers_warning(self):
        """Action exceeding 7 steps (8 steps) should log a warning."""
        goal_dict = {
            "goal": "Follow these steps to perform this Screen Troubleshooting",
            "title": "Display deep configuration",
            "score": 0.85,
            "actions": [
                {
                    "actionName": "Configure Screen Timeout Deeply",
                    "description": "It will adjust all display options",
                    "category": "auto",
                    "stepGroups": [
                        {
                            "steps": [
                                "Open Settings.",
                                "Tap Display.",
                                "Scroll down to Screen timeout.",
                                "Tap Screen timeout.",
                                "Select 2 minutes.",
                                "Tap Back.",
                                "Verify timeout updated.",
                                "Turn screen off and on.",
                            ]
                        }
                    ],
                }
            ],
        }
        with self.assertLogs("diagnos_ai", level="WARNING") as cm:
            goal = _normalize_and_validate_goal(goal_dict)

        step_warnings = [log for log in cm.output if "threshold > 7" in log]
        self.assertGreaterEqual(len(step_warnings), 1)
        self.assertIn("8 steps", step_warnings[0])
        self.assertIsInstance(goal, Goal)

    def test_unnecessary_fragmentation_triggers_warning(self):
        """Two actions in the same goal fragmenting the exact same subscreen should log a warning."""
        goal_dict = {
            "goal": "Follow these steps to perform this Navigation Troubleshooting",
            "title": "Navigation bar settings",
            "score": 0.9,
            "actions": [
                {
                    "actionName": "Configure Navigation Bar Gestures",
                    "description": "It will select navigation gestures mode",
                    "category": "auto",
                    "stepGroups": [
                        {"steps": ["Open Settings.", "Tap Display.", "Tap Navigation bar."]}
                    ],
                },
                {
                    "actionName": "Configure Navigation Bar Buttons",
                    "description": "It will select navigation buttons mode",
                    "category": "auto",
                    "stepGroups": [
                        {"steps": ["Open Settings.", "Tap Display.", "Tap Navigation bar."]}
                    ],
                },
            ],
        }
        with self.assertLogs("diagnos_ai", level="WARNING") as cm:
            goal = _normalize_and_validate_goal(goal_dict)

        frag_warnings = [log for log in cm.output if "unnecessarily fragment one screen" in log]
        self.assertGreaterEqual(len(frag_warnings), 1)
        self.assertIsInstance(goal, Goal)

    def test_passive_concept_mention_no_false_positive(self):
        """Mentioning 'battery' in a Display action without navigation verbs should not warn."""
        goal_dict = {
            "goal": "Follow these steps to perform this Display Troubleshooting",
            "title": "Dark mode settings",
            "score": 0.9,
            "actions": [
                {
                    "actionName": "Enable Dark Mode",
                    "description": "It will switch display dark theme",
                    "category": "auto",
                    "stepGroups": [
                        {
                            "steps": [
                                "Open Settings.",
                                "Tap Display.",
                                "Turn on Dark mode to help save battery power on AMOLED screens.",
                            ]
                        }
                    ],
                }
            ],
        }
        with self.assertLogs("diagnos_ai", level="WARNING") as cm:
            logging.getLogger("diagnos_ai").warning("sentinel")
            goal = _normalize_and_validate_goal(goal_dict)

        bundling_warnings = [log for log in cm.output if "bundles multiple screens" in log]
        self.assertEqual(len(bundling_warnings), 0)
        self.assertIsInstance(goal, Goal)

if __name__ == "__main__":
    unittest.main(verbosity=2)
