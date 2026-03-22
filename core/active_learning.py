"""
active_learning.py
------------------
Implements active learning feedback loop.
Users can mark false positives/negatives which get
stored and used to improve the model over time.

Two components:
1. FeedbackStore  — saves and loads user feedback
2. ActiveLearner  — selects uncertain samples for retraining
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from datetime import datetime


# ── Config ────────────────────────────────────────────────────────────────────
FEEDBACK_FILE    = 'data/feedback.json'
RETRAIN_THRESHOLD = 50    # retrain after this many feedback items


# ────────────────────────────────────────────────────────────────────────────
class FeedbackStore:
    """
    Stores and manages user feedback on scan results.

    Each feedback item contains:
    - code          : the scanned code
    - prediction    : what the model predicted
    - correct_label : what the user says is correct
    - feedback_type : 'false_positive' or 'false_negative'
    - timestamp     : when feedback was given
    - confidence    : model's confidence score
    """

    def __init__(self, feedback_file: str = FEEDBACK_FILE):
        self.feedback_file = feedback_file
        self.feedback      = []
        self._load()

    def _load(self):
        """Loads existing feedback from disk."""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    self.feedback = json.load(f)
                print(f"  FeedbackStore: loaded {len(self.feedback)} items")
            except Exception as e:
                print(f"  FeedbackStore: could not load — {e}")
                self.feedback = []
        else:
            self.feedback = []

    def _save(self):
        """Saves feedback to disk."""
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback, f, indent=2)

    def add_feedback(
        self,
        code          : str,
        predicted_label: int,
        correct_label : int,
        confidence    : float,
        cwe           : str = 'Unknown'
    ) -> dict:
        """
        Adds a new feedback item.

        Args:
            code            : scanned code snippet
            predicted_label : what model predicted (0=safe, 1=vulnerable)
            correct_label   : what user says is correct
            confidence      : model confidence score
            cwe             : predicted CWE category

        Returns: the feedback item added
        """
        # determine feedback type
        if predicted_label == 1 and correct_label == 0:
            feedback_type = 'false_positive'  # model said vuln, actually safe
        elif predicted_label == 0 and correct_label == 1:
            feedback_type = 'false_negative'  # model said safe, actually vuln
        else:
            feedback_type = 'confirmed'       # model was correct

        item = {
            'id'             : len(self.feedback),
            'code'           : code[:500],    # truncate to save space
            'predicted_label': predicted_label,
            'correct_label'  : correct_label,
            'feedback_type'  : feedback_type,
            'confidence'     : round(confidence, 4),
            'cwe'            : cwe,
            'timestamp'      : datetime.now().isoformat(),
            'used_for_training': False
        }

        self.feedback.append(item)
        self._save()

        print(f"  Feedback saved: {feedback_type} "
              f"(confidence: {confidence:.3f})")

        return item

    def get_stats(self) -> dict:
        """Returns summary statistics of collected feedback."""
        if not self.feedback:
            return {
                'total'         : 0,
                'false_positives': 0,
                'false_negatives': 0,
                'confirmed'     : 0,
                'ready_to_train': False
            }

        fp = sum(1 for f in self.feedback if f['feedback_type'] == 'false_positive')
        fn = sum(1 for f in self.feedback if f['feedback_type'] == 'false_negative')
        co = sum(1 for f in self.feedback if f['feedback_type'] == 'confirmed')

        return {
            'total'          : len(self.feedback),
            'false_positives': fp,
            'false_negatives': fn,
            'confirmed'      : co,
            'ready_to_train' : len(self.feedback) >= RETRAIN_THRESHOLD
        }

    def get_training_samples(self) -> list:
        """
        Returns feedback items not yet used for training.
        These are the samples the active learner will retrain on.
        """
        return [f for f in self.feedback if not f['used_for_training']]

    def mark_as_trained(self):
        """Marks all current feedback as used for training."""
        for item in self.feedback:
            item['used_for_training'] = True
        self._save()
        print(f"  Marked {len(self.feedback)} items as trained")


# ────────────────────────────────────────────────────────────────────────────
class ActiveLearner:
    """
    Selects the most informative samples for retraining
    using uncertainty sampling strategy.

    Uncertainty sampling: pick samples where the model
    was least confident — these are the most valuable
    for improving the model.
    """

    def __init__(self):
        self.store = FeedbackStore()

    def get_uncertainty_score(self, confidence: float) -> float:
        """
        Calculates uncertainty score from confidence.
        Score is highest when confidence is closest to 0.5
        (model is most uncertain at 50/50).

        Args:
            confidence: model's vulnerability probability (0-1)

        Returns: uncertainty score (0-1, higher = more uncertain)
        """
        return 1.0 - abs(confidence - 0.5) * 2

    def select_samples_for_retraining(
        self,
        n_samples: int = 20
    ) -> list:
        """
        Selects the most valuable samples for retraining.

        Strategy:
        1. All false positives and false negatives (model was wrong)
        2. High uncertainty confirmed samples (model was right but not sure)
        3. Sort by uncertainty score

        Args:
            n_samples: maximum number of samples to return

        Returns: list of selected feedback items
        """
        training_samples = self.store.get_training_samples()

        if not training_samples:
            print("  No new feedback available for retraining")
            return []

        # separate errors from confirmed
        errors    = [f for f in training_samples
                     if f['feedback_type'] != 'confirmed']
        confirmed = [f for f in training_samples
                     if f['feedback_type'] == 'confirmed']

        # sort confirmed by uncertainty (most uncertain first)
        confirmed_sorted = sorted(
            confirmed,
            key=lambda x: self.get_uncertainty_score(x['confidence']),
            reverse=True
        )

        # combine: all errors first, then uncertain confirmed
        selected = errors + confirmed_sorted
        selected = selected[:n_samples]

        print(f"  Selected {len(selected)} samples for retraining")
        print(f"    Errors     : {len(errors)}")
        print(f"    Confirmed  : {len(confirmed_sorted[:n_samples-len(errors)])}")

        return selected

    def prepare_retraining_data(self, selected_samples: list) -> list:
        """
        Converts selected feedback into training format.
        Compatible with CodeBERT training pipeline.

        Returns: list of {code, label, cwe_normalized} dicts
        """
        training_data = []

        for item in selected_samples:
            training_data.append({
                'code'          : item['code'],
                'label'         : item['correct_label'],
                'cwe_normalized': item['cwe'],
            })

        return training_data

    def should_retrain(self) -> bool:
        """
        Returns True if enough feedback has been collected
        to warrant retraining.
        """
        stats = self.store.get_stats()
        return stats['ready_to_train']

    def retrain(self) -> dict:
        """
        Triggers retraining on collected feedback samples.
        Uses uncertainty sampling to pick most valuable samples.

        Returns: dict with retraining status and metrics
        """
        stats = self.store.get_stats()

        if not stats['ready_to_train']:
            remaining = RETRAIN_THRESHOLD - stats['total']
            return {
                'status' : 'not_ready',
                'message': f"Need {remaining} more feedback items before retraining",
                'stats'  : stats
            }

        print("\n" + "=" * 55)
        print("  Active Learning — Retraining Triggered")
        print(f"  Total feedback items: {stats['total']}")
        print("=" * 55)

        # select best samples
        selected = self.select_samples_for_retraining(n_samples=30)

        if not selected:
            return {
                'status' : 'no_samples',
                'message': 'No new samples available',
                'stats'  : stats
            }

        # prepare training data
        training_data = self.prepare_retraining_data(selected)

        # save to a temporary file for retraining
        retrain_path = 'data/processed/retrain.json'
        with open(retrain_path, 'w') as f:
            json.dump(training_data, f, indent=2)

        print(f"  Saved {len(training_data)} samples to {retrain_path}")
        print("  To retrain: python core/codebert_trainer.py --retrain")

        # mark feedback as used
        self.store.mark_as_trained()

        return {
            'status'        : 'ready',
            'message'       : f'Retraining data prepared with {len(training_data)} samples',
            'samples_count' : len(training_data),
            'retrain_path'  : retrain_path,
            'stats'         : stats
        }

    def get_dashboard_data(self) -> dict:
        """
        Returns data for the Streamlit feedback dashboard.
        Shows feedback trends and model improvement over time.
        """
        stats    = self.store.get_stats()
        feedback = self.store.feedback

        # recent feedback (last 10)
        recent = feedback[-10:] if len(feedback) >= 10 else feedback

        # uncertainty distribution
        confidences  = [f['confidence'] for f in feedback]
        uncertainties = [self.get_uncertainty_score(c) for c in confidences]

        # feedback type distribution
        type_dist = {
            'false_positive': stats['false_positives'],
            'false_negative': stats['false_negatives'],
            'confirmed'     : stats['confirmed']
        }

        return {
            'stats'           : stats,
            'recent_feedback' : recent,
            'type_distribution': type_dist,
            'avg_confidence'  : round(np.mean(confidences), 3) if confidences else 0,
            'avg_uncertainty' : round(np.mean(uncertainties), 3) if uncertainties else 0,
            'retrain_threshold': RETRAIN_THRESHOLD,
            'progress_to_retrain': min(
                stats['total'] / RETRAIN_THRESHOLD * 100, 100
            )
        }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing Active Learning...")
    print("=" * 55)

    learner = ActiveLearner()

    # simulate some user feedback
    test_cases = [
        {
            'code'           : 'void f(char *s) { char buf[64]; strcpy(buf, s); }',
            'predicted_label': 1,
            'correct_label'  : 1,
            'confidence'     : 0.95,
            'cwe'            : 'CWE-119',
            'type'           : 'confirmed'
        },
        {
            'code'           : 'int add(int a, int b) { return a + b; }',
            'predicted_label': 1,
            'correct_label'  : 0,
            'confidence'     : 0.82,
            'cwe'            : 'Safe',
            'type'           : 'false_positive'
        },
        {
            'code'           : 'void f(char *s) { char buf[64]; strncpy(buf,s,63); }',
            'predicted_label': 0,
            'correct_label'  : 1,
            'confidence'     : 0.45,
            'cwe'            : 'CWE-119',
            'type'           : 'false_negative'
        },
    ]

    print("\nAdding test feedback items...")
    for case in test_cases:
        learner.store.add_feedback(
            code            = case['code'],
            predicted_label = case['predicted_label'],
            correct_label   = case['correct_label'],
            confidence      = case['confidence'],
            cwe             = case['cwe']
        )

    print("\nFeedback Statistics:")
    stats = learner.store.get_stats()
    for key, val in stats.items():
        print(f"  {key:<20}: {val}")

    print("\nUncertainty Scores:")
    for case in test_cases:
        u = learner.get_uncertainty_score(case['confidence'])
        print(f"  {case['type']:<20} confidence={case['confidence']} "
              f"uncertainty={u:.3f}")

    print("\nRetraining check:")
    result = learner.retrain()
    print(f"  Status  : {result['status']}")
    print(f"  Message : {result['message']}")

    print("\nDashboard data:")
    dashboard = learner.get_dashboard_data()
    print(f"  Total feedback     : {dashboard['stats']['total']}")
    print(f"  Avg confidence     : {dashboard['avg_confidence']}")
    print(f"  Avg uncertainty    : {dashboard['avg_uncertainty']}")
    print(f"  Progress to retrain: {dashboard['progress_to_retrain']:.1f}%")