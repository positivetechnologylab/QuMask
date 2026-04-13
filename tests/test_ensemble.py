"""
Tests for model/ensemble.py

Covers train_single_member, train_all, QuMaskEnsemble.predict, save/load_ensemble.
Training tests are slow (they run actual gradient steps). Inference-only tests
on pre-built ensembles with random weights are fast.


class TestQuMaskEnsemblePredict:
    # Build a QuMaskEnsemble with M=3 randomly-initialized (untrained) models.
    # Test output dict has keys: "p_hat", "sigma", "member_preds".
    # Test shapes: p_hat (batch, 2**k), sigma (batch, 2**k),
    #   member_preds (M=3, batch, 2**k).
    # Test that p_hat sums to 1.0 per row (it is the mean of softmax outputs).
    # Test that sigma is non-negative everywhere.
    # Test that sigma > 0 for at least some bins — random-weight models will
    #   disagree, so the ensemble should not be degenerate (all members identical).
    # Test M property returns the correct member count.


class TestTrainSingleMember:
    # @pytest.mark.slow — requires tiny_dataset_path fixture.
    # Train for 3 epochs on the 20-instance tiny dataset.
    # Test that training loss at epoch 3 is strictly lower than at epoch 1
    #   (the model is learning on this data, even minimally).
    # Test that the returned history dict has keys "train_loss" and "val_loss",
    #   each a list of length == number of epochs run.
    # Test that all loss values are finite (no NaN, no inf).
    # Test early stopping: configure patience=1 and a val set that produces
    #   non-decreasing loss; verify training stops before the max epoch count.


class TestCheckpointRoundTrip:
    # @pytest.mark.slow — requires training.
    # Train a single member for 2 epochs, save it, reload it, and run predict
    #   on the same input. Verify the output is bit-for-bit identical before and
    #   after the checkpoint round-trip.
    # Test that loading from a non-existent path raises an appropriate error.


class TestTrainAll:
    # @pytest.mark.slow — runs M=2 members for 2 epochs each.
    # Test that train_all returns a QuMaskEnsemble with M members.
    # Test that M checkpoint files exist at the expected paths after training.
    # Test that the two members have different weights (different seeds →
    #   different initialization → different trained weights). Compare state_dicts.


class TestLoadEnsemble:
    # @pytest.mark.slow — requires saved checkpoints from a prior training run.
    # Test that load_ensemble reconstructs a QuMaskEnsemble with the correct M.
    # Test that all members are in eval mode (model.training == False).
    # Test that predict output matches what was produced before saving
    #   (checkpoint round-trip, same as TestCheckpointRoundTrip but at the
    #   ensemble level).
"""
