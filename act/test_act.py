"""CPU checks: python -m unittest test_act.py."""

import unittest

import torch

from act import ACT, TemporalEnsemble, act_loss


class ACTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)
        self.model = ACT(state_dim=3, action_dim=2, chunk_size=4, num_cameras=2,
                         hidden_dim=32, encoder_layers=1, decoder_layers=1, dropout=0)
        self.qpos = torch.randn(2, 3)
        self.images = torch.rand(2, 2, 3, 24, 32)
        self.actions = torch.randn(2, 4, 2)
        self.pad = torch.tensor([[False, False, True, True], [False] * 4])

    def test_training_gradients_and_inference(self):
        pred, mu, logvar = self.model(self.qpos, self.images, self.actions, self.pad)
        loss = act_loss(pred, self.actions, mu, logvar, self.pad)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for component in (self.model.backbone[0], self.model.to_distribution, self.model.action_head):
            self.assertTrue(torch.isfinite(component.weight.grad).all())
            self.assertGreater(component.weight.grad.abs().sum().item(), 0)
        self.model.eval()
        with torch.no_grad():
            first, mu, logvar = self.model(self.qpos, self.images)
            second, _, _ = self.model(self.qpos, self.images)
        self.assertEqual(first.shape, (2, 4, 2))
        self.assertIsNone(mu)
        self.assertIsNone(logvar)
        torch.testing.assert_close(first, second)

    def test_padding_does_not_change_posterior_or_loss(self):
        modified = self.actions.masked_fill(self.pad[..., None], float("nan"))
        torch.manual_seed(5)
        first = self.model(self.qpos, self.images, self.actions, self.pad)
        torch.manual_seed(5)
        second = self.model(self.qpos, self.images, modified, self.pad)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)
        a = act_loss(first[0], self.actions, first[1], first[2], self.pad)
        b = act_loss(second[0], modified, second[1], second[2], self.pad)
        torch.testing.assert_close(a["loss"], b["loss"])
        with self.assertRaises(ValueError):
            act_loss(first[0], self.actions, first[1], first[2], torch.ones_like(self.pad))

    def test_temporal_ensemble_alignment_and_zero_actions(self):
        ensemble = TemporalEnsemble(decay=0)
        torch.testing.assert_close(ensemble(torch.tensor([[1.], [0.]])), torch.tensor([1.]))
        torch.testing.assert_close(ensemble(torch.tensor([[4.], [6.]])), torch.tensor([2.]))
        torch.testing.assert_close(ensemble(torch.tensor([[2.], [9.]])), torch.tensor([4.]))
        ensemble.reset()
        torch.testing.assert_close(ensemble(torch.tensor([[7.]])), torch.tensor([7.]))


if __name__ == "__main__":
    unittest.main()
