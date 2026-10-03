"""Run with: python -m unittest -v test_model"""

import unittest

import torch
import torch.nn.functional as F

from model import Attention, Transformer


class ViTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = dict(img_size=8, patch=4, in_ch=3, d_model=16,
                           head_dim=4, num_heads=4, num_classes=3, depth=2)

    def test_attention_matches_unmasked_reference(self):
        attention = Attention(8, 4, 4).double()
        x = torch.randn(2, 5, 8, dtype=torch.double)
        expected = F.scaled_dot_product_attention(
            x @ attention.W_q, x @ attention.W_k, x @ attention.W_v)
        torch.testing.assert_close(attention(x), expected)

    def test_image_classification_backward(self):
        model = Transformer(**self.config)
        images = torch.randn(2, 3, 8, 8)
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (2, 3))
        loss = F.cross_entropy(logits, torch.tensor([0, 2]))
        loss.backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        before = model.patch_emb.weight.detach().clone()
        torch.optim.SGD(model.parameters(), lr=0.1).step()
        self.assertFalse(torch.equal(before, model.patch_emb.weight))

    def test_batch_independence(self):
        model = Transformer(**self.config).eval()
        images = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            torch.testing.assert_close(model(images)[:1], model(images[:1]))

    def test_invalid_dimensions(self):
        for key in self.config:
            with self.subTest(dimension=key), self.assertRaises(ValueError):
                Transformer(**{**self.config, key: 0})
        with self.assertRaisesRegex(ValueError, 'divisible'):
            Transformer(**{**self.config, 'patch': 3})

    def test_invalid_input_shape(self):
        model = Transformer(**self.config)
        for shape in ((2, 8, 8), (2, 1, 8, 8), (2, 3, 4, 16), (2, 3, 9, 9)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, 'expected images'):
                model(torch.randn(shape))


if __name__ == '__main__':
    unittest.main()
