"""Train weights for the current int4 MNIST RTL model.

The forward pass mirrors the hardware: 7x7 average-pooled pixels, signed int4
weights, ReLU, rounded shifts of 6 and 5, and a 0..127 activation clamp.
"""

import argparse
import os

import torch
from torchvision.datasets import MNIST


def straight_round(value):
    rounded = torch.round(value)
    return value + (rounded - value).detach()


def quantize_weight(value):
    clipped = torch.clamp(value, -8, 7)
    return straight_round(clipped)


def requantize(value, shift):
    relu_value = torch.relu(value)
    scaled = relu_value / (1 << shift)
    quantized = torch.clamp(torch.round(scaled), 0, 127)
    return scaled + (quantized - scaled).detach()


class QuantizedNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(50, 49) * 2.0)
        self.w2 = torch.nn.Parameter(torch.randn(25, 50) * 2.0)
        self.w3 = torch.nn.Parameter(torch.randn(10, 25) * 2.0)

    def forward(self, pixels):
        w1 = quantize_weight(self.w1)
        w2 = quantize_weight(self.w2)
        w3 = quantize_weight(self.w3)
        h1 = requantize(pixels @ w1.t(), 6)
        h2 = requantize(h1 @ w2.t(), 5)
        return h2 @ w3.t()


def pooled_images(images):
    images = images.float().reshape(-1, 7, 4, 7, 4)
    return torch.round(images.mean(dim=(2, 4))).reshape(-1, 49)


def evaluate(model, pixels, labels):
    with torch.no_grad():
        predictions = model(pixels).argmax(dim=1)
    return (predictions == labels).float().mean().item()


def export_weights(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for name in ("w1", "w2", "w3"):
        weights = torch.round(torch.clamp(getattr(model, name), -8, 7)).to(torch.int32)
        rows = []
        for value in weights.flatten().tolist():
            rows.append(f"{value & 0xF:x}\n")
        filename = os.path.join(output_dir, f"weights_{name.upper()}.mem")
        with open(filename, "w", encoding="ascii") as output:
            output.writelines(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train = MNIST("/tmp/mnist-data", train=True, download=True)
    test = MNIST("/tmp/mnist-data", train=False, download=True)
    train_pixels = pooled_images(train.data)
    test_pixels = pooled_images(test.data)
    train_labels = train.targets
    test_labels = test.targets
    validation_pixels = train_pixels[-5000:]
    validation_labels = train_labels[-5000:]
    train_pixels = train_pixels[:-5000]
    train_labels = train_labels[:-5000]

    model = QuantizedNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    batches = range(0, len(train_labels), args.batch_size)
    best_accuracy = -1.0
    best_state = None

    for epoch in range(args.epochs):
        order = torch.randperm(len(train_labels))
        for start in batches:
            indices = order[start:start + args.batch_size]
            logits = model(train_pixels[indices])
            loss = torch.nn.functional.cross_entropy(logits, train_labels[indices])
            if args.l1:
                loss = loss + args.l1 * sum(parameter.abs().mean() for parameter in model.parameters())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accuracy = evaluate(model, validation_pixels, validation_labels)
        print(f"epoch={epoch + 1} validation_accuracy={accuracy:.4%}", flush=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {name: value.detach().clone()
                          for name, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    export_weights(model, args.output_dir)
    with torch.no_grad():
        values = torch.cat([
            torch.round(torch.clamp(parameter, -8, 7)).flatten()
            for parameter in model.parameters()
        ])
    print(f"test_accuracy={evaluate(model, test_pixels, test_labels):.4%}")
    print(f"nonzero_weights={(values != 0).sum().item()}/{values.numel()}")
    print(f"weight_bits={(values != 0).sum().item() * 4}")


if __name__ == "__main__":
    main()