import torch
import torch.nn as nn


# class SimpleNN(nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(3*32*32, 512)

#     def forward(self, x):
#         # x = x.view(-1, 3*32*32)  # Flatten the input tensor
#         # x = self.fc1(x)
#         x = torch.randn(1, 512) # get random data bc why not?
#         return x


if __name__ == "__main__":

    # model = SimpleNN()

    model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3 * 32 * 32, 512),
            )

    # input_tensor = torch.randn(1, 3, 32, 32)
    input_tensor = torch.ones(1, 3, 32, 32)
    output = model(input_tensor)

    print("Output size:", output.size())
    print("Sample output values:", output)

    torch.onnx.export(
        model,
        torch.randn(1, 3, 32, 32),
        "example_submission.onnx",
        export_params=True,
        input_names=["x"],
    )
