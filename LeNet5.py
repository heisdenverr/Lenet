
class Lenet(nn.Module):
  def __init__(self, num_classes=10):
    super().__init__()

    self.feature_extractor = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),
        nn.Sigmoid()        
    )

    self._dummy_input = torch.zeros((1, 1, 32, 32))
    self._feat_out = self.feature_extractor(self._dummy_input).flatten(start_dim=1).shape[1]

    self.classifier = nn.Sequential(
        nn.Linear(in_features=self._feat_out, out_features=84),
        nn.Sigmoid(),
        nn.Linear(in_features=84, out_features=num_classes)
    )

  def forward(self, x):
    x = self.feature_extractor(x)
    x = x.flatten(start_dim=1)
    x = self.classifier(x)
    return x
