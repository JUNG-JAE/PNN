import torch
import torch.nn as nn
import torch.nn.functional as F


class Column(nn.Module):
    def __init__(self, task_id, num_classes=10):
        super(Column, self).__init__()

        self.task_id = task_id
        self.alpha = nn.Parameter(torch.FloatTensor([0.01]))

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layer1 = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU()
        )

        self.fc_layer2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )

        self.fc_layer3 = nn.Linear(84, num_classes)

        self.v1 = nn.ModuleList([nn.Linear(12, 12, bias=False) for _ in range(self.task_id)])
        self.v2 = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(self.task_id)])
        self.v3 = nn.ModuleList([nn.Linear(120, 120, bias=False) for _ in range(self.task_id)])
        self.v4 = nn.ModuleList([nn.Linear(84, 84, bias=False) for _ in range(self.task_id)])

    def forward(self, x, lateral_columns=None):
        output_per_layer = {}

        x = self.conv_layer1(x)
        output_per_layer['conv1'] = x.clone()
        if self.task_id > 0:
           x = F.relu(x + sum(F.relu(self.v1[i](self.alpha * lateral_columns[f'task{i}']['conv1'])) for i in range(self.task_id)))

        x = self.conv_layer2(x)
        output_per_layer['conv2'] = x.clone()
        if self.task_id > 0:
            x = F.relu(x + sum(F.relu(self.v2[i](self.alpha * lateral_columns[f'task{i}']['conv2'])) for i in range(self.task_id)))

        x = x.view(x.size(0), -1)

        x = self.fc_layer1(x)
        output_per_layer['fc1'] = x.clone()
        if self.task_id > 0:
            x = F.relu(x + sum(F.relu(self.v3[i](self.alpha * lateral_columns[f'task{i}']['fc1'])) for i in range(self.task_id)))

        x = self.fc_layer2(x)
        output_per_layer['fc2'] = x.clone()
        if self.task_id > 0:
            x = F.relu(x + sum(F.relu(self.v4[i](self.alpha * lateral_columns[f'task{i}']['fc2'])) for i in range(self.task_id)))

        x = self.fc_layer3(x)

        return x, output_per_layer


class PNN(nn.Module):
    def __init__(self):
        super(PNN, self).__init__()
        self.columns = nn.ModuleList([])

    def add_column(self):
        task_id = len(self.columns)
        print(f"Add new task: {task_id}")
        new_column = Column(task_id=task_id).cuda()
        self.columns.append(new_column)

    def freeze_column(self):
        for column in self.columns:
            for params in column.parameters():
                params.requires_grad = False

    def forward(self, x, task_id):
        if task_id == 0:
            output, _ = self.columns[0].forward(x)
            return output
        else:
            activations = {}
            for column_id, column in enumerate(self.columns[:task_id + 1]):
                output, activation = column.forward(x, activations)
                activations[f"task{column_id}"] = activation
            return output












