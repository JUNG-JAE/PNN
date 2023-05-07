# ----------- Learning library ----------- #
import torch
import torch.nn as nn
import torch.optim as optim

# ------------ system library ------------ #
from tqdm import tqdm

# ------------ custom library ------------ #
from PNN import PNN
from data_loader import source_dataloader, task_dataloader
from conf.global_settings import LEARNING_EPOCH, BATCH_SIZE, LEARNING_RATE


def train(device, train_loader, loss_function, model, task_id):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

    for epoch in range(LEARNING_EPOCH):
        progress = tqdm(total=len(train_loader.dataset), ncols=100)

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, task_id)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            progress.update(BATCH_SIZE)

        progress.close()


@torch.no_grad()
def evaluate(device, test_loader, loss_function, model, task_id):
    model.eval()

    test_loss = 0.0
    correct = 0.0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs, task_id)
        loss = loss_function(outputs, targets)

        test_loss += loss.item()
        _, predicts = outputs.max(1)
        correct += predicts.eq(targets).sum()

    print(f"Accuracy: {correct.float() * 100 / len(test_loader.dataset):.2f}, Average loss: {test_loss / len(test_loader.dataset):.2f}")


device = torch.device('cuda')
model = PNN().cuda()

loss_function = nn.CrossEntropyLoss()

model.add_column()
task0_train_loader, task0_test_loader = task_dataloader(task_id="task0")
train(device, task0_train_loader, loss_function, model, 0)
evaluate(device, task0_test_loader, loss_function, model, 0)
model.freeze_column()

model.add_column()
task1_train_loader, task1_test_loader = task_dataloader(task_id="task1")
train(device, task1_train_loader, loss_function, model, 1)
evaluate(device, task1_test_loader, loss_function, model, 1)
model.freeze_column()

model.add_column()
task2_train_loader, task2_test_loader = task_dataloader(task_id="task2")
train(device, task2_train_loader, loss_function, model, 2)
evaluate(device, task2_test_loader, loss_function, model, 2)
model.freeze_column()

model.add_column()
task3_train_loader, task3_test_loader = task_dataloader(task_id="task3")
train(device, task3_train_loader, loss_function, model, 3)
evaluate(device, task3_test_loader, loss_function, model, 3)
model.freeze_column()

model.add_column()
task4_train_loader, task4_test_loader = task_dataloader(task_id="task4")
train(device, task4_train_loader, loss_function, model, 4)
evaluate(device, task4_test_loader, loss_function, model, 4)
model.freeze_column()

torch.save(model.state_dict(), f"pnn.pt")

print(" ")
evaluate(device, task0_test_loader, loss_function, model, 0)
evaluate(device, task1_test_loader, loss_function, model, 1)
evaluate(device, task2_test_loader, loss_function, model, 2)
evaluate(device, task3_test_loader, loss_function, model, 3)
evaluate(device, task4_test_loader, loss_function, model, 4)

print("------------------------------------------------------")
_, test_loader = source_dataloader()
evaluate(device, test_loader, loss_function, model, 0)
evaluate(device, test_loader, loss_function, model, 1)
evaluate(device, test_loader, loss_function, model, 2)
evaluate(device, test_loader, loss_function, model, 3)
evaluate(device, test_loader, loss_function, model, 4)