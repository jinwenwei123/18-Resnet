from Resnet import *
from Dataset import *
from torch.utils.tensorboard import SummaryWriter


def val(model_epoch):
    model.load_state_dict(torch.load(f"model_resnet/model_resnet_epoch{model_epoch}.pth", weights_only=True))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy_epoch{model_epoch}: {accuracy:.2f}%")
    writer.add_scalar("Accuracy", accuracy, model_epoch)


if __name__ == '__main__':
    writer = SummaryWriter(log_dir="runs/resnet")
    for model_epoch in range(5, 101, 5):
        val(model_epoch)
