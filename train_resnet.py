from Dataset import *
from Resnet import *
import os
from torch.utils.tensorboard import SummaryWriter


def main(epochs: int):
    if os.path.exists("model_resnet/model_resnet.pth"):
        model.load_state_dict(torch.load("model_resnet/model_resnet.pth", weights_only=True))

    for i in range(epochs):
        model.train()
        running_loss = 0.0
        for idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            writer.add_scalar("Loss_batch", loss.item(), i * len(train_dataloader) + idx)
        writer.add_scalar("Loss_epoch", running_loss / len(train_dataloader), i)
        print(f"Epoch [{i + 1}/{epochs}], Loss: {running_loss / len(train_dataloader):.4f}")
        if (i + 1) % 5 == 0:
            torch.save(model.state_dict(), f"model_resnet/model_resnet_epoch{i + 1}.pth")


if __name__ == '__main__':
    epoches = 100  # 训练轮次
    print(f"Using device: {device}")
    writer = SummaryWriter(log_dir="runs/resnet")  # log_dir 是日志保存目录
    main(epoches)
    dummy_input = torch.randn(64, 3, 32, 32).to(device)
    writer.add_graph(model, dummy_input)
