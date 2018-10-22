import torch
import time

from loss import CustomLoss
from datagen import get_data_loader
from model import PIXOR
from utils import get_model_name, load_config, plot_bev, plot_label_map
from postprocess import non_max_suppression


def build_model(config, device, train=True):
    net = PIXOR(config['use_bn']).to(device)
    criterion = CustomLoss(device=device, num_classes=1)
    if not train:
        return net, criterion

    optimizer = torch.optim.SGD(net.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_decay_every'], gamma=0.1)

    return net, criterion, optimizer, scheduler


def validate_batch(net, criterion, batch_size, test_data_loader, device):
    net.eval()
    val_loss = 0
    num_samples = 0
    for i, data in enumerate(test_data_loader):
        input, label_map = data
        input = input.to(device)
        label_map = label_map.to(device)
        predictions = net(input)
        loss = criterion(predictions, label_map)
        val_loss += float(loss)
        num_samples += label_map.shape[0]
    return val_loss * batch_size / num_samples


def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())

def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].norm())


def train(config_name, device):
    config, learning_rate, batch_size, max_epochs = load_config(config_name)
    train_data_loader, test_data_loader = get_data_loader(batch_size=batch_size, use_npy=config['use_npy'], frame_range=config['frame_range'])
    net, criterion, optimizer, scheduler = build_model(config, device, train=True)

    if config['resume_training']:
        saved_ckpt_path = get_model_name(config['old_ckpt_name'])
        net.load_state_dict(torch.load(saved_ckpt_path, map_location=device))
        print("Successfully loaded trained ckpt at {}".format(saved_ckpt_path))

    net.train()
    #net.backbone.conv1.register_forward_hook(printnorm)
    #net.backbone.conv2.register_backward_hook(printgradnorm)

    start_time = time.time()
    for epoch in range(max_epochs):
        train_loss = 0
        num_samples = 0
        scheduler.step()
        print("Learning Rate for Epoch {} is {} ".format(epoch + 1, scheduler.get_lr()))
        for i, (input, label_map) in enumerate(train_data_loader):
            input = input.to(device)
            label_map = label_map.to(device)
            optimizer.zero_grad()
            # Forward
            predictions = net(input)
            loss = criterion(predictions, label_map)
            loss.backward()
            optimizer.step()

            train_loss += float(loss)
            num_samples += label_map.shape[0]

        train_loss = train_loss * batch_size/ num_samples

        val_loss = validate_batch(net, criterion, batch_size, test_data_loader, device)

        print("Epoch {}|Time {:.3f}|Training Loss: {}|Validation Loss: {}".format(
            epoch + 1, time.time() - start_time, train_loss, val_loss))

        if (epoch + 1) == max_epochs or (epoch + 1) % config['save_every'] == 0:
            model_path = get_model_name(config['name']+'__epoch{}'.format(epoch+1))
            torch.save(net.state_dict(), model_path)
            print("Checkpoint saved at {}".format(model_path))

    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))


def experiment(config_name, device):
    config, _, _, _ = load_config(config_name)
    net, criterion = build_model(config, device, train=False)
    net.load_state_dict(torch.load(get_model_name(config['name']), map_location=device))
    net.set_decode(True)
    loader, _ = get_data_loader(batch_size=1, use_npy=config['use_npy'], frame_range=config['frame_range'])
    net.eval()

    image_id = 25
    threshold = config['cls_threshold']

    with torch.no_grad():
        input, label_map = loader.dataset[image_id]
        input = input.to(device)
        label_map = label_map.to(device)
        label_map_unnorm, label_list = loader.dataset.get_label(image_id)

        # Forward Pass
        t_start = time.time()
        pred = net(input.unsqueeze(0)).squeeze_(0)
        print("Forward pass time", time.time() - t_start)

        # Select all the bounding boxes with classification score above threshold
        cls_pred = pred[..., 0]
        activation = cls_pred > threshold

        # Compute (x, y) of the corners of selected bounding box
        num_boxes = int(activation.sum())
        if num_boxes == 0:
            print("No bounding box found")
            return

        corners = torch.zeros((num_boxes, 8))
        for i in range(1, 9):
            corners[:, i - 1] = torch.masked_select(pred[..., i], activation)
        corners = corners.view(-1, 4, 2).numpy()

        scores = torch.masked_select(pred[..., 0], activation).numpy()

        # NMS
        t_start = time.time()
        selected_ids = non_max_suppression(corners, scores, config['nms_iou_threshold'])
        corners = corners[selected_ids]
        scores = scores[selected_ids]
        print("Non max suppression time:", time.time() - t_start)

        # Visualization
        input_np = input.cpu().numpy()
        plot_bev(input_np, label_list, window_name='GT')
        plot_bev(input_np, corners, window_name='Prediction')
        plot_label_map(cls_pred.numpy())

if __name__ == "__main__":

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    print('using device', device)

    name = 'config.json'
    #train(name, device)
    experiment(name, device)
