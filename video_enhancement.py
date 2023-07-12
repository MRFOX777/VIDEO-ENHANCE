import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19
from torchvision.models.vgg import model_urls
from torch import nn
from torch.autograd import Variable

# Define the super-resolution model
class SuperResolutionNet(nn.Module):
    def __init__(self):
        super(SuperResolutionNet, self).__init__()
        model_urls['vgg19'] = model_urls['vgg19'].replace('https://', 'http://')
        self.vgg = vgg19(pretrained=True).features
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(out1))
        out3 = self.conv3(out2)
        return out3

# Load the pre-trained model
model = SuperResolutionNet()
model.load_state_dict(torch.load('super_resolution_model.pth'))
model.eval()

# Define the video enhancement function
def enhance_video(input_path, output_path):
    # Open the video file
    video_capture = cv2.VideoCapture(input_path)
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Create an output video writer
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Process each frame in the video
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert the frame to tensor
        img = transforms.ToTensor()(frame).unsqueeze(0)

        # Perform super-resolution
        img = Variable(img, volatile=True)
        output = model(img)
        output = output.data.squeeze(0).clamp(0, 1)

        # Convert the tensor back to an OpenCV image
        output = transforms.ToPILImage()(output)
        output = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)

        # Write the enhanced frame to the output video
        video_writer.write(output)

    # Release the video capture and writer
    video_capture.release()
    video_writer.release()

    print("Video enhancement complete!")

# Call the video enhancement function
enhance_video('input_video.mp4', 'output_video.mp4')
