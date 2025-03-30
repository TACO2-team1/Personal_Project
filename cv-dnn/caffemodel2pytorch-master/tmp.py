import torch
import caffemodel2pytorch

model = caffemodel2pytorch.Net(
	prototxt=r'C:\Users\KO SEONGHUN\Desktop\deepfake_curry\Code\cv-dnn\caffemodel2pytorch-master\resnet_ssd_v1.prototxt',
	weights=r'C:\Users\KO SEONGHUN\Desktop\deepfake_curry\Code\cv-dnn\caffemodel2pytorch-master\resnet_ssd_v1.caffemodel',
	# caffe_proto='https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto'
    caffe_proto='https://github.com/HardenCurry/xincunjian/blob/main/caffe.proto'
    
)
model.cuda()
model.eval()
torch.set_grad_enabled(False)

# make sure to have right procedure of image normalization and channel reordering
image = torch.Tensor(8, 3, 224, 224).cuda()

# outputs dict of PyTorch Variables
# in this example the dict contains the only key "prob"
# output_dict = model(data = image)

# you can remove unneeded layers:
# del model.prob
# del model.fc8

# a single input variable is interpreted as an input blob named "data"
# in this example the dict contains the only key "fc7"
output_dict = model(image)
