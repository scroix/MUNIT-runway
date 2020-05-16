import runway
from runway.data_types import number, file, image
from trainer import MUNIT_Trainer
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image

a2b = 1

@runway.setup(options={'generator_checkpoint': runway.file(description="Checkpoint for the generator", extension='.pt')})
def setup(opts):
	generator_checkpoint_path = opts['generator_checkpoint']
	# generator_checkpoint_path = './checkpoints/ffhq2ladiescrop.pt'

	# Load experiment settings
	config = {'image_save_iter': 10000, 'image_display_iter': 100, 'display_size': 16, 'snapshot_save_iter': 10000, 'log_iter': 100, 'max_iter': 1000000, 'batch_size': 1, 'weight_decay': 0.0001, 'beta1': 0.5, 'beta2': 0.999, 'init': 'kaiming', 'lr': 0.0001, 'lr_policy': 'step', 'step_size': 100000, 'gamma': 0.5, 'gan_w': 1, 'recon_x_w': 10, 'recon_s_w': 1, 'recon_c_w': 1, 'recon_x_cyc_w': 10, 'vgg_w': 0, 'gen': {'dim': 64, 'mlp_dim': 256, 'style_dim': 8, 'activ': 'relu', 'n_downsample': 2, 'n_res': 4, 'pad_type': 'reflect'}, 'dis': {'dim': 64, 'norm': 'none', 'activ': 'lrelu', 'n_layer': 4, 'gan_type': 'lsgan', 'num_scales': 3, 'pad_type': 'reflect'}, 'input_dim_a': 3, 'input_dim_b': 3, 'num_workers': 8, 'new_size': 1024, 'crop_image_height': 400, 'crop_image_width': 400, 'data_root': './datasets/ffhq2ladies/'}

	# Setup model and data loader
	trainer = MUNIT_Trainer(config)

	state_dict = torch.load(generator_checkpoint_path)
	trainer.gen_a.load_state_dict(state_dict['a'])
	trainer.gen_b.load_state_dict(state_dict['b'])

	return {'model': trainer, 'config': config}

@runway.command(name='generate',
                inputs={ 'image': image(description='Input image'), 'style': number(default=1, min=0, max=1000, description='Style Seed') },
                outputs={ 'image': image(description='Output image') },
                description='Image translation with style seeding')
def generate(model, args):
	#start command here?
	trainer = model['model']
	config = model['config']
	style_dim = config['gen']['style_dim']

	image_in = args['image'].convert('RGB')

	# replace this
	num_style_start = args['style']
	torch.manual_seed(num_style_start)
	torch.cuda.manual_seed(num_style_start)
	new_size = config['new_size']

	trainer.cuda()
	trainer.eval()
	encode = trainer.gen_a.encode if a2b else trainer.gen_b.encode # encode function
	# style_encode = trainer.gen_b.encode if a2b else trainer.gen_a.encode # encode function
	decode = trainer.gen_b.decode if a2b else trainer.gen_a.decode # decode function

	with torch.no_grad():
	    transform = transforms.Compose([transforms.Resize(new_size),
	                                    transforms.ToTensor(),
	                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	    image = Variable(transform(image_in).unsqueeze(0).cuda())
	    #maybe for a future version?
	    # style_image = Variable(transform(Image.open(style).convert('RGB')).unsqueeze(0).cuda()) if opts.style != '' else None

	    # Start testing
	    content, _ = encode(image)

	    style_rand = Variable(torch.randn(num_style_start, style_dim, 1, 1).cuda())
	    style = style_rand

	    s = style[0].unsqueeze(0)
	    outputs = decode(content, s)
	    outputs = (outputs + 1) / 2.

	return {
        'image': transforms.ToPILImage()(outputs.cpu().squeeze_(0)).convert("RGB")
    }  

if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=8000, debug=True)  

