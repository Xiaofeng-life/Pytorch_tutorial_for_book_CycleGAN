import torch
import models

device = torch.device("cpu")
generator_x2y = models.Generator(in_ch=3, out_ch=3, ngf=64, num_res=6).to(device)
generator_y2x = models.Generator(in_ch=3, out_ch=3, ngf=64, num_res=6).to(device)
generator_x2y.load_state_dict(torch.load("results/pth/60_x2y.pth"))
generator_y2x.load_state_dict(torch.load("results/pth/60_y2x.pth"))

generator_x2y.eval()
generator_y2x.eval()

x = torch.rand(1, 3, 256, 256)
traced_script_module = torch.jit.trace(func=generator_x2y, example_inputs=x)
traced_script_module.save("60_x2y.pt")

traced_script_module = torch.jit.trace(func=generator_y2x, example_inputs=x)
traced_script_module.save("60_y2x.pt")
