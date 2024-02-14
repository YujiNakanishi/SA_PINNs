import numpy as np
import torch
import model
import pandas as pd

x = 2.*torch.pi*0.01*torch.arange(100).view(-1,1).to("cuda")
y = torch.sin(x)

net = model.Net_sin(10).to("cuda")
net.load_state_dict(torch.load("best_params.pth"))

pred = net(x).cpu().detach().numpy()[:,0]
pred = pd.Series(pred)
pred.to_csv("pred.csv")