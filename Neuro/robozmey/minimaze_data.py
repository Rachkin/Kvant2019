import scipy.optimize
import numpy as np
import torch

model = HarukaNet()
model.load_state_dict(torch.load('my_models/HN.pt'))
model.eval()

loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

x0 = torch.zeros([1,7])

for i in range(7):
    
    x0 = torch.zeros([1,7])
    x0[0][i] = 1
    
    result = scipy.optimize.minimize(loss_value, x0, y = x0)
    
    import numpy as np
    plt.imshow(result[0,0,:,:])
    plt.show()