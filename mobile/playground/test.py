from abc import ABC

import numpy as np
import torch as T

device = T.device("cpu")  # apply to Tensor or Module


# -----------------------------------------------------------




# -----------------------------------------------------------

def main():
    # 0. get started


    # 1. set up training data










    # 2. create network
      # could use Sequential()

    # 3. train model
    max_epochs = 100
    lrn_rate = 0.04
    loss_func = T.nn.CrossEntropyLoss()  # applies softmax()
    optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)

    print("\nStarting training ")
    net.train()
    indices = np.arange(6)
    for epoch in range(0, max_epochs):
        np.random.shuffle(indices)
        for i in indices:
            X = train_x[i].reshape(1, 4)  # device inherited
            Y = train_y[i].reshape(1, )
            optimizer.zero_grad()
            oupt = net(X)
            loss_obj = loss_func(oupt, Y)
            loss_obj.backward()
            optimizer.step()
        # (monitor error)
    print("Done training ")

    # 4. (evaluate model accuracy)

    # 5. use model to make a prediction
    net.eval()
    print("\nPredicting species for [5.8, 2.8, 4.5, 1.3]: ")
    unk = np.array([[5.8, 2.8, 4.5, 1.3]], dtype=np.float32)
    unk = T.tensor(unk, dtype=T.float32).to(device)
    logits = net(unk).to(device)
    probs = T.softmax(logits, dim=1)
    probs = probs.detach().numpy()  # allows printoptions

    np.set_printoptions(precision=4)
    print(probs)

    # 6. (save model)

    print("\nEnd Iris demo")


if __name__ == "__main__":
    main()
