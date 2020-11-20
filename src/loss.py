# This one is deprecated but waiting to remove
def loss_batch(model, loss_func, x_batch, y_batch, opt=None):
    y_hat = model(**x_batch)
    loss = loss_func(y_hat, y_batch)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(next(iter(x_batch.values())))
