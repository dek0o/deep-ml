import torch

def train_step(model, x, y, optimizer, loss_fn):
    """Run one training step and return the pre-update loss as a float.

    Args:
        model: torch.nn.Module to train.
        x: Input batch tensor.
        y: Target batch tensor.
        optimizer: torch.optim optimizer bound to model parameters.
        loss_fn: Callable (pred, y) -> scalar loss tensor.

    Returns:
        float: Loss value computed before optimizer.step().
    """
    # TODO: zero_grad -> forward -> loss -> backward -> step; return float loss
    optimizer.zero_grad() # Обнуляем градиенты, чтобы они не влияли на обновление весов после подсчета лосса
    outputs = model(x)
    loss = loss_fn(outputs, y) # Загружаем в критеон для подсчета

    loss_val = loss.item() # Сохраняем

    loss.backward() # Считаем градиенты
    optimizer.step() # Обновляем веса

    return loss_val
