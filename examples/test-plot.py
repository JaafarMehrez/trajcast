from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

metrics = ["loss_training", "loss_validation"]
fig, ax = plt.subplots(1, 1)
for metric in metrics:
    event = event_accumulator.EventAccumulator(f"tb_log/{metric}")
    event.Reload()
    data = event.Scalars("loss")
    epochs = [i.step for i in data]
    loss = [i.value for i in data]
    ax.plot(epochs, loss, label=metric)

ax.set_ylabel("Loss")
ax.set_xlabel("Epoch")
ax.legend()
plt.show()
