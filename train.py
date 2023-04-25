import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as torchdata
from torch.autograd import Variable

from transformers import BertTokenizer, BertForPreTraining
from datasets import load_dataset, DatasetDict, Dataset

import flor
from flor import MTK as Flor

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
num_epochs = 5
batch_size = 6
learning_rate = 0.001
max_length = 480

# Data loader
data = load_dataset("wikipedia", "20220301.en")
assert isinstance(data, DatasetDict)
assert set(data.keys()) == {
    "train",
}  # type: ignore
assert isinstance(data["train"], Dataset)
assert set(data["train"].features) == {"id", "url", "title", "text"}

feature_extractor = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForPreTraining.from_pretrained("bert-base-uncased").to(device)  # type: ignore
Flor.checkpoints(model)


def my_collate(batch):
    original_text = []
    for i, record in enumerate(batch):
        original_text.append(record["text"])
    new_features = feature_extractor(
        original_text,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )

    # torchdata.default_collate(new_features)
    return new_features


train_loader = torchdata.DataLoader(dataset=data["train"].with_format("torch"), batch_size=batch_size, shuffle=True, collate_fn=my_collate)  # type: ignore

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
Flor.checkpoints(optimizer)

# Train the model
total_step = len(train_loader)
num_steps = 1000
for epoch in Flor.loop(range(num_epochs)):
    model.train()
    for i, batch in Flor.loop(enumerate(train_loader)):
        # Move tensors to the configured device
        # text = feature_extractor.decode(each) for each in batch["input_ids"]
        batch = batch.to(device)

        # Forward pass
        outputs = model(**batch)
        loss = criterion(
            outputs.prediction_logits.reshape((batch_size, -1, max_length)),
            batch["input_ids"],
        )
        # loss = Variable(
        #     criterion(
        #         outputs.prediction_logits.argmax(2).float(),
        #         batch["input_ids"].float(),
        #     ),
        #     requires_grad=True,
        # )

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, num_epochs, i, num_steps, flor.log("loss", loss.item())
                )
            )
            if i == num_steps:
                # bootleg sampling
                break

    print("Model Validate")

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
print("Model TEST")
