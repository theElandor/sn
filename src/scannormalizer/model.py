import torch
import torch.nn as nn


ROTATION_CLASS_MATRICES = torch.diag_embed(
    torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=torch.float32,
    )
)


class RotationNormalizer(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.register_buffer(
            "flip_matrices",
            ROTATION_CLASS_MATRICES.clone(),
            persistent=False,
        )

        #self.encoder = nn.Sequential(
        #    nn.Conv1d(3, 64, 1),
        #    nn.BatchNorm1d(64),
        #    nn.ReLU(inplace=True),
        #    nn.Conv1d(64, 128, 1),
        #    nn.BatchNorm1d(128),
        #    nn.ReLU(inplace=True),
        #    nn.Conv1d(128, 256, 1),
        #    nn.BatchNorm1d(256),
        #    nn.ReLU(inplace=True),
        #    nn.Conv1d(256, 512, 1),
        #    nn.BatchNorm1d(512),
        #    nn.ReLU(inplace=True),
        #)
        #self.head = nn.Sequential(
        #    nn.Linear(512, 512),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(512, 256),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(256, 128),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(128, 4),
        #)

        # Previous larger model. Uncomment this encoder/head pair to load old checkpoints.
        # self.encoder = nn.Sequential(
        #     nn.Conv1d(3, 256, 1),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(256, 512, 1),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(512, 1024, 1),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(1024, 2048, 1),
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(inplace=True),
        # )
        # self.head = nn.Sequential(
        #     nn.Linear(2048, 2048),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 4),
        # )
        #
        #  1,5M parameters
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Conv1d(1024, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
        )

        # 2048 max-pooled features + 2048 average-pooled features = 4096
        self.head = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 4),
        )
    def forward(self, points):
        """
        points: [B, N, 3]
        """

        features = self.encoder(points.transpose(1, 2))
        # features: [B, 2048, N]

        global_max = features.max(dim=-1).values
        global_avg = features.mean(dim=-1)

        global_feature = torch.cat([global_max, global_avg], dim=-1)
        # global_feature: [B, 4096]

        logits = self.head(global_feature)

        flip_matrices = self.flip_matrices.to(
            device=points.device,
            dtype=points.dtype,
        )

        predicted_rotation = flip_matrices[logits.argmax(dim=-1)]

        return {
            "logits": logits,
            "predicted_rotation": predicted_rotation,
        }
