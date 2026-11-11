
import torch 

def truncate_spatial_rank(features, rank):
    """
    features: [B, N, C]
    N = spatial tokens
    C = feature dimension
    """
    B, N, C = features.shape
    output = []
    
    for b in range(B):
        X = features[b].float()
        # Center across spatial locations
        X_mean = X.mean(dim=0, keepdim=True)
        X_centered = X - X_mean
        # SVD
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        r = min(rank, S.shape[0])
        X_r = (U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :])
        # Restore mean
        X_r = X_r + X_mean
        output.append(X_r)

    return torch.stack(output)


# Rank truncation backbone (we're not directly changing SegmentationProbing)
class RankTruncatedBackbone(torch.nn.Module):
    def __init__(self, backbone, rank):
        super().__init__()
        self.backbone = backbone
        self.rank = rank

    def encode(self, inputs):
        outputs = self.backbone.encode(inputs)
        dense = outputs["dense"]
        dense_rank = truncate_spatial_rank(dense, self.rank)
        outputs = dict(outputs)
        outputs["dense"] = dense_rank
        return outputs
