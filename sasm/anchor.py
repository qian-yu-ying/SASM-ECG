"""Core affine-coordinate operations; no dataset or execution workflow.

Inputs must already have compatible shapes and clinically aligned concepts.
These functions do not infer mappings or resolve diagnostic aliases.
"""


def ecgfounder_raw_head(dense_weight, dense_bias, source_weight,
                       source_bias, source_mean, source_scale, concept_map):
    """Convert a standardized score-space head to raw pooled features.

    dense_weight: (score_dim, feature_dim)
    dense_bias, source_mean, source_scale: (score_dim,)
    source_weight: (score_dim, source_labels)
    source_bias: (source_labels,)
    concept_map: (source_labels, target_labels)

    The affine identity is specific to an interface where the source head
    consumes standardized linear scores, rather than sigmoid probabilities.
    """
    mapped_weight = source_weight @ concept_map
    mapped_bias = source_bias @ concept_map
    raw_weight = dense_weight.T @ (mapped_weight / source_scale[:, None])
    raw_bias = mapped_bias + ((dense_bias - source_mean) / source_scale) @ mapped_weight
    return raw_weight, raw_bias


def direct_feature_raw_head(mapped_weight, mapped_bias, source_mean, source_scale):
    """Convert an already concept-aligned direct-feature source head.

    mapped_weight: (feature_dim, target_labels)
    mapped_bias: (target_labels,)
    source_mean, source_scale: (feature_dim,)
    """
    raw_weight = mapped_weight / source_scale[:, None]
    raw_bias = mapped_bias - (source_mean / source_scale) @ mapped_weight
    return raw_weight, raw_bias


def target_coordinate_anchor(raw_weight, raw_bias, target_mean, target_scale):
    """Express the same affine scores in target-standardized coordinates.

    The returned bias preserves source scores. It is not a bias penalty
    or an instruction to fix the fitted target intercept.
    """
    anchor = target_scale[:, None] * raw_weight
    anchor_bias = target_mean @ raw_weight + raw_bias
    return anchor, anchor_bias
