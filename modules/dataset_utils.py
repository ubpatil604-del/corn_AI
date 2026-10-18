import hashlib
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from modules.vision import SUPPORTED_EXTENSIONS


DISEASE_CLASS_ALIASES = {
    "healthy": {"healthy", "healthy corn"},
    "common rust": {"common rust", "rust", "rust disease"},
    "gray leaf spot": {"gray leaf spot", "gray leaf disease"},
    "northern leaf blight": {"northern leaf blight", "northern corn leaf blight", "nlb"},
}

_AUGMENTATION_SUFFIX = re.compile(
    r"(?:[_\s-]+\d+deg|[_\s-]+new\d+degflip(?:lr|ud)?|"
    r"[_\s-]+flip(?:lr|ud)|[\s_-]+copy(?:\s*\(\d+\))?)$",
    re.IGNORECASE,
)


def canonical_disease_name(folder_name):
    normalized = " ".join(folder_name.casefold().replace("_", " ").replace("-", " ").split())
    for canonical, aliases in DISEASE_CLASS_ALIASES.items():
        if normalized in aliases:
            return canonical
    return None


def list_images(folder):
    return sorted(
        path for path in Path(folder).rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _normalized_stem(path):
    stem = path.stem.casefold().strip()
    while True:
        reduced = _AUGMENTATION_SUFFIX.sub("", stem)
        if reduced == stem:
            return stem
        stem = reduced


def _perceptual_hash(path):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode dataset image: {path}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)[:8, :8]
    values = dct.flatten()
    median = np.median(values[1:])
    result = 0
    for bit, value in enumerate(values):
        result |= int(value > median) << bit
    return result


def source_groups(paths, max_hash_distance=4):
    paths = [Path(path) for path in paths]
    parent = list(range(len(paths)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    by_name = defaultdict(list)
    hashes = []
    by_digest = {}
    for index, path in enumerate(paths):
        by_name[_normalized_stem(path)].append(index)
        digest = hashlib.sha256(path.read_bytes()).digest()
        if digest in by_digest:
            union(index, by_digest[digest])
        else:
            by_digest[digest] = index
        hashes.append(_perceptual_hash(path))

    for indices in by_name.values():
        for index in indices[1:]:
            union(indices[0], index)

    buckets = defaultdict(set)
    for index, value in enumerate(hashes):
        for chunk in range(8):
            buckets[(chunk, (value >> (chunk * 8)) & 0xFF)].add(index)

    candidate_pairs = set()
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        ordered = sorted(indices)
        for position, first in enumerate(ordered):
            for second in ordered[position + 1:]:
                candidate_pairs.add((first, second))

    for first, second in candidate_pairs:
        if (hashes[first] ^ hashes[second]).bit_count() <= max_hash_distance:
            union(first, second)

    roots = {}
    return [
        roots.setdefault(find(index), len(roots))
        for index in range(len(paths))
    ]


def stratified_group_split(labels, groups, seed):
    labels = np.asarray(labels, dtype=np.int32)
    groups = np.asarray(groups, dtype=np.int64)
    class_group_counts = [
        len(set(groups[labels == label]))
        for label in np.unique(labels)
    ]
    smallest_class = min(class_group_counts)
    if smallest_class < 3:
        raise ValueError(
            "Each class needs at least 3 distinct source groups for train/validation/test; "
            f"found per-class group counts {class_group_counts}."
        )

    test_folds = min(7, smallest_class)
    first_split = StratifiedGroupKFold(
        n_splits=test_folds, shuffle=True, random_state=seed
    )
    train_validation, test = next(first_split.split(np.zeros(len(labels)), labels, groups))

    remaining_labels = labels[train_validation]
    remaining_groups = groups[train_validation]
    validation_folds = min(6, min(
        len(set(remaining_groups[remaining_labels == label]))
        for label in np.unique(remaining_labels)
    ))
    second_split = StratifiedGroupKFold(
        n_splits=validation_folds, shuffle=True, random_state=seed + 1
    )
    train_local, validation_local = next(
        second_split.split(np.zeros(len(remaining_labels)), remaining_labels, remaining_groups)
    )
    train = train_validation[train_local]
    validation = train_validation[validation_local]

    train_groups = set(groups[train])
    validation_groups = set(groups[validation])
    test_groups_set = set(groups[test])
    if (
        train_groups & validation_groups
        or train_groups & test_groups_set
        or validation_groups & test_groups_set
    ):
        raise RuntimeError("Source-image leakage detected across dataset partitions.")
    return train, validation, test


def verify_group_labels(labels, groups, paths):
    group_labels = defaultdict(set)
    group_paths = defaultdict(list)
    for label, group, path in zip(labels, groups, paths):
        group_labels[int(group)].add(int(label))
        group_paths[int(group)].append(str(path))
    conflicts = [
        group_paths[group]
        for group, class_labels in group_labels.items()
        if len(class_labels) > 1
    ]
    if conflicts:
        examples = [path for paths_in_group in conflicts[:3] for path in paths_in_group[:3]]
        raise ValueError(
            "Near-duplicate/source groups have conflicting class labels. Review these "
            f"images before training: {examples}"
        )


def verify_external_sources_are_distinct(training_paths, external_test_paths):
    if not external_test_paths:
        return
    all_paths = [*training_paths, *external_test_paths]
    combined_groups = source_groups(all_paths)
    training_groups = set(combined_groups[:len(training_paths)])
    test_groups = set(combined_groups[len(training_paths):])
    overlapping_groups = training_groups & test_groups
    if overlapping_groups:
        examples = []
        for group in sorted(overlapping_groups):
            train_examples = [
                str(training_paths[index])
                for index, group_id in enumerate(combined_groups[:len(training_paths)])
                if group_id == group
            ]
            test_examples = [
                str(external_test_paths[index])
                for index, group_id in enumerate(combined_groups[len(training_paths):])
                if group_id == group
            ]
            examples.append((train_examples[:2], test_examples[:2]))
            if len(examples) == 5:
                break
        raise ValueError(
            "Training images overlap the independent test set by normalized source name, "
            "exact hash, or perceptual near-duplicate. Remove duplicate sources before "
            "training; test images are not used for model fitting. Example path pairs "
            f"(train, test): {examples}"
        )
