"""The faces metric's contact sheet keeps one column per variant."""

import numpy as np
import pytest

from kinovsr.eval import face_yunet_metrics as faces

pytestmark = pytest.mark.unit


def test_a_short_variant_leaves_a_blank_cell_not_a_shifted_row(tmp_path, monkeypatch):
    # A variant that decoded fewer frames used to be skipped, so every later
    # panel moved one cell left under the wrong variant's column.
    labels: list[str] = []

    def label_panel(rgb, label):
        labels.append(label)
        return np.full((286, 256, 3), len(labels), dtype=np.uint8)

    written = {}
    monkeypatch.setattr(faces, "_label_panel", label_panel)
    monkeypatch.setattr(faces.cv, "imwrite", lambda path, image: written.setdefault("image", image))

    frame = np.zeros((64, 64, 3), dtype=np.float32)
    frames = {"base": [frame] * 10, "short": [frame] * 4, "other": [frame] * 10}
    observations = [
        faces.FaceObs(frame=f, track=track, box=(8, 8, 40, 40), score=0.9)
        for track, span in ((1, range(3)), (2, range(6, 9)))
        for f in span
    ]
    faces.make_contact_sheet(
        tmp_path / "sheet.png", frames, observations, ["base", "short", "other"]
    )

    assert [label.split()[-1] for label in labels] == [
        "base",
        "short",
        "other",
        "base",
        "missing",
        "other",
    ]
    image = written["image"][..., ::-1]  # cv.imwrite receives BGR
    cells = [int(image[row * 286, col * 256, 0]) for row in range(2) for col in range(3)]
    assert cells == [1, 2, 3, 4, 5, 6]
