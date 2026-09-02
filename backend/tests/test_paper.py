import numpy as np

from app.vision.paper import order_corners


def test_order_corners_returns_tl_tr_br_bl() -> None:
    points = np.array(
        [
            [620, 460],
            [120, 90],
            [650, 120],
            [80, 500],
        ],
        dtype="float32",
    )

    ordered = order_corners(points)

    assert ordered.tolist() == [
        [120.0, 90.0],
        [650.0, 120.0],
        [620.0, 460.0],
        [80.0, 500.0],
    ]


def test_order_corners_accepts_nested_contour_shape() -> None:
    contour = np.array([[[300, 300]], [[10, 20]], [[320, 30]], [[20, 310]]], dtype="float32")

    ordered = order_corners(contour)

    assert ordered.shape == (4, 2)
    assert ordered[0].tolist() == [10.0, 20.0]
    assert ordered[2].tolist() == [300.0, 300.0]
