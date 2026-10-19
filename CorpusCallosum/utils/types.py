from typing import Literal, TypedDict

from numpy import dtype, float64, ndarray

from FastSurferCNN.utils import ScalarType

__all__ = [
    "CCMeasuresDict",
    "ContourList",
    "ContourThickness",
    "Points2dType",
    "Points3dType",
    "Polygon2dType",
    "Polygon3dType",
    "SliceSelection",
    "SubdivisionMethod",
]

Polygon2dType = ndarray[tuple[Literal[2], int], dtype[ScalarType]]
Polygon3dType = ndarray[tuple[Literal[3], int], dtype[ScalarType]]
Points2dType = ndarray[tuple[int, Literal[2]], dtype[ScalarType]]
Points3dType = ndarray[tuple[int, Literal[3]], dtype[ScalarType]]
ContourList = list[type[Polygon2dType]]
ContourThickness = ndarray[tuple[Literal[3], int], dtype[ScalarType]]
SliceSelection = Literal["middle", "all"] | int
SubdivisionMethod = Literal["shape", "vertical", "angular", "eigenvector"]

class CCMeasuresDict(TypedDict):
    """TypedDict for corpus callosum measures.

    Attributes
    ----------
    cc_index : float
        Dimensionless corpus callosum index: the sum of anterior, middle, and
        posterior widths divided by the anterior-posterior length.
    circularity : float
        Dimensionless circularity, calculated as 4 pi times area divided by
        squared perimeter.
    areas : np.ndarray
        Areas of subdivided regions.
    midline_length : float
        Length along the midline.
    thickness : float
        Mean thickness across all level paths.
    curvature : float
        Mean absolute turning angle of the midline in degrees per midline sample step (depends on num_thickness_points).
    thickness_profile : np.ndarray of type float
        Thickness measurements (level path lengths) from anterior to posterior along the midline.
    total_area : float
        Total area of the CC.
    total_perimeter : float
        Total perimeter length.
    split_contours : list of np.ndarray
        Subdivided contour segments in AS-slice coordinates.
    midline_equidistant : np.ndarray
        Equidistant points along midline in AS-slice coordinates.
    curvature_subsegments : np.ndarray
        Curvature (same unit as curvature) of the midline within each subsegment.
    curvature_body : float
        Curvature (same unit as curvature) of the midline within the body of the CC.
    levelpaths : list of np.ndarray
        Paths for thickness measurements in AS-slice coordinates.
    subdivision_lines : list of np.ndarray
        Lines dividing the subsegments in AS-slice coordinates, each given by its two endpoints on the contour.
    slice_index : int
        Index of the processed slice.
    """
    cc_index: float
    circularity: float
    areas: ndarray
    midline_length: float
    thickness: float
    curvature: float
    thickness_profile: ndarray[tuple[int], dtype[float64]]
    total_area: float
    total_perimeter: float
    split_contours: ContourList
    midline_equidistant: ndarray
    curvature_subsegments: ndarray
    curvature_body: float
    levelpaths: list[ndarray]
    subdivision_lines: list[Points2dType]
    slice_index: int
