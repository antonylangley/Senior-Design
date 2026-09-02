import type { Point } from './intent'

export interface BrickDimensions { studs_x: number; studs_y: number }
export interface GridPosition { row: number; column: number }
export interface RepresentativeColor { hsv: number[]; lab: number[] }
export interface RawPose {
  center_px: Point
  angle_degrees: number
  bounding_polygon: [number, number][]
}
export interface ModelFitDiagnostics {
  matched_studs: number
  expected_studs: number
  detected_candidates: number
  rejected_studs: number
  reprojection_error_px: number
  scale_px_per_mm: number
  predicted_stud_centers_px: [number, number][]
  matched_stud_centers_px: [number, number][]
  rejected_stud_centers_px: [number, number][]
}

export interface LegoBrick {
  id: number
  color: string
  dimensions: BrickDimensions | null
  stud_count: number
  stud_centers_px: [number, number][]
  center_px: Point
  center_normalized: Point
  angle_degrees: number
  rotational_symmetry_degrees: number
  pose_source: 'lego_model_fit' | 'contour_fallback'
  pose_confidence: number
  model_fit: ModelFitDiagnostics | null
  raw_pose: RawPose
  grid_position: GridPosition
  bounding_polygon: [number, number][]
  confidence: number
  dimension_confidence: number
  dimension_source: string
  representative_color: RepresentativeColor
}

export interface LegoDetectResponse {
  image: { width: number; height: number }
  processed_image_data: string
  grid: { rows: number; columns: number }
  bricks: LegoBrick[]
  debug: {
    rectified_view: string | null
    segmentation_mask: string
    components: string
    studs: string
    pose_refinement: string
  } | null
  rectification: { active: boolean; method: string }
  scene_scale: {
    scale_px_per_mm: number | null
    confidence: number
    relative_variation: number | null
    sample_count: number
    candidate_count: number
    outlier_count: number
    trustworthy: boolean
  }
  warning: string | null
}
