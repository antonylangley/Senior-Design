import type { Point } from './intent'

export interface BrickDimensions { studs_x: number; studs_y: number }
export interface GridPosition { row: number; column: number }
export interface RepresentativeColor { hsv: number[]; lab: number[] }

export interface LegoBrick {
  id: number
  color: string
  dimensions: BrickDimensions | null
  stud_count: number
  stud_centers_px: [number, number][]
  center_px: Point
  center_normalized: Point
  angle_degrees: number
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
  } | null
  rectification: { active: boolean; method: string }
  warning: string | null
}
