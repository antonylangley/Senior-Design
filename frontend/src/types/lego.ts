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
  representative_color: RepresentativeColor
}

export interface LegoDetectResponse {
  image: { width: number; height: number }
  grid: { rows: number; columns: number }
  bricks: LegoBrick[]
  debug: {
    segmentation_mask: string
    components: string
    studs: string
  } | null
  warning: string | null
}
