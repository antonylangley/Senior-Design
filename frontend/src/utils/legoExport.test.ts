import { describe, expect, it } from 'vitest'
import type { LegoDetectResponse } from '../types/lego'
import { formatLegoSummary, serializeLegoResult } from './legoExport'

const result: LegoDetectResponse = {
  image: { width: 1000, height: 800 },
  processed_image_data: 'data:image/png;base64,test',
  grid: { rows: 10, columns: 10 },
  rectification: { active: false, method: 'none' },
  scene_scale: { scale_px_per_mm: null, confidence: 0, relative_variation: null, sample_count: 0, candidate_count: 0, outlier_count: 0, trustworthy: false },
  debug: null,
  warning: null,
  bricks: [{
    id: 1, color: 'red', dimensions: { studs_x: 2, studs_y: 3 }, stud_count: 6,
    stud_centers_px: [], center_px: { x: 203.7, y: 600.6 },
    center_normalized: { x: 0.141, y: 0.553 }, angle_degrees: 155.3,
    rotational_symmetry_degrees: 180, pose_source: 'lego_model_fit', pose_confidence: 0.91,
    model_fit: null, raw_pose: { center_px: { x: 204, y: 601 }, angle_degrees: 154, bounding_polygon: [] },
    grid_position: { row: 5, column: 1 }, bounding_polygon: [], confidence: 0.73,
    dimension_confidence: 0.8, dimension_source: 'stud_lattice+aspect_ratio',
    representative_color: { hsv: [0, 0, 0], lab: [0, 0, 0] },
  }],
}

describe('LEGO result exports', () => {
  it('formats a readable summary', () => {
    expect(formatLegoSummary(result)).toBe(`Results
Detected bricks: 1

Brick #1
Color: red
Dimensions: 2x3
Stud count: 6
Center: (203.7, 600.6)
Normalized: (0.141, 0.553)
Yaw: 155.3°
Grid: [5,1]
Confidence: 73%
Pose: Model fit
Pose confidence: 91%`)
  })

  it('serializes the full response as formatted JSON', () => {
    expect(JSON.parse(serializeLegoResult(result))).toEqual(result)
    expect(serializeLegoResult(result)).toContain('\n  "image"')
  })
})
