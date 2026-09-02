import type { LegoDetectResponse } from '../types/lego'

export function formatLegoSummary(result: LegoDetectResponse): string {
  const lines = ['Results', `Detected bricks: ${result.bricks.length}`]
  for (const brick of result.bricks) {
    const dimensions = brick.dimensions
      ? `${brick.dimensions.studs_x}x${brick.dimensions.studs_y}`
      : 'Unresolved'
    lines.push(
      '',
      `Brick #${brick.id}`,
      `Color: ${brick.color}`,
      `Dimensions: ${dimensions}`,
      `Stud count: ${brick.stud_count}`,
      `Center: (${brick.center_px.x.toFixed(1)}, ${brick.center_px.y.toFixed(1)})`,
      `Normalized: (${brick.center_normalized.x.toFixed(3)}, ${brick.center_normalized.y.toFixed(3)})`,
      `Yaw: ${brick.angle_degrees.toFixed(1)}°`,
      `Grid: [${brick.grid_position.row},${brick.grid_position.column}]`,
      `Confidence: ${Math.round(brick.confidence * 100)}%`,
      `Pose: ${brick.pose_source === 'lego_model_fit' ? 'Model fit' : 'Contour fallback'}`,
      `Pose confidence: ${Math.round(brick.pose_confidence * 100)}%`,
    )
  }
  return lines.join('\n')
}

export function serializeLegoResult(result: LegoDetectResponse): string {
  return JSON.stringify(result, null, 2)
}
