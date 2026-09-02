import type { LegoDetectResponse } from '../types/lego'

interface LegoOverlayProps {
  imageData: string
  result: LegoDetectResponse
}

export function LegoOverlay({ imageData, result }: LegoOverlayProps) {
  const { width, height } = result.image
  return (
    <div className="lego-image-stage">
      <img src={imageData} alt="Uploaded LEGO bricks with detection overlay" />
      <svg viewBox={`0 0 ${width} ${height}`} aria-label="LEGO detection overlay">
        {Array.from({ length: result.grid.columns - 1 }, (_, index) => (
          <line key={`column-${index}`} className="grid-line" x1={(index + 1) * width / result.grid.columns} x2={(index + 1) * width / result.grid.columns} y1="0" y2={height} />
        ))}
        {Array.from({ length: result.grid.rows - 1 }, (_, index) => (
          <line key={`row-${index}`} className="grid-line" x1="0" x2={width} y1={(index + 1) * height / result.grid.rows} y2={(index + 1) * height / result.grid.rows} />
        ))}
        {result.bricks.map((brick) => {
          const dimensions = brick.dimensions ? `${brick.dimensions.studs_x}x${brick.dimensions.studs_y}` : 'size ?'
          return (
            <g key={brick.id}>
              <polygon className="brick-polygon" points={brick.bounding_polygon.map(([x, y]) => `${x},${y}`).join(' ')} />
              <circle className="brick-center" cx={brick.center_px.x} cy={brick.center_px.y} r="6" />
              {brick.stud_centers_px.map(([x, y], index) => <circle key={index} className="stud-marker" cx={x} cy={y} r="5" />)}
              <text className="brick-label" x={brick.center_px.x + 9} y={brick.center_px.y - 23}>
                <tspan x={brick.center_px.x + 9}>#{brick.id} {brick.color} {dimensions}</tspan>
                <tspan x={brick.center_px.x + 9} dy="18">{brick.stud_count} studs · {brick.angle_degrees.toFixed(1)}°</tspan>
                <tspan x={brick.center_px.x + 9} dy="18">Grid [{brick.grid_position.row},{brick.grid_position.column}]</tspan>
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
