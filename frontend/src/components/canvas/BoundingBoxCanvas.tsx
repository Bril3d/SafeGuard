import { useEffect, useRef } from 'react';
import { getClassStyle } from '../../lib/ppeColors';
import type { DetectionSchema, PersonAssociationSchema } from '../../types/api';

interface BoundingBoxCanvasProps {
  imageSrc: string | null;
  imageWidth: number;
  imageHeight: number;
  detections: DetectionSchema[];
  associations?: PersonAssociationSchema[];
  showPersonBoxes?: boolean;
  alt?: string;
}

export function BoundingBoxCanvas({
  imageSrc,
  imageWidth,
  imageHeight,
  detections,
  associations,
  showPersonBoxes = true,
  alt = 'Inspection preview',
}: BoundingBoxCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || imageWidth <= 0 || imageHeight <= 0) {
      return;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }

    canvas.width = imageWidth;
    canvas.height = imageHeight;
    ctx.clearRect(0, 0, imageWidth, imageHeight);

    const visibleDetections = showPersonBoxes
      ? detections
      : detections.filter((d) => d.class_name !== 'person');

    const lineW = Math.max(2, Math.round(Math.min(imageWidth, imageHeight) / 280));
    const fontSize = Math.max(12, Math.round(Math.min(imageWidth, imageHeight) / 42));
    ctx.font = `600 ${fontSize}px Inter, system-ui, sans-serif`;

    for (const det of visibleDetections) {
      const style = getClassStyle(det.class_name);
      const [x1, y1, x2, y2] = det.bbox_xyxy;
      const bw = Math.max(1, x2 - x1);
      const bh = Math.max(1, y2 - y1);

      ctx.fillStyle = style.fill;
      ctx.fillRect(x1, y1, bw, bh);

      ctx.strokeStyle = style.stroke;
      ctx.lineWidth = lineW;
      ctx.strokeRect(x1, y1, bw, bh);

      const label = `${style.label} ${(det.confidence * 100).toFixed(0)}%`;
      const textWidth = ctx.measureText(label).width;
      const padX = 5;
      const pillH = fontSize + 6;
      const pillY = y1 - pillH >= 0 ? y1 - pillH : y1;

      ctx.fillStyle = style.stroke;
      ctx.fillRect(x1, pillY, Math.min(imageWidth - x1, textWidth + padX * 2), pillH);
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(label, x1 + padX, pillY + fontSize + 1);
    }

    if (associations && associations.length > 0) {
      for (const assoc of associations) {
        const [px1, py1, px2] = assoc.bbox_xyxy;
        const statusColor =
          assoc.overall_status === 'potential_violation'
            ? '#B91C1C'
            : assoc.overall_status === 'pending_confirmation'
              ? '#B45309'
              : assoc.overall_status === 'compliant'
                ? '#15803D'
                : '#475569';
        const tag = `#${assoc.track_id} ${assoc.overall_status.replaceAll('_', ' ')}`;
        const tw = ctx.measureText(tag).width;
        const pillH = fontSize + 6;
        const tx = Math.max(0, Math.min(imageWidth - tw - 10, px2 - tw - 10));
        const ty = Math.min(imageHeight - pillH, py1 + 2);
        if (px2 - px1 > 30) {
          ctx.fillStyle = statusColor;
          ctx.fillRect(tx, ty, tw + 10, pillH);
          ctx.fillStyle = '#FFFFFF';
          ctx.fillText(tag, tx + 5, ty + fontSize + 1);
        }
      }
    }
  }, [detections, associations, imageHeight, imageWidth, showPersonBoxes]);

  if (!imageSrc) {
    return null;
  }

  return (
    <div className="relative w-full overflow-hidden rounded border border-[#D1D5DB] bg-[#0F172A]">
      <img
        src={imageSrc}
        alt={alt}
        className="block h-auto max-h-[540px] w-full object-contain select-none"
      />
      <canvas
        ref={canvasRef}
        className="pointer-events-none absolute inset-0 h-full w-full object-contain"
        aria-label="Bounding box detection overlay"
      />
    </div>
  );
}
