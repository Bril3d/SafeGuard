export interface ClassVisualStyle {
  stroke: string;
  fill: string;
  badgeBg: string;
  badgeText: string;
  label: string;
}

export const CLASS_STYLES: Record<string, ClassVisualStyle> = {
  person: {
    stroke: '#0369A1',
    fill: 'rgba(3, 105, 161, 0.10)',
    badgeBg: 'bg-sky-50 border-sky-300',
    badgeText: 'text-sky-800',
    label: 'Person',
  },
  helmet: {
    stroke: '#15803D',
    fill: 'rgba(21, 128, 61, 0.14)',
    badgeBg: 'bg-emerald-50 border-emerald-300',
    badgeText: 'text-emerald-800',
    label: 'Helmet',
  },
  vest: {
    stroke: '#0F766E',
    fill: 'rgba(15, 118, 110, 0.14)',
    badgeBg: 'bg-teal-50 border-teal-300',
    badgeText: 'text-teal-800',
    label: 'Safety vest',
  },
  gloves: {
    stroke: '#B45309',
    fill: 'rgba(180, 83, 9, 0.14)',
    badgeBg: 'bg-amber-50 border-amber-300',
    badgeText: 'text-amber-800',
    label: 'Gloves',
  },
  no_helmet: {
    stroke: '#B91C1C',
    fill: 'rgba(185, 28, 28, 0.16)',
    badgeBg: 'bg-red-50 border-red-300',
    badgeText: 'text-red-800',
    label: 'Missing helmet',
  },
};

export function getClassStyle(className: string): ClassVisualStyle {
  return (
    CLASS_STYLES[className] ?? {
      stroke: '#475569',
      fill: 'rgba(71, 85, 105, 0.12)',
      badgeBg: 'bg-slate-100 border-slate-300',
      badgeText: 'text-slate-800',
      label: className,
    }
  );
}
