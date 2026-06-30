"use client";

import { Badge } from "@/components/ui/badge";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function SHAPBeeswarm() {
  const imgSrc = `${API_BASE}/static/shap_beeswarm.png`;
  const today = new Date().toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });

  return (
    <div>
      <div className="flex items-center gap-3 mb-3">
        <h3 className="text-base font-semibold">SHAP Feature Importance</h3>
        <Badge variant="secondary">Generated: {today}</Badge>
      </div>
      <div className="rounded-lg border border-border bg-card p-2 overflow-hidden">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={imgSrc}
          alt="SHAP Beeswarm Plot"
          className="w-full h-auto rounded"
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
            (e.target as HTMLImageElement).parentElement!.innerHTML =
              '<p class="text-muted-foreground text-sm p-4">SHAP plot not yet generated. Run run_daily.py to compute.</p>';
          }}
        />
      </div>
      <p className="text-[10px] text-muted-foreground mt-1">
        Generated server-side · Updated daily
      </p>
    </div>
  );
}
