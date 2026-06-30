/**
 * Test 1: HealthDot shows "error" state (red dot) when last_run_status === "error"
 */
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";

const mockStatusData = { status: undefined as string | undefined, last_run_status: undefined as string | undefined, last_run_at: undefined as string | undefined };

jest.mock("@/lib/api", () => ({
  useStatus: () => ({ data: mockStatusData }),
  usePositions: () => ({ data: null }),
}));

jest.mock("next/navigation", () => ({
  usePathname: () => "/",
}));

jest.mock("next/link", () => {
  return function MockLink({ children, ...props }: any) {
    return <a {...props}>{children}</a>;
  };
});

import { Sidebar } from "@/components/sidebar";

describe("HealthDot", () => {
  it("shows red dot when last_run_status is 'error'", () => {
    mockStatusData.status = undefined;
    mockStatusData.last_run_status = "error";
    mockStatusData.last_run_at = new Date().toISOString();

    const { container } = render(<Sidebar />);

    const dots = container.querySelectorAll(".bg-red-500");
    expect(dots.length).toBeGreaterThan(0);
  });

  it("shows green dot when status is ok and recent", () => {
    mockStatusData.status = undefined;
    mockStatusData.last_run_status = "ok";
    mockStatusData.last_run_at = new Date().toISOString();

    const { container } = render(<Sidebar />);

    const dots = container.querySelectorAll(".bg-emerald-500");
    expect(dots.length).toBeGreaterThan(0);
  });

  it("shows yellow dot when last_run_at is stale (>26h)", () => {
    mockStatusData.status = undefined;
    mockStatusData.last_run_status = "ok";
    mockStatusData.last_run_at = new Date(Date.now() - 30 * 60 * 60 * 1000).toISOString();

    const { container } = render(<Sidebar />);

    const dots = container.querySelectorAll(".bg-yellow-500");
    expect(dots.length).toBeGreaterThan(0);
  });

  it("shows neutral dot when not configured", () => {
    mockStatusData.status = "not_configured";
    mockStatusData.last_run_status = undefined;
    mockStatusData.last_run_at = undefined;

    const { container } = render(<Sidebar />);

    const dots = container.querySelectorAll(".bg-neutral-500");
    expect(dots.length).toBeGreaterThan(0);
  });
});
