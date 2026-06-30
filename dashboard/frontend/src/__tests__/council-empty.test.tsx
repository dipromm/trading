/**
 * Test 2: CouncilVerdictTable shows empty state when paper trading hasn't started
 */
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";

jest.mock("@/lib/api", () => ({
  useCouncilVerdict: () => ({
    data: { status: "not_started" },
    isLoading: false,
  }),
}));

import { CouncilVerdictTable } from "@/components/desk/CouncilVerdictTable";

describe("CouncilVerdictTable empty state", () => {
  it("shows contextual message when paper trading has not started", () => {
    render(<CouncilVerdictTable />);

    expect(screen.getByText(/paper trading not yet started/i)).toBeInTheDocument();
    expect(screen.getByText("Lab")).toBeInTheDocument();
  });

  it("does not render the table when not started", () => {
    const { container } = render(<CouncilVerdictTable />);

    const table = container.querySelector("table");
    expect(table).toBeNull();
  });
});
