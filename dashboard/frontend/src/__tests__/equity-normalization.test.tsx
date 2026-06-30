/**
 * Test 3: Base-100 normalization of the equity curve
 *
 * The EquityChart component receives raw portfolio values from the API
 * and the data is already normalized before reaching the component.
 * This test verifies the normalization logic that converts raw values
 * to a base-100 index for visual comparison.
 */
import "@testing-library/jest-dom";

function normalizeToBase100(data: { portfolio_value: number; buyhold_value: number | null }[]) {
  if (data.length === 0) return data;

  const firstPortfolio = data[0].portfolio_value;
  const firstBuyhold = data[0].buyhold_value;

  return data.map((point) => ({
    ...point,
    portfolio_value: (point.portfolio_value / firstPortfolio) * 100,
    buyhold_value:
      point.buyhold_value != null && firstBuyhold != null
        ? (point.buyhold_value / firstBuyhold) * 100
        : null,
  }));
}

describe("Equity curve base-100 normalization", () => {
  it("normalizes first portfolio point to exactly 100", () => {
    const raw = [
      { portfolio_value: 10000, buyhold_value: 10000 },
      { portfolio_value: 10500, buyhold_value: 10200 },
      { portfolio_value: 9800, buyhold_value: 9900 },
    ];

    const normalized = normalizeToBase100(raw);

    expect(normalized[0].portfolio_value).toBe(100);
    expect(normalized[0].buyhold_value).toBe(100);
  });

  it("correctly calculates relative changes", () => {
    const raw = [
      { portfolio_value: 10000, buyhold_value: 10000 },
      { portfolio_value: 11000, buyhold_value: 10500 },
    ];

    const normalized = normalizeToBase100(raw);

    expect(normalized[1].portfolio_value).toBeCloseTo(110, 10);
    expect(normalized[1].buyhold_value).toBeCloseTo(105, 10);
  });

  it("handles null buyhold values", () => {
    const raw = [
      { portfolio_value: 10000, buyhold_value: null },
      { portfolio_value: 10500, buyhold_value: null },
    ];

    const normalized = normalizeToBase100(raw);

    expect(normalized[0].portfolio_value).toBe(100);
    expect(normalized[0].buyhold_value).toBeNull();
    expect(normalized[1].buyhold_value).toBeNull();
  });

  it("returns empty array for empty input", () => {
    expect(normalizeToBase100([])).toEqual([]);
  });

  it("preserves proportional relationships regardless of starting capital", () => {
    const capital50k = [
      { portfolio_value: 50000, buyhold_value: 50000 },
      { portfolio_value: 55000, buyhold_value: 52000 },
    ];
    const capital10k = [
      { portfolio_value: 10000, buyhold_value: 10000 },
      { portfolio_value: 11000, buyhold_value: 10400 },
    ];

    const norm50k = normalizeToBase100(capital50k);
    const norm10k = normalizeToBase100(capital10k);

    expect(norm50k[1].portfolio_value).toBe(norm10k[1].portfolio_value);
    expect(norm50k[1].buyhold_value).toBe(norm10k[1].buyhold_value);
  });
});
