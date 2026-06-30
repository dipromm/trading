"use client";

import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallbackMessage?: string;
}

interface State {
  hasError: boolean;
  errorTime: string | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, errorTime: null };
  }

  static getDerivedStateFromError(): State {
    return {
      hasError: true,
      errorTime: new Date().toLocaleString(),
    };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="rounded-lg border border-border bg-card p-8 text-center">
          <p className="text-muted-foreground">
            {this.props.fallbackMessage || "System offline"} · Last data:{" "}
            <span className="font-mono text-xs">
              {this.state.errorTime || "unknown"}
            </span>
          </p>
          <button
            className="mt-4 text-xs text-accent-foreground border border-border rounded px-3 py-1.5 hover:bg-muted transition-colors"
            onClick={() => this.setState({ hasError: false, errorTime: null })}
          >
            Retry
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
