"""Data analysis tools for pandas/SQL operations."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

from dawu_agent.tools.base import ConcurrencyMode, Tool, ToolCategory, ToolResult


class DataQueryTool(Tool):
    """Query and transform data using pandas or SQL."""

    @property
    def name(self) -> str:
        return "data_query"

    @property
    def description(self) -> str:
        return (
            "Execute data queries and transformations using pandas or SQL. "
            "Supports CSV, Excel, JSON, and Parquet files. "
            "Use this for filtering, aggregating, joining, or analyzing datasets. "
            "Do NOT use this for general file reading - use file_read instead.\n"
            "Parameters:\n"
            "  - source (string, required): Path to data file or 'memory' for in-memory data\n"
            "  - query_type (string, required): 'pandas' or 'sql'\n"
            "  - operation (string, required): Python code (pandas) or SQL query\n"
            "  - output_path (string, optional): Save result to this path\n"
            "Returns: Query results as formatted string or summary"
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Path to data file or 'memory'",
                },
                "query_type": {
                    "type": "string",
                    "enum": ["pandas", "sql"],
                    "description": "Query language to use",
                },
                "operation": {
                    "type": "string",
                    "description": "Pandas code or SQL query to execute",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional path to save results",
                },
            },
            "required": ["source", "query_type", "operation"],
        }

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.DATA

    @property
    def concurrency_mode(self) -> ConcurrencyMode:
        return ConcurrencyMode.READ_ONLY

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            import pandas as pd
        except ImportError:
            return ToolResult.error("pandas not installed. Run: pip install pandas")

        try:
            source = arguments["source"]
            query_type = arguments["query_type"]
            operation = arguments["operation"]

            # Load data
            if source == "memory":
                return ToolResult.error("In-memory data not yet supported")

            path = Path(source)
            if not path.exists():
                return ToolResult.error(f"Data file not found: {source}")

            # Determine file type and load
            suffix = path.suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(source)
            elif suffix in (".xlsx", ".xls"):
                df = pd.read_excel(source)
            elif suffix == ".json":
                df = pd.read_json(source)
            elif suffix == ".parquet":
                df = pd.read_parquet(source)
            else:
                return ToolResult.error(f"Unsupported file format: {suffix}")

            # Execute query
            if query_type == "pandas":
                # Execute pandas code in restricted namespace
                namespace = {"df": df, "pd": pd}
                result = eval(operation, {"__builtins__": {}}, namespace)

                if isinstance(result, pd.DataFrame):
                    output = f"Shape: {result.shape}\n\n{result.head(50).to_string()}"
                    if len(result) > 50:
                        output += f"\n\n... ({len(result) - 50} more rows)"
                else:
                    output = str(result)

            elif query_type == "sql":
                try:
                    result = df.query(operation) if "@" not in operation else pd.read_sql(operation, None)
                    output = f"Shape: {result.shape}\n\n{result.head(50).to_string()}"
                except Exception as e:
                    return ToolResult.error(f"SQL query failed: {e}")
            else:
                return ToolResult.error(f"Unknown query_type: {query_type}")

            # Save if output_path specified
            if arguments.get("output_path"):
                out_path = Path(arguments["output_path"])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(result, pd.DataFrame):
                    if out_path.suffix == ".csv":
                        result.to_csv(out_path, index=False)
                    elif out_path.suffix == ".json":
                        result.to_json(out_path, orient="records")
                    elif out_path.suffix == ".xlsx":
                        result.to_excel(out_path, index=False)
                    else:
                        result.to_csv(out_path, index=False)
                else:
                    out_path.write_text(str(result))
                output += f"\n\nSaved to: {out_path}"

            return ToolResult.ok(output)

        except Exception as e:
            return ToolResult.error(f"Data query failed: {e}")


class DataVisualizeTool(Tool):
    """Generate data visualizations."""

    @property
    def name(self) -> str:
        return "data_visualize"

    @property
    def description(self) -> str:
        return (
            "Create data visualizations (charts, plots) from datasets. "
            "Supports various chart types for data analysis. "
            "Use this after data_query to visualize findings.\n"
            "Parameters:\n"
            "  - source (string, required): Path to data file\n"
            "  - chart_type (string, required): 'line', 'bar', 'scatter', 'histogram', 'heatmap'\n"
            "  - x_column (string, required): Column for x-axis\n"
            "  - y_column (string, optional): Column for y-axis\n"
            "  - output_path (string, required): Path to save the chart image\n"
            "Returns: Path to generated visualization"
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Path to data file"},
                "chart_type": {
                    "type": "string",
                    "enum": ["line", "bar", "scatter", "histogram", "heatmap"],
                    "description": "Type of chart to generate",
                },
                "x_column": {"type": "string", "description": "Column for x-axis"},
                "y_column": {"type": "string", "description": "Column for y-axis (optional for histogram)"},
                "output_path": {"type": "string", "description": "Path to save chart image"},
            },
            "required": ["source", "chart_type", "x_column", "output_path"],
        }

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.DATA

    @property
    def concurrency_mode(self) -> ConcurrencyMode:
        return ConcurrencyMode.READ_ONLY

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            import pandas as pd
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as e:
            return ToolResult.error(f"Required library not installed: {e}")

        try:
            source = arguments["source"]
            chart_type = arguments["chart_type"]
            x_col = arguments["x_column"]
            y_col = arguments.get("y_column")
            output_path = arguments["output_path"]

            # Load data
            path = Path(source)
            suffix = path.suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(source)
            elif suffix in (".xlsx", ".xls"):
                df = pd.read_excel(source)
            else:
                return ToolResult.error(f"Unsupported format: {suffix}")

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            if chart_type == "line":
                if y_col:
                    df.plot(x=x_col, y=y_col, kind="line", ax=ax)
                else:
                    df[x_col].plot(kind="line", ax=ax)
            elif chart_type == "bar":
                if y_col:
                    df.plot(x=x_col, y=y_col, kind="bar", ax=ax)
                else:
                    df[x_col].value_counts().plot(kind="bar", ax=ax)
            elif chart_type == "scatter":
                if not y_col:
                    return ToolResult.error("y_column required for scatter plot")
                df.plot(x=x_col, y=y_col, kind="scatter", ax=ax)
            elif chart_type == "histogram":
                df[x_col].plot(kind="hist", ax=ax, bins=30)
            elif chart_type == "heatmap":
                import seaborn as sns
                numeric_df = df.select_dtypes(include=["number"])
                sns.heatmap(numeric_df.corr(), annot=True, ax=ax)

            ax.set_title(f"{chart_type.title()} Chart: {x_col}")
            plt.tight_layout()

            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            return ToolResult.ok(f"Chart saved to: {output_path}")

        except Exception as e:
            return ToolResult.error(f"Visualization failed: {e}")


class ReportGenerateTool(Tool):
    """Generate structured reports from analysis results."""

    @property
    def name(self) -> str:
        return "report_generate"

    @property
    def description(self) -> str:
        return (
            "Generate a structured analysis report in Markdown or HTML format. "
            "Use this as the final step after data analysis to compile findings.\n"
            "Parameters:\n"
            "  - title (string, required): Report title\n"
            "  - sections (array, required): List of section objects with 'heading' and 'content'\n"
            "  - format (string, required): 'markdown' or 'html'\n"
            "  - output_path (string, required): Path to save the report\n"
            "Returns: Path to generated report"
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Report title"},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["heading", "content"],
                    },
                    "description": "Report sections",
                },
                "format": {"type": "string", "enum": ["markdown", "html"], "description": "Output format"},
                "output_path": {"type": "string", "description": "Path to save report"},
            },
            "required": ["title", "sections", "format", "output_path"],
        }

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.DATA

    @property
    def concurrency_mode(self) -> ConcurrencyMode:
        return ConcurrencyMode.WRITE

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            title = arguments["title"]
            sections = arguments["sections"]
            fmt = arguments["format"]
            output_path = arguments["output_path"]

            if fmt == "markdown":
                content = f"# {title}\n\n"
                for section in sections:
                    content += f"## {section['heading']}\n\n{section['content']}\n\n"
            elif fmt == "html":
                content = f"""<!DOCTYPE html>
<html>
<head><title>{title}</title></head>
<body>
<h1>{title}</h1>
"""
                for section in sections:
                    content += f"<h2>{section['heading']}</h2>\n<p>{section['content']}</p>\n"
                content += "</body></html>"
            else:
                return ToolResult.error(f"Unknown format: {fmt}")

            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")

            return ToolResult.ok(f"Report saved to: {output_path}")

        except Exception as e:
            return ToolResult.error(f"Report generation failed: {e}")
