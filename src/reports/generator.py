"""PDF report generation for UrbanTransit IQ.

Generates professional PDF reports from actual project analytics data.
Supports filtered reports based on dashboard state.
"""

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import settings
from src.analytics.core import read_processed
from src.audit import timeline, summary as audit_summary
from src.log import get_logger
from src.paths import DATA_ANALYTICS, DATA_METADATA, REPORTS_ROOT
from src.services.data_access import (
    load_route_metrics, load_crowding_analytics, load_demand_analytics,
    load_stop_metrics, load_segmentation, load_alerts, load_recommendations,
    calculate_kpis
)
from src.storage import read_dataset

logger = get_logger(__name__)

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, KeepTogether, Image, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("reportlab not available - PDF generation disabled")

# Try to import kaleido for Plotly chart export
try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False
    logger.warning("kaleido not available - Plotly chart embedding disabled")


class ReportGenerator:
    """Generates PDF reports from analytics data."""

    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError("reportlab not installed. Install with: pip install reportlab")

        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Add custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Title'],
            fontSize=24,
            leading=28,
            spaceAfter=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1f4e79'),
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            leading=18,
            spaceBefore=16,
            spaceAfter=8,
            textColor=colors.HexColor('#2e5c8a'),
            borderWidth=0,
            borderPadding=0,
        ))
        self.styles.add(ParagraphStyle(
            name='SubSectionHeader',
            parent=self.styles['Heading3'],
            fontSize=12,
            leading=16,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor('#3a7ca5'),
        ))
        self.styles.add(ParagraphStyle(
            name='BodyTextCustom',
            parent=self.styles['BodyText'],
            fontSize=10,
            leading=13,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
        ))
        self.styles.add(ParagraphStyle(
            name='TableHeader',
            parent=self.styles['Normal'],
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.white,
            fontName='Helvetica-Bold',
        ))
        self.styles.add(ParagraphStyle(
            name='TableCell',
            parent=self.styles['Normal'],
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
        ))
        self.styles.add(ParagraphStyle(
            name='TableCellLeft',
            parent=self.styles['Normal'],
            fontSize=8,
            leading=10,
            alignment=TA_LEFT,
        ))
        self.styles.add(ParagraphStyle(
            name='Footnote',
            parent=self.styles['Normal'],
            fontSize=8,
            leading=10,
            textColor=colors.grey,
            alignment=TA_LEFT,
        ))

    def _header_footer(self, canvas, doc):
        """Add header and footer to each page."""
        canvas.saveState()
        # Header
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(1*inch, A4[1] - 0.5*inch, "UrbanTransit IQ — Smart Public Transport Analytics")
        canvas.line(1*inch, A4[1] - 0.55*inch, A4[0] - 1*inch, A4[1] - 0.55*inch)
        # Footer
        canvas.setFont('Helvetica', 7)
        page_num = doc.page
        canvas.drawCentredString(A4[0]/2, 0.4*inch, f"Page {page_num}")
        canvas.drawString(1*inch, 0.4*inch, f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        canvas.drawRightString(A4[0] - 1*inch, 0.4*inch, "CONFIDENTIAL")
        canvas.restoreState()

    def _kpi_table(self, kpis: dict[str, Any]) -> Table:
        """Create KPI summary table."""
        data = [
            ["KPI", "Value"],
            ["Total Routes", str(kpis.get("total_routes", "N/A"))],
            ["Daily Passengers", f"{kpis.get('total_passengers_daily', 0):,.0f}"],
            ["Avg Delay", f"{kpis.get('avg_delay_min', 0):.1f} min"],
            ["On-Time %", f"{kpis.get('on_time_pct', 0):.1f}%"],
            ["Avg Occupancy", f"{kpis.get('avg_occupancy_pct', 0):.1f}%"],
            ["Overcrowding Events", str(kpis.get("overcrowding_events", "N/A"))],
            ["Persistent Overcrowding Routes", str(kpis.get("persistent_overcrowding_routes", "N/A"))],
            ["Underutilized Routes", str(kpis.get("underutilized_routes", "N/A"))],
            ["Active Alerts", str(kpis.get("active_alerts", "N/A"))],
            ["Critical Alerts", str(kpis.get("critical_alerts", "N/A"))],
            ["Bottleneck Stops", str(kpis.get("bottleneck_stops", "N/A"))],
            ["Avg Route Score", f"{kpis.get('route_score_avg', 0):.1f}"],
        ]
        table = Table(data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4e79')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        return table

    def _route_metrics_table(self, df: pd.DataFrame, max_rows: int = 20) -> Table:
        """Create route metrics table."""
        if df.empty:
            return Paragraph("No route data available", self.styles['BodyTextCustom'])

        cols = ["route_id", "route_name", "category", "route_score", "route_class",
                "passengers", "avg_delay_min", "on_time_share", "occupancy_avg_pct",
                "crowding_share", "bunching_share", "route_rank"]
        available = [c for c in cols if c in df.columns]
        df_display = df[available].head(max_rows).copy()

        if "on_time_share" in df_display.columns:
            df_display["on_time_share"] = (df_display["on_time_share"] * 100).round(1).astype(str) + "%"
        if "crowding_share" in df_display.columns:
            df_display["crowding_share"] = (df_display["crowding_share"] * 100).round(1).astype(str) + "%"
        if "bunching_share" in df_display.columns:
            df_display["bunching_share"] = (df_display["bunching_share"] * 100).round(1).astype(str) + "%"
        if "avg_delay_min" in df_display.columns:
            df_display["avg_delay_min"] = df_display["avg_delay_min"].round(1)
        if "occupancy_avg_pct" in df_display.columns:
            df_display["occupancy_avg_pct"] = df_display["occupancy_avg_pct"].round(1)
        if "route_score" in df_display.columns:
            df_display["route_score"] = df_display["route_score"].round(1)

        data = [df_display.columns.tolist()] + df_display.values.tolist()

        col_widths = [0.7*inch, 1*inch, 0.6*inch, 0.7*inch, 0.9*inch,
                      0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.5*inch]
        col_widths = col_widths[:len(df_display.columns)]

        table = Table(data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4e79')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        return table

    def _alerts_table(self, df: pd.DataFrame, max_rows: int = 30) -> Table:
        """Create alerts table."""
        if df.empty:
            return Paragraph("No alerts", self.styles['BodyTextCustom'])

        cols = ["alert_id", "alert_type", "severity", "status", "affected_route",
                "title", "timestamp"]
        available = [c for c in cols if c in df.columns]
        df_display = df[available].head(max_rows).copy()

        if "severity" in df_display.columns:
            df_display["severity"] = df_display["severity"].str.upper()
        if "status" in df_display.columns:
            df_display["status"] = df_display["status"].str.upper()
        if "timestamp" in df_display.columns:
            df_display["timestamp"] = pd.to_datetime(df_display["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")

        data = [df_display.columns.tolist()] + df_display.values.tolist()

        table = Table(data, colWidths=[0.8*inch, 1.2*inch, 0.6*inch, 0.6*inch,
                                        0.8*inch, 2.5*inch, 1.2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4e79')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        return table

    def _recommendations_table(self, df: pd.DataFrame, max_rows: int = 20) -> Table:
        """Create recommendations table."""
        if df.empty:
            return Paragraph("No recommendations", self.styles['BodyTextCustom'])

        cols = ["recommendation_id", "category", "priority", "affected_route",
                "title", "suggested_action", "confidence"]
        available = [c for c in cols if c in df.columns]
        df_display = df[available].head(max_rows).copy()

        if "confidence" in df_display.columns:
            df_display["confidence"] = (df_display["confidence"] * 100).round(0).astype(int).astype(str) + "%"

        data = [df_display.columns.tolist()] + df_display.values.tolist()

        table = Table(data, colWidths=[0.8*inch, 1*inch, 0.6*inch, 0.7*inch,
                                        2*inch, 2*inch, 0.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4e79')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        return table

    def _create_plotly_chart_image(self, fig, width: int = 800, height: int = 500) -> bytes | None:
        """Export a Plotly figure to image bytes for PDF embedding."""
        if not KALEIDO_AVAILABLE:
            logger.warning("kaleido not available - cannot export Plotly chart")
            return None
        try:
            img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
            return img_bytes
        except Exception as e:
            logger.warning(f"Failed to export Plotly chart: {e}")
            return None

    def _add_chart_to_story(self, story: list, fig, title: str = "", width: float = 6*inch, height: float = 3.5*inch):
        """Add a Plotly chart to the PDF story as an image."""
        if not KALEIDO_AVAILABLE:
            story.append(Paragraph(f"[Chart: {title} - kaleido not available for embedding]", self.styles['Footnote']))
            return

        img_bytes = self._create_plotly_chart_image(fig)
        if img_bytes is None:
            story.append(Paragraph(f"[Chart: {title} - export failed]", self.styles['Footnote']))
            return

        try:
            img_buffer = io.BytesIO(img_bytes)
            img = Image(img_buffer, width=width, height=height)
            if title:
                story.append(Paragraph(title, self.styles['SubSectionHeader']))
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
        except Exception as e:
            logger.warning(f"Failed to embed chart in PDF: {e}")
            story.append(Paragraph(f"[Chart: {title} - embed failed]", self.styles['Footnote']))

    def _create_route_score_chart(self, route_metrics: pd.DataFrame):
        """Create route score distribution chart."""
        import plotly.express as px
        fig = px.histogram(
            route_metrics, x="route_score", nbins=20,
            title="Route Score Distribution",
            labels={"route_score": "Route Score", "count": "Number of Routes"},
            color_discrete_sequence=["#1f77b4"]
        )
        fig.update_layout(height=400, showlegend=False)
        return fig

    def _create_delay_vs_reliability_chart(self, route_metrics: pd.DataFrame):
        """Create delay vs reliability scatter chart."""
        import plotly.express as px
        fig = px.scatter(
            route_metrics, x="avg_delay_min", y="on_time_share",
            size="passengers", color="category",
            hover_data=["route_id", "route_name", "route_class"],
            labels={"avg_delay_min": "Avg Delay (min)", "on_time_share": "On-Time Share"},
            title="Delay vs Reliability"
        )
        fig.add_hline(y=0.75, line_dash="dash", annotation_text="Degraded Threshold")
        fig.add_hline(y=0.55, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
        fig.update_layout(height=400)
        return fig

    def _create_occupancy_vs_crowding_chart(self, route_metrics: pd.DataFrame):
        """Create occupancy vs crowding scatter chart."""
        import plotly.express as px
        fig = px.scatter(
            route_metrics, x="occupancy_avg_pct", y="crowding_share",
            size="passengers", color="category",
            hover_data=["route_id", "route_name"],
            labels={"occupancy_avg_pct": "Avg Occupancy %", "crowding_share": "Crowding Share"},
            title="Occupancy vs Crowding"
        )
        fig.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="High Crowding")
        fig.add_vline(x=85, line_dash="dash", line_color="red", annotation_text="High Occupancy")
        fig.update_layout(height=400)
        return fig

    def _create_route_class_pie(self, route_metrics: pd.DataFrame):
        """Create route class pie chart."""
        import plotly.express as px
        rel_counts = route_metrics["route_class"].value_counts().reset_index()
        rel_counts.columns = ["Classification", "Count"]
        fig = px.pie(
            rel_counts, values="Count", names="Classification",
            title="Routes by Reliability Classification",
            color="Classification",
            color_discrete_map={
                "Top Performer": "#28a745",
                "Solid": "#1f77b4",
                "Needs Attention": "#ffc107",
                "Action Required": "#dc3545",
            }
        )
        fig.update_layout(height=400)
        return fig

    def _create_daily_demand_chart(self, daily_demand: pd.DataFrame):
        """Create daily demand trend chart."""
        import plotly.express as px
        daily = daily_demand.groupby("date")["passengers"].sum().reset_index()
        fig = px.line(
            daily, x="date", y="passengers",
            title="Daily Passenger Demand Trend",
            labels={"passengers": "Total Passengers", "date": "Date"}
        )
        fig.update_layout(height=400)
        return fig

    def generate_executive_report(
        self,
        output_path: Path,
        filters: dict[str, Any] = None,
    ) -> Path:
        """Generate executive summary report."""
        logger.info(f"Generating executive report to {output_path}")

        kpis = calculate_kpis()
        route_metrics = load_route_metrics()
        crowding = load_crowding_analytics()
        alerts = load_alerts()
        recommendations = load_recommendations()

        story = []

        story.append(Spacer(1, 1.5*inch))
        story.append(Paragraph("UrbanTransit IQ", self.styles['ReportTitle']))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Executive Summary Report", self.styles['Heading2']))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 0.2*inch))
        story.append(PageBreak())

        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            f"This report provides a comprehensive overview of the UrbanTransit IQ network "
            f"as of {datetime.now(timezone.utc).strftime('%Y-%m-%d')}. "
            f"The network comprises {calculate_kpis().get('total_routes', 0)} routes serving "
            f"{calculate_kpis().get('total_passengers_daily', 0):,.0f} daily passengers "
            f"across {calculate_kpis().get('active_stops', 0)} stops. "
            f"Overall on-time performance is {calculate_kpis().get('on_time_pct', 0):.1f}% "
            f"with an average delay of {calculate_kpis().get('avg_delay_min', 0):.1f} minutes.",
            self.styles['BodyTextCustom']
        ))
        story.append(Spacer(1, 0.2*inch))

        story.append(Paragraph("Key Performance Indicators", self.styles['SectionHeader']))
        story.append(self._kpi_table(calculate_kpis()))
        story.append(Spacer(1, 0.2*inch))

        route_metrics = load_route_metrics()
        if not route_metrics.empty:
            story.append(Paragraph("Top Performing Routes", self.styles['SubSectionHeader']))
            top_routes = route_metrics.nlargest(10, "route_score")
            story.append(self._route_metrics_table(top_routes, max_rows=10))
            story.append(Spacer(1, 0.1*inch))

            story.append(Paragraph("Routes Needing Attention", self.styles['SubSectionHeader']))
            bottom_routes = route_metrics.nsmallest(10, "route_score")
            story.append(self._route_metrics_table(bottom_routes, max_rows=10))
            story.append(Spacer(1, 0.2*inch))

            # Add charts
            if KALEIDO_AVAILABLE:
                story.append(Paragraph("Network Visualizations", self.styles['SectionHeader']))
                
                # Route score distribution
                fig1 = self._create_route_score_chart(route_metrics)
                self._add_chart_to_story(story, fig1, "Route Score Distribution")
                
                # Delay vs Reliability
                fig2 = self._create_delay_vs_reliability_chart(route_metrics)
                self._add_chart_to_story(story, fig2, "Delay vs Reliability")
                
                # Occupancy vs Crowding
                fig3 = self._create_occupancy_vs_crowding_chart(route_metrics)
                self._add_chart_to_story(story, fig3, "Occupancy vs Crowding")
                
                # Route class pie
                fig4 = self._create_route_class_pie(route_metrics)
                self._add_chart_to_story(story, fig4, "Route Reliability Classification")
                
                story.append(PageBreak())

        crowding = load_crowding_analytics()
        if not crowding.get("overcrowding_events", pd.DataFrame()).empty:
            story.append(Paragraph("Crowding Overview", self.styles['SectionHeader']))
            oc = crowding["overcrowding_events"]
            story.append(Paragraph(
                f"Total overcrowding events: {len(oc)}. "
                f"Persistent overcrowding routes: {len(crowding.get('persistent_overcrowding', pd.DataFrame()))}. "
                f"Underutilized route-timebands: {len(crowding.get('underutilization', pd.DataFrame()))}.",
                self.styles['BodyTextCustom']
            ))
            story.append(Spacer(1, 0.2*inch))

        alerts = load_alerts()
        if not alerts.empty:
            story.append(Paragraph("Active Alerts Summary", self.styles['SectionHeader']))
            active_alerts = alerts[alerts["status"] == "new"]
            critical = len(active_alerts[active_alerts["severity"] == "critical"])
            high = len(active_alerts[active_alerts["severity"] == "high"])
            story.append(Paragraph(
                f"Active alerts: {len(active_alerts)} total "
                f"({critical} critical, {high} high). "
                f"Most common: {load_alerts()['alert_type'].value_counts().head(3).to_dict()}",
                self.styles['BodyTextCustom']
            ))
            story.append(Spacer(1, 0.2*inch))

        recommendations = load_recommendations()
        if not recommendations.empty:
            story.append(Paragraph("Top Recommendations", self.styles['SectionHeader']))
            story.append(self._recommendations_table(recommendations.head(10)))
            story.append(Spacer(1, 0.2*inch))

        # Daily demand chart
        if KALEIDO_AVAILABLE:
            demand = load_demand_analytics()
            daily_demand = demand.get("daily_demand", pd.DataFrame())
            if not daily_demand.empty:
                story.append(Paragraph("Demand Trends", self.styles['SectionHeader']))
                fig5 = self._create_daily_demand_chart(daily_demand)
                self._add_chart_to_story(story, fig5, "Daily Passenger Demand Trend")

        story.append(Spacer(1, 0.5*inch))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        story.append(Paragraph(
            f"Report generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}<br/>"
            f"Data sources: UrbanTransit IQ analytics pipeline",
            self.styles['Footnote']
        ))

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=1*inch,
            rightMargin=1*inch,
            topMargin=1*inch,
            bottomMargin=1*inch,
        )

        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)
        logger.info(f"Report saved to {output_path}")
        return output_path


SUPPORTED_REPORT_TYPES = ("executive",)


def generate_pdf_report(
    report_type: str = "executive",
    output_path: Path = None,
    filters: dict[str, Any] = None,
    **kwargs,
) -> Path:
    """Generate a PDF report and return the path of the file written.

    This previously returned an output path without ever creating a file, so the
    Reports page silently produced nothing. It now delegates to
    :class:`ReportGenerator`, which actually builds the document.
    """
    if report_type not in SUPPORTED_REPORT_TYPES:
        raise ValueError(
            f"Unsupported report_type {report_type!r}; "
            f"expected one of {list(SUPPORTED_REPORT_TYPES)}"
        )

    if output_path is None:
        REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        output_path = REPORTS_ROOT / f"report_{report_type}_{stamp}.pdf"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = ReportGenerator()
    return Path(generator.generate_executive_report(output_path, filters=filters))


if __name__ == "__main__":
    try:
        from reportlab.lib import colors
    except ImportError:
        print("reportlab not available - skipping test")
    else:
        # Quick test
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch
        
        doc = SimpleDocTemplate("test_report.pdf", pagesize=A4)
        styles = getSampleStyleSheet()
        story = [Paragraph("Test Report", styles['Title'])]
        doc.build(story)
        print("Test PDF generated successfully")