import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether

class ReportingEngine:
    """
    Generates a professional PDF report with:
    - Executive Summary & Narrative
    - Hyperparameter Analysis (Tables + Plots)
    - Detailed Metrics (Validation vs Test)
    - Diagnostic Visualization
    """
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.styles = getSampleStyleSheet()
        self.title_style = self.styles['Title']
        self.h1 = self.styles['Heading1']
        self.h2 = self.styles['Heading2']
        self.normal = self.styles['Normal']
        
    def generate_report(self, report_data: Dict[str, Any], run_id: str) -> str:
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        output_dir = Path(base_dir) / "10_REPORT"
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / "final_report.pdf"
        
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        story = []
        
        # 1. Header & Overview
        story.append(Paragraph(f"Offshore Riser ML Report: {run_id}", self.title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Executive Summary
        story.append(Paragraph("1. Executive Summary", self.h1))
        story.append(self._create_summary_table(report_data))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(self._generate_summary_narrative(report_data), self.normal))
        
        # 2. Hyperparameter Analysis (New Section)
        hpo_data = report_data.get('hpo_analysis', {})
        if hpo_data.get('enabled'):
            story.append(PageBreak())
            story.append(Paragraph("2. Hyperparameter Analysis", self.h1))
            self._add_hpo_section(story, hpo_data)
        
        # 3. Model Evaluation
        story.append(PageBreak())
        story.append(Paragraph("3. Model Evaluation", self.h1))
        story.append(self._create_metrics_table(report_data.get('metrics', {})))
        story.append(Spacer(1, 0.2*inch))
        
        # 4. Diagnostics
        story.append(Paragraph("4. Diagnostic Plots", self.h1))
        self._add_diagnostic_plots(story, report_data.get('plots', []))
        
        try:
            doc.build(story)
            self.logger.info(f"Report generated successfully: {pdf_path}")
            return str(pdf_path)
        except Exception as e:
            self.logger.error(f"PDF Gen failed: {e}")
            return ""

    def _add_hpo_section(self, story, hpo_data):
        """Adds Optimal Ranges Tables and Parameter Landscape Plots."""
        story.append(Paragraph("Optimal Parameter Ranges", self.h2))
        story.append(Paragraph("The following tables show the top performing parameter configurations found during search.", self.normal))
        story.append(Spacer(1, 0.1*inch))
        
        # Add Tables per model
        tables = hpo_data.get('tables', {})
        if not tables:
             story.append(Paragraph("No optimization tables available.", self.normal))

        for model_name, table_rows in tables.items():
            story.append(Paragraph(f"Model: {model_name}", self.h2))
            if table_rows and len(table_rows) > 1:
                # Calculate cols based on data
                col_count = len(table_rows[0])
                # Simple style
                t = Table(table_rows, colWidths=[6.5*inch/col_count]*col_count)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.darkgrey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('GRID', (0,0), (-1,-1), 1, colors.black),
                    ('FONTSIZE', (0,0), (-1,-1), 8),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.aliceblue])
                ]))
                story.append(t)
                story.append(Spacer(1, 0.2*inch))
        
        # Add Plots per model
        story.append(PageBreak())
        story.append(Paragraph("Parameter Landscape Visualizations", self.h2))
        
        models_plots = hpo_data.get('models', {})
        if not models_plots:
            story.append(Paragraph("No visualization plots available.", self.normal))

        for model_name, plot_paths in models_plots.items():
            story.append(Paragraph(f"Analysis for {model_name}", self.h2))
            if not plot_paths:
                story.append(Paragraph("No plots available.", self.normal))
                continue
            
            # Sort plots to keep related ones together (heatmap, contour, 3d)
            plot_paths.sort()
            
            for path in plot_paths:
                # Extract a readable caption from filename
                caption = Path(path).stem.replace('_', ' ').title()
                self._add_image(story, path, width=6*inch, height=4.5*inch, caption=caption)

    def _add_diagnostic_plots(self, story, plots):
        for path in plots:
             caption = Path(path).stem.replace('_', ' ').title()
             self._add_image(story, path, width=6*inch, height=4*inch, caption=caption)

    def _add_image(self, story, path, width, height, caption=None):
        if Path(path).exists():
            # preserveAspectRatio=True ensures no distortion
            img = Image(str(path), width=width, height=height, kind='proportional')
            elements = [img]
            if caption:
                elements.append(Paragraph(caption, self.normal))
            elements.append(Spacer(1, 0.2*inch))
            story.append(KeepTogether(elements))

    def _create_summary_table(self, data):
        metrics = data.get('metrics', {}).get('test', {})
        
        # Fix Accuracy Bug: Check keys and scale
        acc = metrics.get('accuracy_5deg', metrics.get('accuracy_within_5deg', 0))
        # Heuristic: if accuracy is 0.0 < acc < 1.0, it's likely a probability -> scale to %
        if 0.0 < acc <= 1.0: 
            acc *= 100 
        
        info = [
            ["Metric", "Value"],
            ["Run ID", data.get('run_metadata', {}).get('id', 'N/A')],
            ["Model", data.get('model_info', {}).get('model', 'N/A')],
            ["Test CMAE", f"{metrics.get('cmae', metrics.get('cmae_deg', 0)):.4f}°"],
            ["Accuracy @ 5°", f"{acc:.2f}%"]
        ]
        t = Table(info, colWidths=[2.5*inch, 2.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ]))
        return t

    def _create_metrics_table(self, metrics):
        # Helper to get value with multiple potential keys
        def get_val(d, keys):
            for k in keys:
                if k in d: return d[k]
            return 0.0
            
        data = [["Metric", "Validation", "Test"]]
        metric_defs = [
            ("CMAE (°)", ['cmae', 'cmae_deg']),
            ("CRMSE (°)", ['crmse', 'crmse_deg']),
            ("Max Error (°)", ['max_error', 'max_error_deg']),
            ("Acc @ 5° (%)", ['accuracy_5deg', 'accuracy_within_5deg']),
            ("Acc @ 10° (%)", ['accuracy_10deg', 'accuracy_within_10deg'])
        ]
        
        for label, keys in metric_defs:
            v_val = get_val(metrics.get('val', {}), keys)
            v_test = get_val(metrics.get('test', {}), keys)
            
            # Auto-scale accuracy if it looks like a probability (<=1.0)
            if "Acc" in label:
                if 0 < v_val <= 1.0: v_val *= 100
                if 0 < v_test <= 1.0: v_test *= 100
                
            data.append([label, f"{v_val:.4f}", f"{v_test:.4f}"])
            
        t = Table(data, colWidths=[2*inch, 2*inch, 2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
            ('ALIGN', (1,1), (-1,-1), 'CENTER')
        ]))
        return t

    def _generate_summary_narrative(self, data):
        m = data.get('metrics', {}).get('test', {})
        cmae = m.get('cmae', m.get('cmae_deg', 0))
        model = data.get('model_info', {}).get('model', 'Unknown Model')
        
        narrative = (
            f"The <b>{model}</b> model was selected as the optimal architecture. "
            f"On the independent test set, it achieved a Circular Mean Absolute Error (CMAE) of <b>{cmae:.4f}°</b>. "
            "The following sections provide a deep dive into the hyperparameter optimization process, "
            "comparing parameter landscapes to identify stable operating regions, followed by "
            "comprehensive diagnostic plots for the final model."
        )
        return narrative