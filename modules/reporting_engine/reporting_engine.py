import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, 
    PageBreak, KeepTogether, ListFlowable, ListItem
)

class ReportingEngine:
    """
    Advanced Reporting Engine.
    Generates structured, multi-document PDF reports with smart layouts (grids) 
    and automated narrative analysis.
    """
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._init_styles()
        
    def _init_styles(self):
        """Initialize professional report styles."""
        self.styles = getSampleStyleSheet()
        
        # Custom Title
        self.title_style = ParagraphStyle(
            'ReportTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=20,
            textColor=colors.darkblue
        )
        
        # Heading 1 (Section)
        self.h1 = ParagraphStyle(
            'CustomH1',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceBefore=12,
            spaceAfter=10,
            textColor=colors.darkblue,
            keepWithNext=True
        )
        
        # Heading 2 (Subsection)
        self.h2 = ParagraphStyle(
            'CustomH2',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.black,
            keepWithNext=True
        )
        
        # Normal Text
        self.normal = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=6
        )
        
        # Captions
        self.caption_style = ParagraphStyle(
            'Caption',
            parent=self.styles['Normal'],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.dimgrey,
            spaceAfter=12
        )

    def generate_report(self, report_data: Dict[str, Any], run_id: str) -> str:
        """
        Orchestrates the generation of multiple report files.
        Returns the path to the primary Executive Summary.
        """
        self.logger.info("Starting Report Generation Phase...")
        
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        output_dir = Path(base_dir) / "12_REPORTING"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Generate Executive Summary (High Level)
        exec_path = output_dir / f"01_Executive_Summary_{run_id}.pdf"
        self._build_executive_summary(exec_path, report_data, run_id)
        
        # 2. Generate Technical Deep Dive (Full Details)
        tech_path = output_dir / f"02_Technical_Deep_Dive_{run_id}.pdf"
        self._build_technical_report(tech_path, report_data, run_id)
        
        return str(exec_path)

    # =========================================================================
    #                       REPORT 1: EXECUTIVE SUMMARY
    # =========================================================================
    def _build_executive_summary(self, path: Path, data: Dict, run_id: str):
        doc = SimpleDocTemplate(str(path), pagesize=letter)
        story = []
        
        # Title Page
        story.append(Paragraph("Offshore Riser ML Prediction System", self.title_style))
        story.append(Paragraph(f"Executive Summary | Run ID: {run_id}", self.h2))
        story.append(Spacer(1, 0.5*inch))
        
        # 1. Key Performance Indicators (KPIs)
        story.append(Paragraph("1. Key Performance Indicators", self.h1))
        story.append(self._create_kpi_table(data))
        story.append(Spacer(1, 0.2*inch))
        
        # 2. Automated Narrative
        story.append(Paragraph("2. Performance Narrative", self.h1))
        narrative = self._generate_smart_narrative(data)
        story.append(Paragraph(narrative, self.normal))
        story.append(Spacer(1, 0.2*inch))
        
        # 3. Top Critical Plots (Just the best 2)
        story.append(Paragraph("3. Critical Diagnostics", self.h1))
        plots = data.get('plots', [])
        # Filter for 'actual_vs_pred' and 'error_hist'
        critical_plots = [p for p in plots if 'actual_vs_pred' in str(p) or 'error_hist' in str(p)]
        # Fallback if specific plots not found
        if not critical_plots and plots:
            critical_plots = plots[:2]
            
        self._add_image_grid(story, critical_plots[:2], cols=2)
        
        doc.build(story)
        self.logger.info(f"Generated Executive Summary: {path}")

    # =========================================================================
    #                       REPORT 2: TECHNICAL DEEP DIVE
    # =========================================================================
    def _build_technical_report(self, path: Path, data: Dict, run_id: str):
        doc = SimpleDocTemplate(str(path), pagesize=letter)
        story = []
        
        story.append(Paragraph("Technical Analysis & Diagnostics", self.title_style))
        story.append(Paragraph(f"Detailed Engineering Report | Run ID: {run_id}", self.normal))
        story.append(Spacer(1, 0.3*inch))
        
        # 1. Full Metrics Evaluation
        story.append(Paragraph("1. Comprehensive Model Evaluation", self.h1))
        story.append(Paragraph("Detailed breakdown of model performance across Validation and Test sets.", self.normal))
        story.append(self._create_full_metrics_table(data.get('metrics', {})))
        story.append(Spacer(1, 0.2*inch))
        
        # 2. Hyperparameter Optimization (HPO)
        hpo_data = data.get('hpo_analysis', {})
        if hpo_data.get('enabled', False):
            story.append(PageBreak())
            story.append(Paragraph("2. Hyperparameter Optimization Analysis", self.h1))
            self._add_hpo_details(story, hpo_data)
        
        # 3. Full Diagnostics Suite
        story.append(PageBreak())
        story.append(Paragraph("3. Full Diagnostic Suite", self.h1))
        story.append(Paragraph("Visualizing residuals, distributions, and physics compliance.", self.normal))
        
        all_plots = data.get('plots', [])
        self._add_image_grid(story, all_plots, cols=2)
        
        doc.build(story)
        self.logger.info(f"Generated Technical Deep Dive: {path}")

    # =========================================================================
    #                           COMPONENT BUILDERS
    # =========================================================================
    
    def _create_kpi_table(self, data: Dict) -> Table:
        """Compact KPI table for Executive Summary."""
        m_test = data.get('metrics', {}).get('test', {})
        
        # Safe Retrieval with corrected key lookup
        cmae = m_test.get('cmae', m_test.get('cmae_deg', 0.0))
        # FIX: Ensure we look for 'accuracy_at_5deg' as produced by EvaluationEngine
        acc_5 = m_test.get('accuracy_at_5deg', 0.0) 
        
        kpi_data = [
            ["Metric", "Test Result", "Target / Status"],
            ["Circular MAE", f"{cmae:.4f}°", "Lower is better"],
            ["Accuracy @ 5°", f"{acc_5:.2f}%", "> 90% (Target)"]
        ]
        
        t = Table(kpi_data, colWidths=[2.5*inch, 2*inch, 2.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.navy),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 1, colors.lightgrey),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        return t

    def _create_full_metrics_table(self, metrics: Dict) -> Table:
        """Detailed comparison table."""
        # Helper to find keys
        def find_val(d, candidates):
            for k in candidates:
                if k in d: return d[k]
            return 0.0

        val_m = metrics.get('val', {})
        test_m = metrics.get('test', {})
        
        rows = [["Metric", "Validation Set", "Test Set", "Delta"]]
        
        # Define metrics to show (Standardized Keys)
        configs = [
            ("CMAE (°)", ['cmae', 'cmae_deg']),
            ("CRMSE (°)", ['crmse', 'crmse_deg']),
            ("Max Error (°)", ['max_error', 'max_error_deg']),
            ("Acc @ 5° (%)", ['accuracy_at_5deg']),
            ("Acc @ 10° (%)", ['accuracy_at_10deg']),
        ]
        
        for label, keys in configs:
            v_val = find_val(val_m, keys)
            v_test = find_val(test_m, keys)
            delta = v_test - v_val
            
            rows.append([
                label, 
                f"{v_val:.4f}", 
                f"{v_test:.4f}", 
                f"{delta:+.4f}"
            ])
            
        t = Table(rows, colWidths=[2*inch, 2*inch, 2*inch, 1.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkslategray),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (1,1), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.aliceblue, colors.white]),
        ]))
        return t

    def _generate_smart_narrative(self, data: Dict) -> str:
        """Generates dynamic text based on metrics."""
        m = data.get('metrics', {}).get('test', {})
        cmae = m.get('cmae', m.get('cmae_deg', 0))
        acc = m.get('accuracy_at_5deg', 0)
        
        model_name = data.get('model_info', {}).get('model', 'The model')
        
        text = f"The optimized <b>{model_name}</b> architecture achieved a test CMAE of <b>{cmae:.4f}°</b>. "
        
        if acc > 95:
            text += "The model demonstrates <b>exceptional stability</b>, with over 95% of predictions falling within the 5-degree safety margin. "
        elif acc > 85:
            text += "The model shows <b>strong performance</b>, suitable for general operational guidance, though edge cases may require review. "
        else:
            text += "Performance indicates <b>potential volatility</b> under certain conditions. Further feature engineering or data cleaning is recommended. "
            
        text += "Please refer to the Technical Deep Dive document for granular error distribution analysis and hyperparameter sensitivity plots."
        return text

    def _add_hpo_details(self, story, hpo_data):
        """Adds HPO tables and plots with layout control."""
        # Tables
        tables = hpo_data.get('tables', {})
        for model, rows in tables.items():
            story.append(Paragraph(f"Optimal Parameters: {model}", self.h2))
            if len(rows) > 0:
                col_w = 7.0 / len(rows[0])
                t = Table(rows, colWidths=[col_w*inch]*len(rows[0]))
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.teal),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('GRID', (0,0), (-1,-1), 1, colors.black),
                    ('FONTSIZE', (0,0), (-1,-1), 8)
                ]))
                # Keep table with header
                story.append(KeepTogether([t, Spacer(1, 0.2*inch)]))

        # Plots
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Parameter Landscape Visualizations", self.h2))
        
        # Flatten plot structure
        all_plots = []
        models_plots = hpo_data.get('models', {})
        for _, metric_map in models_plots.items():
            if isinstance(metric_map, dict):
                for _, paths in metric_map.items():
                    all_plots.extend(paths)
        
        if all_plots:
            self._add_image_grid(story, sorted(all_plots), cols=2)
        else:
            story.append(Paragraph("No HPO visualization plots found.", self.normal))

    def _add_image_grid(self, story, image_paths: List[str], cols=2):
        """
        Arranges images in a Grid (Table) to save space.
        cols=2 means 2 images per row.
        """
        if not image_paths:
            return

        # Prepare grid data
        grid_data = []
        current_row = []
        
        # Image dimensions for 2-column layout
        # Page width is ~8.5 inch. Margins ~1 inch. Usable ~7.5.
        # So each image max 3.5 inch wide.
        img_width = 3.4 * inch 
        img_height = 2.6 * inch
        
        for p_str in image_paths:
            path = Path(p_str)
            if not path.exists():
                continue
                
            # Create Image Flowable
            img = Image(str(path), width=img_width, height=img_height, kind='proportional')
            
            # Caption
            caption_text = path.stem.replace('_', ' ').title()
            # Wrap image and caption in a mini-list to keep them together in the cell
            cell_content = [img, Paragraph(caption_text, self.caption_style)]
            
            current_row.append(cell_content)
            
            if len(current_row) == cols:
                grid_data.append(current_row)
                current_row = []
        
        # Append leftovers
        if current_row:
            # Fill empty cells with empty string
            while len(current_row) < cols:
                current_row.append("")
            grid_data.append(current_row)
            
        # Create the Table
        if grid_data:
            t = Table(grid_data, colWidths=[img_width + 0.1*inch] * cols)
            t.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('LEFTPADDING', (0,0), (-1,-1), 2),
                ('RIGHTPADDING', (0,0), (-1,-1), 2),
                ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.1*inch))