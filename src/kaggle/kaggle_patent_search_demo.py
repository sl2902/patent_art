"""
Kaggle Patent Search Interface
Interactive search widget for patent semantic search with explainable results
"""

from IPython.display import display, HTML
import ipywidgets as widgets
import pandas as pd
from typing import List, Optional
from datetime import date
import json
from run_patent_search_pipeline import(
    sanitize_input_query,
    run_semantic_search_pipeline,
    technology_selection,
    generate_random_patents
)

MIN_DATE = date(2024, 1, 1)
MAX_DATE = date(2024, 6, 30)

class PatentSearchInterface:
    def __init__(self, search_function):
        """
        Initialize the patent search interface
        
        Args:
            search_function: Function that takes (start_date, end_date, query_text, patent_ids, top_k) 
                           and returns search results DataFrame
        """
        self.search_function = search_function
        self.start_date = widgets.DatePicker(
            description='Start Date:',
            value=MIN_DATE,
            style={'description_width': 'initial'}
        )
        
        self.end_date = widgets.DatePicker(
            description='End Date:',
            value=MAX_DATE,
            style={'description_width': 'initial'}
        )
        self.start_date.observe(self.validate_dates, names='value')
        self.end_date.observe(self.validate_dates, names='value')
        self.create_interface()

        
    # Date Validation logic
    def validate_dates(self, change=None):
        # Clamp start date
        if self.start_date.value is not None:
            if self.start_date.value < MIN_DATE:
                self.start_date.value = MIN_DATE
            elif self.start_date.value > MAX_DATE:
                self.start_date.value = MAX_DATE
        
        # Clamp end date
        if self.end_date.value is not None:
            if self.end_date.value < MIN_DATE:
                self.end_date.value = MIN_DATE
            elif self.end_date.value > MAX_DATE:
                self.end_date.value = MAX_DATE
        
        # Ensure start <= end
        if self.start_date.value and self.end_date.value:
            if self.start_date.value > self.end_date.value:
                # self.warning_box.value = (
                #     "<div style='color:red;'>⚠️ End date adjusted to match start date.</div>"
                # )
                self.end_date.value = self.start_date.value
            # else:
            #     self.warning_box.value = "" 
    

    
    def create_interface(self):
        """Create the search interface widgets"""
        
        # Search method selection
        self.search_method = widgets.RadioButtons(
            options=['Text Query', 'Patent Numbers'],
            value='Text Query',
            description='Search by:',
            style={'description_width': 'initial'}
        )
        
        # Text query input
        self.query_text = widgets.Textarea(
            value='',
            placeholder='Enter technology description (e.g., machine learning optimization)',
            description='Query:',
            layout=widgets.Layout(width='600px', height='80px'),
            style={'description_width': 'initial'}
        )
        
        # Patent numbers input
        self.patent_numbers = widgets.Textarea(
            value='',
            placeholder='Enter patent numbers, one per line (e.g., US1234567A1)',
            description='Patents:',
            layout=widgets.Layout(width='600px', height='80px'),
            style={'description_width': 'initial'}
        )
        
        # Date inputs
        self.start_date = widgets.DatePicker(
            description='Start Date:',
            value=MIN_DATE,
            style={'description_width': 'initial'}
        )

        self.start_date.observe(self.validate_dates, names='value')
        
        self.end_date = widgets.DatePicker(
            description='End Date:',
            value=MAX_DATE,
            style={'description_width': 'initial'}
        )

        self.note_widget = widgets.HTML(
            "<div style='font-size:12px; color:#555; margin-top:4px;'>"
            "<b>Note:</b> If you enter multiple patents, the engine takes the "
            "<i>average of their embeddings</i> before computing similarity."
            "</div>"
        )
        self.note_widget.layout.display = 'none'

        self.end_date.observe(self.validate_dates, names='value')
        
        # Top K results
        self.top_k = widgets.IntSlider(
            value=5,
            min=1,
            max=20,
            step=1,
            description='Max Results:',
            style={'description_width': 'initial'}
        )
        
        # Search button
        self.search_button = widgets.Button(
            description='🔍 Search Patents',
            button_style='primary',
            layout=widgets.Layout(width='200px', height='40px')
        )
        
        # Output area
        self.output = widgets.Output()
        
        # Wire up events
        self.search_method.observe(self.on_search_method_change, names='value')
        self.search_button.on_click(self.perform_search)
        
        # Initial state
        self.update_input_visibility()
    
    def on_search_method_change(self, change):
        """Handle search method change"""
        self.update_input_visibility()
    
    def update_input_visibility(self):
        """Show/hide input fields based on search method"""
        if self.search_method.value == 'Text Query':
            self.query_text.layout.display = 'block'
            self.patent_numbers.layout.display = 'none'
            self.note_widget.layout.display = 'none'
        else:
            self.query_text.layout.display = 'none'
            self.patent_numbers.layout.display = 'block'
            self.note_widget.layout.display = 'block'
    
    def display_query_info(self, query_text=None, patent_ids=None):
        """Display information about the current query"""
        if query_text:
            return f"""
            <div style="background:#e8f4f8; border-left:4px solid #2196F3; padding:15px; margin:15px 0; border-radius:5px;">
                <h4 style="margin:0; color:#1976D2;">📝 Text Query</h4>
                <p style="margin:5px 0; font-size:16px; font-style:italic;">"{query_text}"</p>
            </div>
            """
        elif patent_ids:
            # For patent queries, you might want to fetch titles
            patents_display = ""
            for patent_id in patent_ids:
                patents_display += f"<li><b>{patent_id}</b></li>"
            
            return f"""
            <div style="background:#f3e5f5; border-left:4px solid #9C27B0; padding:15px; margin:15px 0; border-radius:5px;">
                <h4 style="margin:0; color:#7B1FA2;">🔗 Patent Similarity Search</h4>
                <p style="margin:5px 0;">Finding patents similar to:</p>
                <ul style="margin:5px 0;">{patents_display}</ul>
            </div>
            """
        return ""
    
    def format_results_html(self, df, query_text=None, patent_ids=None):
        """Format search results as HTML"""
        if df.empty:
            return """
            <div style="text-align:center; padding:40px; color:#666;">
                <h3>No results found</h3>
                <p>Try adjusting your search terms or date range.</p>
            </div>
            """
        
        # Display query info
        html_output = self.display_query_info(query_text, patent_ids)
        
        # Results header
        html_output += f"""
        <div style="background:#f8f9fa; padding:15px; border-radius:8px; margin:15px 0; border:1px solid #e9ecef;">
            <h3 style="margin:0; color:#333;">🎯 Search Results ({len(df)} patents found)</h3>
        </div>
        """
        
        # Individual results
        for idx, row in df.iterrows():
            # Handle explanation formatting
            explanation_html = ""
            if isinstance(row['explanation'], list) and row['explanation']:
                for exp in row['explanation']:
                    if isinstance(exp, dict):
                        sentence = exp.get('sentence', '')
                        similarity = exp.get('similarity', 0)
                        explanation_html += f"<li>{sentence} <i>(similarity: {similarity:.3f})</i></li>"
                    else:
                        explanation_html += f"<li>{exp}</li>"
            elif isinstance(row['explanation'], str):
                explanation_html = f"<li>{row['explanation']}</li>"
            else:
                explanation_html = "<li>No explanation available</li>"
            
            # Relevance color coding
            score = row['cosine_score']
            if score >= 0.7:
                score_color = "#4CAF50"  # Green
                score_label = "High"
            elif score >= 0.5:
                score_color = "#FF9800"  # Orange
                score_label = "Medium"
            else:
                score_color = "#757575"  # Gray
                score_label = "Low"
            
            html_output += f"""
            <div style="border:1px solid #ddd; border-radius:10px; padding:20px; margin:15px 0; background:#fafafa; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="margin:0 0 10px 0; color:#333; line-height:1.4;">{row['title_en']}</h3>
                
                <div style="margin:10px 0; font-size:14px; color:#555;">
                    <span style="background:#e3f2fd; padding:4px 8px; border-radius:4px; margin-right:10px;">
                        <b>Patent:</b> {row['publication_number']}
                    </span>
                    <span style="background:#f3e5f5; padding:4px 8px; border-radius:4px; margin-right:10px;">
                        <b>Country:</b> {row['country_code']}
                    </span>
                    <span style="background:#f1f8e9; padding:4px 8px; border-radius:4px; margin-right:10px;">
                        <b>Date:</b> {row['pub_date']}
                    </span>
                    <span style="background-color:{score_color}; color:white; padding:4px 8px; border-radius:4px; font-weight:bold;">
                        {score_label} Relevance: {score:.1%}
                    </span>
                </div>
                
                <div style="margin:15px 0; padding:10px; background:white; border-radius:5px; border-left:3px solid #2196F3;">
                    <b>Abstract:</b> {row['abstract_en'][:300]}{'...' if len(row['abstract_en']) > 300 else ''}
                </div>
                
                <details style="margin-top:15px; cursor:pointer;">
                    <summary style="cursor:pointer; color:#1976D2; font-weight:bold; padding:5px 0;">
                        🔍 Why was this patent retrieved? (Click to expand)
                    </summary>
                    <div style="margin-top:10px; padding:10px; background:#f8f9fa; border-radius:5px;">
                        <ul style="margin:5px 0; padding-left:20px;">
                            {explanation_html}
                        </ul>
                    </div>
                </details>
            </div>
            """
        
        return html_output
    
    def perform_search(self, button):
        """Execute the search"""
        with self.output:
            self.output.clear_output()
            
            try:
                # Get search parameters
                start_date_str = self.start_date.value.strftime('%Y-%m-%d')
                end_date_str = self.end_date.value.strftime('%Y-%m-%d')
                top_k = self.top_k.value
                
                if self.search_method.value == 'Text Query':
                    query_text = self.query_text.value.strip()
                    if not query_text:
                        display(HTML('<div style="color:red; padding:10px;">Please enter a search query.</div>'))
                        return

                    query_text, is_valid, error_msg = sanitize_input_query(query_text)
                    if not is_valid:
                        display(HTML(f'<div style="color:red; padding:10px;">{error_msg}.</div>'))
                        return
                        # disable search
                    # Perform text search
                    results_df = self.search_function(
                        start_date=start_date_str,
                        end_date=end_date_str,
                        query_text=query_text,
                        patent_ids=None,
                        top_k=top_k
                    )
                    
                    html_output = self.format_results_html(results_df, query_text=query_text)
                
                else:  # Patent Numbers
                    patent_text = self.patent_numbers.value.strip()
                    if not patent_text:
                        display(HTML('<div style="color:red; padding:10px;">Please enter patent numbers.</div>'))
                        return
                    
                    patent_ids = [p.strip() for p in patent_text.split('\n') if p.strip()]
                    
                    # Perform patent similarity search
                    results_df = self.search_function(
                        start_date=start_date_str,
                        end_date=end_date_str,
                        query_text=None,
                        patent_ids=patent_ids,
                        top_k=top_k
                    )
                    
                    html_output = self.format_results_html(results_df, patent_ids=patent_ids)
                
                display(HTML(html_output))
                
            except Exception as e:
                display(HTML(f'<div style="color:red; padding:10px;"><b>Error:</b> {str(e)}</div>'))
    
    def display(self):
        """Display the complete interface"""
        interface_layout = widgets.VBox([
            widgets.HTML('<h2>🔍 Patent Semantic Search</h2>'),
            widgets.HTML('<p>Search for patents using natural language or find similar patents to existing ones.</p>'),
            
            self.search_method,
            self.query_text,
            self.patent_numbers,
            
            widgets.HTML('<h4 style="margin-top:20px;">Search Parameters:</h4>'),
            widgets.HBox([self.start_date, self.end_date]),
            self.top_k,
            
            widgets.HTML('<br>'),
            self.search_button,
            self.output
        ])
        
        display(interface_layout)

# Usage example:
def demo_search_interface():
    """Demo of how to use the search interface"""
    
    # Mock search function for demonstration
    def mock_search_function(start_date, end_date, query_text=None, patent_ids=None, top_k=5):
        """Mock search function that returns sample data"""
        
        # Sample data
        data = {
            "publication_number": ["CN-117476025-A", "CN-112053692-B", "CN-114038484-B"],
            "country_code": ["CN", "CN", "CN"],
            "title_en": [
                "Voice data processing method and device, electronic equipment and storage medium",
                "Speech recognition processing method, device and system",
                "Voice data processing method, device, computer equipment and storage medium"
            ],
            "abstract_en": [
                "The invention discloses a voice data processing method and device, electronic equipment and storage medium. The method comprises the following steps: acquiring voice data; extracting features from the voice data; processing the extracted features using machine learning algorithms; and outputting processed results.",
                "The invention discloses a voice recognition processing method, device and system. The method includes collecting audio signals, preprocessing the signals, applying deep learning models for recognition, and generating text output with high accuracy rates.",
                "The present application relates to a voice data processing method, device, computer equipment and storage medium. The method involves signal analysis, pattern recognition, and automated processing workflows for enhanced voice processing capabilities."
            ],
            "pub_date": ["2024-01-30", "2024-01-12", "2024-01-30"],
            "cosine_score": [0.8346, 0.8269, 0.8095],
            "explanation": [
                [{"sentence": "Voice data processing method and device for electronic equipment", "similarity": 0.91}],
                [{"sentence": "Speech recognition processing using deep learning algorithms", "similarity": 0.88}],
                [{"sentence": "Voice data processing with automated workflows and pattern recognition", "similarity": 0.86}],
            ],
        }
        
        df = pd.DataFrame(data)
        return df.head(top_k)
    
    # Create and display the interface
    search_interface = PatentSearchInterface(run_semantic_search_pipeline)
    search_interface.display()