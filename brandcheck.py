import streamlit as st
import cv2
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import plotly.express as px

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ImageWorldAnalyzer:
    def __init__(self):
        self.channels = ['brightness', 'saturation', 'contrast', 'blur']
        self.weights = {
            'brightness': 0.3,
            'contrast': 0.3,
            'saturation': 0.2,
            'blur': 0.2
        }
        # Reference values calculated from the provided consistent images
        self.reference_stats = {
            'brightness': {
                'mean': 63.97,
                'std': 12.26
            },
            'saturation': {
                'mean': 99.47,
                'std': 17.49
            },
            'contrast': {
                'mean': 50.78,
                'std': 4.98
            },
            'blur': {
                'mean': 214.40,
                'std': 49.93
            }
        }

    def analyze_image(self, image_file) -> Optional[Dict]:
        """Analyzes a single image from uploaded file."""
        try:
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                return None

            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            return {
                "filename": image_file.name,
                "brightness": float(np.mean(img_gray)),
                "saturation": float(np.mean(img_hsv[:, :, 1])),
                "contrast": float(np.std(img_gray)),
                "blur": float(np.var(cv2.Laplacian(img_gray, cv2.CV_64F)))
            }
        except Exception as e:
            st.error(f"Error analyzing image {image_file.name}: {str(e)}")
            return None

    def check_image_consistency(self, image_data: Dict) -> Tuple[bool, Dict]:
        """Checks if an image is consistent with reference data."""
        total_threshold = 2.0
        results = {}
        weighted_total_score = 0
        
        for channel in self.channels:
            value = image_data[channel]
            ref = self.reference_stats[channel]
            z_score = abs(value - ref['mean']) / ref['std']
            weighted_score = z_score * self.weights[channel]
            weighted_total_score += weighted_score
            
            results[channel] = {
                'value': value,
                'z_score': z_score,
                'weighted_score': weighted_score,
                'reference_mean': ref['mean'],
                'reference_std': ref['std']
            }
        
        results['overall_score'] = weighted_total_score
        is_consistent = weighted_total_score <= total_threshold
        
        return is_consistent, results

def main():
    st.set_page_config(page_title="BRAND CONSISTENCY CHECK", layout="wide")

    st.title("YOUR BRAND CONSISTENCY CHECK (beta)")
    st.write("Upload images to analyze their consistency with your corporate brand imagery.")

    analyzer = ImageWorldAnalyzer()
    detailed_results = []

    image_files = st.file_uploader(
        "Upload images to analyze",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Select one or more images to analyze"
    )

    if image_files:
        st.subheader("Analysis Results")
        progress_bar = st.progress(0)
        
        # Analyze each image
        for idx, img_file in enumerate(image_files):
            img_file.seek(0)
            
            # Create columns for image and results
            img_col, status_col, details_col = st.columns([2, 1, 2])
            
            with img_col:
                st.image(img_file, caption=img_file.name, use_column_width=True)
            
            image_data = analyzer.analyze_image(img_file)
            
            if image_data:
                is_consistent, consistency_results = analyzer.check_image_consistency(image_data)
                
                # Prepare detailed results for report
                detailed_result = {
                    'filename': image_data['filename'],
                    'consistency': 'konsistent' if is_consistent else 'inkonsistent',
                    'overall_score': consistency_results['overall_score']
                }
                
                # Add detailed metrics
                for channel in analyzer.channels:
                    channel_results = consistency_results[channel]
                    detailed_result[channel] = channel_results['value']
                    detailed_result[f'{channel}_score'] = channel_results['weighted_score']
                    detailed_result[f'{channel}_reference_mean'] = channel_results['reference_mean']
                    detailed_result[f'{channel}_reference_std'] = channel_results['reference_std']
                
                detailed_results.append(detailed_result)
                
                # Display simple status next to image
                with status_col:
                    st.markdown("### Status")
                    if is_consistent:
                        st.success("✅ KONSISTENT")
                    else:
                        st.error("❌ INKONSISTENT")
                
                # Display metrics visualization
                with details_col:
                    # Create fixed reference values list
                    metrics_data = []
                    for channel in analyzer.channels:
                        metrics_data.append({
                            'Metric': channel,
                            'Type': 'Auswahl',
                            'Value': consistency_results[channel]['value']
                        })
                        metrics_data.append({
                            'Metric': channel,
                            'Type': 'Bildwelt',
                            'Value': analyzer.reference_stats[channel]['mean']
                        })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    fig = px.bar(metrics_df, 
                                x='Metric', 
                                y='Value',
                                color='Type',
                                title='Metrics Comparison',
                                barmode='group',
                                color_discrete_map={
                                    'Auswahl': '#1f77b4',
                                    'Bildwelt': '#2ca02c'
                                })
                    st.plotly_chart(fig, use_container_width=True)
            
            # Update progress
            progress_bar.progress((idx + 1) / len(image_files))
            st.divider()
        
        # Summary statistics and full report
        if detailed_results:
            st.subheader("Summary")
            results_df = pd.DataFrame(detailed_results)
            
            total = len(results_df)
            consistent = len(results_df[results_df['consistency'] == 'konsistent'])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Images", total)
            col2.metric("Konsistent", consistent)
            col3.metric("Inkonsistent", total - consistent)
            
            # Full report download button
            st.write("---")
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 FULL REPORT",
                data=csv,
                file_name=f"image_analysis_full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download detailed analysis report in CSV format"
            )

if __name__ == "__main__":
    main()
